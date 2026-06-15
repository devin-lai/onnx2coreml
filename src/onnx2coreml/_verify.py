# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity verification against ONNX Runtime.

After a model is converted, ``verify`` runs the same inputs through ONNX Runtime
(the reference) and the Core ML ``MLModel`` and reports per-output error metrics.
This is the evidence that a conversion is faithful, not merely structurally
well-formed.

ONNX Runtime is an optional dependency; it is imported lazily so the converter
itself never depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from ._io import load
from ._types import narrow_array
from .errors import Onnx2CoreMLError

# Fixed seed so generated inputs (and therefore reported metrics) are
# reproducible across runs and machines.
_SEED = 0x0117C0DE


@dataclass
class OutputMetrics:
    """Error metrics for a single model output."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    max_abs_err: float
    max_rel_err: float
    psnr: float

    def as_dict(self) -> dict:
        # PSNR is ``inf`` for an exact match; JSON has no inf literal, so emit a
        # string there to keep the payload standard-conformant.
        psnr = "inf" if np.isinf(self.psnr) else self.psnr
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "psnr": psnr,
        }


@dataclass
class VerifyReport:
    """Outcome of comparing a Core ML model against the ONNX reference.

    ``passed`` is the bottom line: every output is within tolerance (and, when
    requested, above ``min_psnr``). ``outputs`` carries the per-output metrics.
    """

    outputs: list[OutputMetrics] = field(default_factory=list)
    passed: bool = False
    rtol: float = 1e-3
    atol: float = 1e-4
    min_psnr: float | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "rtol": self.rtol,
            "atol": self.atol,
            "min_psnr": self.min_psnr,
            "outputs": [o.as_dict() for o in self.outputs],
        }

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        crit = f"rtol={self.rtol:g} atol={self.atol:g}"
        if self.min_psnr is not None:
            crit += f" min_psnr={self.min_psnr:g}"
        lines = [f"verify: {status} ({crit})"]
        for o in self.outputs:
            psnr = "inf" if np.isinf(o.psnr) else f"{o.psnr:.2f} dB"
            lines.append(
                f"  {o.name} {tuple(o.shape)} {o.dtype}: "
                f"max_abs={o.max_abs_err:.3e} max_rel={o.max_rel_err:.3e} psnr={psnr}"
            )
        return "\n".join(lines)


def generate_inputs(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Seeded random inputs for every non-initializer graph input.

    Floats are drawn from ``[0, 1)``; integer inputs get small non-negative
    values (safe as indices/counts). Shapes must be static — the converter
    already requires that. Determinism comes from a fixed RNG seed so reported
    metrics are reproducible.
    """
    rng = np.random.default_rng(_SEED)
    init = {i.name for i in model.graph.initializer}
    feed: dict[str, np.ndarray] = {}
    for vi in model.graph.input:
        if vi.name in init:
            continue
        feed[vi.name] = _random_for(vi, rng)
    return feed


def _random_for(vi: onnx.ValueInfoProto, rng: np.random.Generator) -> np.ndarray:
    tt = vi.type.tensor_type
    shape: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            shape.append(d.dim_value)
        else:
            raise Onnx2CoreMLError(
                f"input '{vi.name}' has a dynamic or unknown dimension; "
                "pass explicit `inputs` to verify a model with dynamic shapes"
            )
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
    kind = np.dtype(np_dtype).kind
    if kind == "f":
        return rng.random(shape, dtype=np.float32).astype(np_dtype)
    if kind == "b":
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if kind in "iu":
        # Small non-negative ints are valid as indices/gather/counts.
        return rng.integers(0, 4, size=shape).astype(np_dtype)
    raise Onnx2CoreMLError(
        f"input '{vi.name}' has dtype {np_dtype} which verify cannot synthesize; "
        "pass explicit `inputs`"
    )


def _onnxruntime():
    """Import onnxruntime lazily, raising a clear error if it is missing."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only without ort
        raise Onnx2CoreMLError(
            "numerical verification requires onnxruntime; install it with "
            "`pip install onnx2coreml[verify]` (or `pip install onnxruntime`)"
        ) from exc
    return onnxruntime


def _run_reference(
    onnx_model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    ort = _onnxruntime()
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, inputs)


def _run_coreml(
    mlmodel, onnx_model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Run ``MLModel.predict``, aligning ONNX names to (possibly sanitized) Core
    ML feature names by position, and returning outputs in graph-output order.

    Mirrors ``tests/helpers._predict``: Core ML may sanitize feature names, so we
    key the feed by ``spec.description.input[i].name`` and read outputs by
    ``spec.description.output[i].name``.
    """
    spec = mlmodel.get_spec()
    init = {i.name for i in onnx_model.graph.initializer}
    onnx_inputs = [i.name for i in onnx_model.graph.input if i.name not in init]
    feed = {
        cm.name: inputs[onnx_name]
        for cm, onnx_name in zip(spec.description.input, onnx_inputs, strict=True)
    }
    out = mlmodel.predict(feed)
    return [np.asarray(out[o.name]) for o in spec.description.output]


def _metrics(name: str, expected: np.ndarray, got: np.ndarray) -> OutputMetrics:
    """Per-output error metrics. Integer outputs are compared exactly."""
    e64 = expected.astype(np.float64)
    g64 = got.astype(np.float64)
    diff = np.abs(g64 - e64)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(e64), 1e-12)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    psnr = _psnr(e64, g64)
    return OutputMetrics(
        name=name,
        shape=tuple(expected.shape),
        dtype=str(expected.dtype),
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        psnr=psnr,
    )


def _psnr(expected: np.ndarray, got: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB; ``inf`` when the outputs are identical."""
    mse = float(np.mean((got - expected) ** 2)) if expected.size else 0.0
    if mse == 0.0:
        return float("inf")
    peak = float(np.max(np.abs(expected)))
    if peak == 0.0:
        peak = 1.0
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


def verify_model(
    onnx_model: onnx.ModelProto,
    mlmodel,
    *,
    inputs: dict[str, np.ndarray] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    min_psnr: float | None = None,
) -> VerifyReport:
    """Compare ``mlmodel`` predictions against the ONNX reference.

    Runs ``inputs`` (seeded random if ``None``) through ONNX Runtime and the Core
    ML model, then computes per-output max abs/rel error and PSNR. Float outputs
    pass when ``|got - ref| <= atol + rtol*|ref|`` elementwise (and, if given,
    PSNR >= ``min_psnr``); integer outputs must match exactly.

    The reference is narrowed via :func:`onnx2coreml._types.narrow_array` so int64
    / float64 references line up with Core ML's 32-bit outputs.
    """
    if inputs is None:
        inputs = generate_inputs(onnx_model)

    expected = _run_reference(onnx_model, inputs)
    got = _run_coreml(mlmodel, onnx_model, inputs)
    if len(got) != len(expected):
        raise Onnx2CoreMLError(
            f"output count mismatch: Core ML produced {len(got)}, "
            f"ONNX reference produced {len(expected)}"
        )

    spec = mlmodel.get_spec()
    out_names = [o.name for o in spec.description.output]
    report = VerifyReport(rtol=rtol, atol=atol, min_psnr=min_psnr)
    passed = True
    for name, e, g in zip(out_names, expected, got, strict=True):
        e = narrow_array(np.asarray(e), context=f"reference output '{name}'")
        g = np.asarray(g)
        m = _metrics(name, e, g)
        report.outputs.append(m)

        if g.shape != e.shape:
            passed = False
            continue
        if e.dtype.kind in "biu":
            if not np.array_equal(g, e):
                passed = False
        else:
            within = np.allclose(
                g.astype(np.float64), e.astype(np.float64), rtol=rtol, atol=atol
            )
            if not within:
                passed = False
            if min_psnr is not None and m.psnr < min_psnr:
                passed = False
    report.passed = passed
    return report


def verify(
    onnx_model: onnx.ModelProto | str | Path | bytes,
    coreml_model,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    min_psnr: float | None = None,
) -> VerifyReport:
    """Verify a Core ML model against its ONNX source with seeded random inputs.

    Parameters
    ----------
    onnx_model:
        ONNX model as a proto, file path, or serialized bytes.
    coreml_model:
        A coremltools ``MLModel`` object, or a path to a saved ``.mlpackage`` /
        ``.mlmodel`` to load.
    rtol, atol:
        Float tolerance: a model passes when every float output satisfies
        ``|got - ref| <= atol + rtol*|ref|`` elementwise.
    min_psnr:
        Optional minimum PSNR (dB) required of every float output.

    Returns
    -------
    :class:`VerifyReport`.

    Requires ONNX Runtime; raises :class:`onnx2coreml.errors.Onnx2CoreMLError`
    if it is not installed.
    """
    onnx_proto = load(onnx_model)
    mlmodel = _load_mlmodel(coreml_model)
    return verify_model(
        onnx_proto, mlmodel, rtol=rtol, atol=atol, min_psnr=min_psnr
    )


def _load_mlmodel(coreml_model):
    """Return ``coreml_model`` if it is already an MLModel, else load it by path.

    The coremltools import funnels through the ``_mil`` seam, consistent with the
    rest of the package.
    """
    if isinstance(coreml_model, (str, Path)):
        from ._mil import ct

        return ct.models.MLModel(str(coreml_model))
    return coreml_model
