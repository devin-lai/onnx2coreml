# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test harness: build single-op ONNX models, convert, run reference + Core ML,
compare.

Parity tests pin **fp32 compute on CPU_ONLY** so that a mismatch means a real
lowering bug, not fp16 rounding. Where the Core ML runtime is unavailable, the
model is still built and serialized (structural check) and the numeric compare
is skipped.
"""

from __future__ import annotations

import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import onnx2coreml as o2c
from onnx2coreml._types import narrow_array


def _has_coreml_runtime() -> bool:
    return platform.system() == "Darwin"


_PREDICT_LOCK_PATH = Path(tempfile.gettempdir()) / "onnx2coreml_predict.lock"


@contextmanager
def _predict_lock():
    """Serialize Core ML predictions across concurrent test processes.

    The Core ML runtime races on concurrent model loads/compiles, which
    surfaces as transient, converter-unrelated failures under pytest-xdist
    or parallel workers. An exclusive file lock makes prediction safe
    regardless of worker count.
    """
    if platform.system() != "Darwin":
        yield
        return
    import fcntl

    with _PREDICT_LOCK_PATH.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


requires_predict = pytest.mark.skipif(
    not _has_coreml_runtime(),
    reason="MLModel.predict requires the macOS Core ML runtime",
)

_NP_TO_ONNX = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.bool_): TensorProto.BOOL,
}


def single_op_model(
    op_type: str,
    inputs: dict[str, np.ndarray],
    n_outputs: int = 1,
    *,
    attrs: dict[str, Any] | None = None,
    initializers: dict[str, np.ndarray] | None = None,
    opset: int = 17,
) -> onnx.ModelProto:
    """Build a one-node ONNX model; output shapes/dtypes filled by inference."""
    initializers = initializers or {}
    input_vis = [
        helper.make_tensor_value_info(n, _NP_TO_ONNX[a.dtype], a.shape)
        for n, a in inputs.items()
    ]
    output_names = [f"out{i}" for i in range(n_outputs)]
    node = helper.make_node(
        op_type,
        inputs=list(inputs) + list(initializers),
        outputs=output_names,
        **(attrs or {}),
    )
    graph = helper.make_graph(
        [node],
        f"test_{op_type}",
        input_vis,
        [
            helper.make_tensor_value_info(n, TensorProto.UNDEFINED, None)
            for n in output_names
        ],
        initializer=[
            onnx.numpy_helper.from_array(a, name=n) for n, a in initializers.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def run_onnxruntime(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, inputs)


def _predict(mlmodel, model: onnx.ModelProto, inputs: dict[str, np.ndarray]):
    """Run MLModel.predict, aligning ONNX names to (possibly sanitized) Core ML
    feature names by position, and returning outputs in graph-output order."""
    spec = mlmodel.get_spec()
    init = {i.name for i in model.graph.initializer}
    onnx_inputs = [i.name for i in model.graph.input if i.name not in init]
    feed = {
        cm.name: inputs[onnx_name]
        for cm, onnx_name in zip(spec.description.input, onnx_inputs, strict=True)
    }
    with _predict_lock():
        out = mlmodel.predict(feed)
    return [np.asarray(out[o.name]) for o in spec.description.output]


def assert_parity(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    fmt: str = "mlpackage",
    fuse: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Convert ``model`` to ``fmt`` and assert numerical parity with ONNX Runtime.

    Off-device (no Core ML runtime), the model is still built + serialized and
    the numeric comparison is skipped.
    """
    expected = run_onnxruntime(model, inputs)
    mlmodel = o2c.convert(
        model, format=fmt, fuse=fuse, compute_precision="fp32", compute_units="cpu_only"
    )
    # Structural check: the spec exists and is serializable.
    assert mlmodel.get_spec() is not None

    if not _has_coreml_runtime():
        return

    got = _predict(mlmodel, model, inputs)
    assert len(got) == len(expected), "output count mismatch"
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        e = narrow_array(np.asarray(e), context=f"reference output {i}")
        g = np.asarray(g)
        assert g.shape == e.shape, (
            f"output {i} shape mismatch: got {g.shape}, want {e.shape}"
        )
        if e.dtype.kind in "biu":
            np.testing.assert_array_equal(g, e, err_msg=f"output {i} mismatch")
        else:
            np.testing.assert_allclose(
                g.astype(np.float64),
                e.astype(np.float64),
                rtol=rtol,
                atol=atol,
                err_msg=f"output {i} mismatch",
            )
