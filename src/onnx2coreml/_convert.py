# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The user-facing ``convert`` entry point."""

from __future__ import annotations

from pathlib import Path

import onnx

from ._backend import program_to_mlmodel
from ._io import load
from ._target import (
    Format,
    resolve_compute_units,
    resolve_precision,
    resolve_target,
)
from .converter import Converter
from .errors import Onnx2CoreMLError


def convert(
    model: onnx.ModelProto | str | Path | bytes,
    *,
    format: str | Format = "mlpackage",
    minimum_deployment_target: str | None = None,
    compute_precision: str | None = None,
    compute_units: str | None = None,
    fp32_op_types=None,
    fuse: bool = True,
    verify: bool = False,
):
    """Convert an ONNX model to a Core ML ``MLModel``.

    Parameters
    ----------
    model:
        ONNX model as a proto, file path, or serialized bytes.
    format:
        ``"mlpackage"`` (ML Program, default) or ``"mlmodel"`` (NeuralNetwork).
    minimum_deployment_target:
        e.g. ``"iOS17"``. Defaults to iOS17 for ML Program.
    compute_precision:
        ``"fp16"`` (default for ML Program) or ``"fp32"``.
    compute_units:
        ``"all"`` (default), ``"cpu_only"``, ``"cpu_and_gpu"``, ``"cpu_and_ne"``.
    fp32_op_types:
        Optional iterable of MIL op-type names to keep in fp32 while the rest of
        the model runs fp16 (e.g. ``{"softmax", "layer_norm"}``). The escape hatch
        for numerically sensitive subgraphs that lose accuracy in pure fp16; only
        applies to fp16 ML Program output.
    fuse:
        Enable semantics-preserving graph fusion passes.
    verify:
        Run numerical parity against ONNX Runtime after conversion.

    Returns
    -------
    coremltools ``MLModel`` — call ``.save(path)`` to write ``.mlpackage`` /
    ``.mlmodel`` to disk.
    """
    fmt = Format.from_str(format)
    onnx_model = load(model)
    target = resolve_target(minimum_deployment_target, fmt)
    precision = resolve_precision(compute_precision, fmt)
    units = resolve_compute_units(compute_units)

    program = Converter().to_mil(onnx_model, target=target, precision=precision, fuse=fuse)
    mlmodel = program_to_mlmodel(
        program, fmt=fmt, precision=precision, target=target, compute_units=units,
        fp32_op_types=fp32_op_types,
    )

    if verify:
        from ._verify import verify_model

        report = verify_model(onnx_model, mlmodel)
        if not report.passed:
            raise Onnx2CoreMLError(f"numerical verification failed: {report}")
    return mlmodel
