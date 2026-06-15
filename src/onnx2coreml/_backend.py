# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The MIL-Program -> MLModel serialization seam.

Together with ``_mil``, this is the only place that touches coremltools'
conversion entry point. Format/spec evolution is absorbed here.
"""

from __future__ import annotations

from ._mil import ct
from ._target import Format

_CONVERT_TO = {Format.MLPACKAGE: "mlprogram", Format.MLMODEL: "neuralnetwork"}


def program_to_mlmodel(
    prog, *, fmt: Format, precision=None, target=None, compute_units=None, fp32_op_types=None
):
    """Serialize a MIL ``Program`` to a coremltools ``MLModel`` for ``fmt``.

    ML Program honors ``minimum_deployment_target`` and ``compute_precision``;
    NeuralNetwork does not take those, so they are omitted for that path.

    ``fp32_op_types`` (only meaningful for fp16 ML Program) keeps the named MIL op
    types in fp32 while the rest run fp16 — the lever for numerically sensitive
    subgraphs (softmax/layer_norm, geometric coordinate math, detection heads)
    that lose too much in pure fp16.
    """
    kwargs: dict = {"convert_to": _CONVERT_TO[fmt]}
    if fmt is Format.MLPACKAGE:
        if target is not None:
            kwargs["minimum_deployment_target"] = target
        if precision is not None:
            kwargs["compute_precision"] = _compute_precision(precision, fp32_op_types)
    if compute_units is not None:
        kwargs["compute_units"] = compute_units
    return ct.convert(prog, **kwargs)


def _compute_precision(precision, fp32_op_types):
    """Resolve a coremltools ``compute_precision``: the plain fp16/fp32 enum, or —
    when keeping select op types in fp32 under fp16 — an ``FP16ComputePrecision``
    whose op selector excludes them."""
    if not fp32_op_types or precision != ct.precision.FLOAT16:
        return precision
    keep = set(fp32_op_types)
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep)
