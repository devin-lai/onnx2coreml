# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""ONNX dtype <-> MIL dtype mapping and the 64->32-bit narrowing policy.

Core ML compute is 32-bit. ONNX models routinely carry int64 (shapes, indices)
and float64 constants that Core ML cannot execute, so we narrow them on the way
in. Narrowing is explicit and centralized here so the policy is auditable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto

from ._mil import types as mil_types
from .errors import ConversionError

# ONNX elem_type -> MIL builtin type, with 64->32 narrowing already applied.
_ONNX_TO_MIL: dict[int, Any] = {
    TensorProto.FLOAT: mil_types.fp32,
    TensorProto.FLOAT16: mil_types.fp16,
    TensorProto.DOUBLE: mil_types.fp32,  # narrow
    TensorProto.INT32: mil_types.int32,
    TensorProto.INT64: mil_types.int32,  # narrow
    TensorProto.BOOL: mil_types.bool,
}


def onnx_dtype_to_mil(elem_type: int):
    """Map an ONNX ``TensorProto`` elem_type to a MIL builtin type.

    Raises :class:`ConversionError` for dtypes onnx2coreml does not yet support
    (rather than silently producing a wrong-typed model).
    """
    mil = _ONNX_TO_MIL.get(elem_type)
    if mil is None:
        name = TensorProto.DataType.Name(elem_type) if elem_type else "UNDEFINED"
        raise ConversionError(
            "<input>",
            "<dtype>",
            ValueError(f"ONNX dtype {name} is not supported yet"),
        )
    return mil


# numpy dtype -> narrowed numpy dtype for initializer constants.
_NARROW = {
    np.dtype(np.int64): np.int32,
    np.dtype(np.uint64): np.int32,
    np.dtype(np.uint32): np.int32,
    np.dtype(np.float64): np.float32,
}


def narrow_array(arr: np.ndarray, *, context: str = "value") -> np.ndarray:
    """Narrow a 64-bit numpy array to the matching 32-bit Core ML dtype.

    int64/uint64/uint32 -> int32, float64 -> float32. Other dtypes pass through.

    Out-of-range integers are **saturated** to the int32 range rather than
    rejected: ONNX exporters routinely emit ``INT64_MAX`` / ``INT64_MIN`` as
    open-ended sentinels (e.g. a Slice "to the end" bound, or a mask fill), and a
    saturated sentinel is still an out-of-bounds index that Core ML clamps to the
    real dimension — so the semantics are preserved on a 32-bit runtime.
    """
    target = _NARROW.get(arr.dtype)
    if target is None:
        return arr
    if target is np.int32:
        info = np.iinfo(np.int32)
        if arr.size and (arr.min() < info.min or arr.max() > info.max):
            arr = np.clip(arr, info.min, info.max)
    return arr.astype(target)


# Largest finite float16. Constants beyond this become +/-inf once Core ML casts
# the program to fp16, which is catastrophic: an attention-mask fill of
# ``-FLT_MAX`` turns into ``-inf``, and ``(1 - mask) * -inf`` is ``0 * -inf = NaN``
# for the *unmasked* positions, poisoning the whole softmax.
_FP16_MAX = 65504.0


def saturate_fp16(arr: np.ndarray) -> np.ndarray:
    """Clamp a float array to the float16 range so fp16 conversion cannot make it
    ``+/-inf``.

    Only values that would otherwise overflow are touched (real weights are far
    inside the range), and a saturated sentinel keeps its meaning — ``-65504`` is
    still "masked out" in a softmax, just without the ``NaN``. A no-op for
    non-float arrays.
    """
    if arr.dtype.kind != "f" or not arr.size:
        return arr
    if arr.max() > _FP16_MAX or arr.min() < -_FP16_MAX:
        return np.clip(arr, -_FP16_MAX, _FP16_MAX).astype(arr.dtype)
    return arr
