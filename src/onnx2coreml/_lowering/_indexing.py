# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Indexing and gather/scatter lowerings (Gather, Slice, Expand, Tile, ...).

These ops read their index/shape operands from constants (initializers or const
ops) wherever ONNX allows it, and recover the static input shape from the source
``Var`` (this converter requires fixed input shapes). ``Shape`` is therefore a
compile-time constant rather than a runtime ``mb.shape``.

The fiddly one is ``Slice``: ONNX only slices the listed axes and leaves the rest
full, whereas MIL ``slice_by_index`` is full-rank numpy-style slicing. We bridge
by building full-rank begin/end/stride arrays with ``begin_mask``/``end_mask`` set
on the untouched axes, and resolving each listed axis's (start, end) against the
static dim with Python's ``slice.indices`` — which implements exactly the clamping
MIL then re-applies, so the two agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb, types
from ._common import const_array, get_attr
from ._context import Lowering, LoweringContext


def _np_dtype(var: Any) -> np.dtype:
    """Return the numpy dtype backing a MIL Var, for building matching scalars."""
    return np.dtype(types.nptype_from_builtin(var.dtype))


def _gather(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Gather -> MIL ``gather``.

    MIL ``gather`` takes negative indices natively (``-D[axis] <= v < D[axis]``),
    so no normalization is needed. Indices arrive as int32 (the converter narrows
    int64 initializers); they may be a const or a runtime input — ``gather`` does
    not care.
    """
    x = ctx.values_map[node.input[0]]
    indices = ctx.values_map[node.input[1]]
    axis = int(get_attr(node, "axis", 0))
    return mb.gather(x=x, indices=indices, axis=axis, name=node.output[0])


def _gather_nd(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX GatherND -> MIL ``gather_nd``.

    ``batch_dims`` is only emitted when non-zero: it is an iOS16+ parameter, and
    omitting it (the default 0) keeps the common case building on the iOS15
    NeuralNetwork backend too.
    """
    x = ctx.values_map[node.input[0]]
    indices = ctx.values_map[node.input[1]]
    batch_dims = int(get_attr(node, "batch_dims", 0))
    kwargs: dict[str, Any] = {"x": x, "indices": indices, "name": node.output[0]}
    if batch_dims:
        kwargs["batch_dims"] = batch_dims
    return mb.gather_nd(**kwargs)


def _gather_elements(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX GatherElements -> MIL ``gather_along_axis`` (same per-axis gather)."""
    x = ctx.values_map[node.input[0]]
    indices = ctx.values_map[node.input[1]]
    axis = int(get_attr(node, "axis", 0))
    return mb.gather_along_axis(x=x, indices=indices, axis=axis, name=node.output[0])


# ONNX scatter ``reduction`` -> MIL scatter ``mode``. ONNX's default "none" is a
# plain overwrite, which MIL spells "update".
_SCATTER_MODE = {"none": "update", "add": "add", "mul": "mul", "max": "max", "min": "min"}


def _scatter_mode(node: onnx.NodeProto) -> str:
    reduction = get_attr(node, "reduction", b"none")
    reduction = reduction.decode() if isinstance(reduction, bytes) else reduction
    if reduction not in _SCATTER_MODE:
        raise ValueError(f"Scatter reduction '{reduction}' is not supported")
    return _SCATTER_MODE[reduction]


def _scatter_nd(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX ScatterND -> MIL ``scatter_nd``; ``reduction`` maps onto ``mode``."""
    data = ctx.values_map[node.input[0]]
    indices = ctx.values_map[node.input[1]]
    updates = ctx.values_map[node.input[2]]
    return mb.scatter_nd(
        data=data, indices=indices, updates=updates, mode=_scatter_mode(node),
        name=node.output[0],
    )


def _scatter_elements(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX ScatterElements -> MIL ``scatter_along_axis``."""
    data = ctx.values_map[node.input[0]]
    indices = ctx.values_map[node.input[1]]
    updates = ctx.values_map[node.input[2]]
    axis = int(get_attr(node, "axis", 0))
    return mb.scatter_along_axis(
        data=data, indices=indices, updates=updates, axis=axis, mode=_scatter_mode(node),
        name=node.output[0],
    )


def _non_zero(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX NonZero -> MIL ``non_zero``.

    MIL returns nonzero coordinates as ``(num_nonzero, rank)`` (numpy
    ``argwhere`` order); ONNX wants ``(rank, num_nonzero)``, so transpose.
    """
    x = ctx.values_map[node.input[0]]
    coords = mb.non_zero(x=x)
    return mb.transpose(x=coords, perm=[1, 0], name=node.output[0])


def _slice(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Slice (opset 10+) -> MIL ``slice_by_index``.

    ``starts``/``ends`` are required constants; ``axes``/``steps`` are optional
    constants. We emit full-rank begin/end/stride and use begin/end masks for the
    axes ONNX leaves untouched.
    """
    x = ctx.values_map[node.input[0]]
    shape = x.shape
    rank = len(shape)

    starts = const_array(ctx, node, 1)
    ends = const_array(ctx, node, 2)
    if starts is None or ends is None:
        raise ValueError("Slice requires constant 'starts' and 'ends'")
    axes_arr = const_array(ctx, node, 3)
    steps_arr = const_array(ctx, node, 4)

    axes = (
        [int(a) % rank for a in axes_arr.tolist()]
        if axes_arr is not None
        else list(range(len(starts)))
    )
    steps = steps_arr.tolist() if steps_arr is not None else [1] * len(starts)

    begin = [0] * rank
    end = [0] * rank
    stride = [1] * rank
    begin_mask = [True] * rank  # untouched axes: full slice
    end_mask = [True] * rank

    for start, stop, step, axis in zip(
        starts.tolist(), ends.tolist(), steps, axes, strict=True
    ):
        dim = shape[axis]
        # slice.indices implements ONNX's clamping of negative/out-of-range
        # start/stop against the dim, for the (forward) step.
        r_start, r_stop, _ = slice(int(start), int(stop), int(step)).indices(dim)
        begin[axis] = r_start
        stride[axis] = int(step)
        begin_mask[axis] = False
        if step < 0 and r_stop < 0:
            # Reverse to the very start: end=-1 would be read as an index, so use
            # the mask to let MIL terminate at the beginning of the axis instead.
            end_mask[axis] = True
        else:
            end[axis] = r_stop
            end_mask[axis] = False

    return mb.slice_by_index(
        x=x,
        begin=np.array(begin, dtype=np.int32),
        end=np.array(end, dtype=np.int32),
        stride=np.array(stride, dtype=np.int32),
        begin_mask=begin_mask,
        end_mask=end_mask,
        name=node.output[0],
    )


def _expand(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Expand (opset 8+) -> bidirectional broadcast to ``shape``.

    MIL has no standalone broadcast op, so broadcast against a constant of the
    numpy broadcast of ``x.shape`` and the requested ``shape`` (ONNX broadcasts
    both operands, so the result may be larger than ``shape`` itself). Numeric
    tensors multiply by ones; boolean tensors use ``logical_or`` with falses,
    since ``mul`` is not defined on bool.
    """
    x = ctx.values_map[node.input[0]]
    target = const_array(ctx, node, 1)
    if target is None:
        raise ValueError("Expand requires a constant 'shape'")
    dtype = _np_dtype(x)
    out_shape = np.broadcast_shapes(tuple(x.shape), tuple(int(s) for s in target))
    if dtype == np.bool_:
        return mb.logical_or(x=x, y=np.zeros(out_shape, dtype=np.bool_), name=node.output[0])
    return mb.mul(x=x, y=np.ones(out_shape, dtype=dtype), name=node.output[0])


def _tile(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Tile -> MIL ``tile`` (``reps`` is a const int of length ``rank``)."""
    x = ctx.values_map[node.input[0]]
    reps = const_array(ctx, node, 1)
    if reps is None:
        raise ValueError("Tile requires a constant 'repeats'")
    return mb.tile(x=x, reps=np.array(reps, dtype=np.int32), name=node.output[0])


def _shape(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Shape -> a constant int32 vector of the (static) input shape.

    Input shapes are fixed in this converter, so the shape is known at convert
    time. ``start``/``end`` (opset 15) slice the reported shape, with ONNX's
    negative-index and clamping rules (matched here via ``range``-style indexing).
    """
    x = ctx.values_map[node.input[0]]
    dims = list(x.shape)
    rank = len(dims)
    start = int(get_attr(node, "start", 0))
    end = get_attr(node, "end", None)
    end = rank if end is None else int(end)
    # Normalize negatives, then clamp to [0, rank] per ONNX Shape semantics.
    start = max(0, min(start + rank if start < 0 else start, rank))
    end = max(0, min(end + rank if end < 0 else end, rank))
    return mb.const(val=np.array(dims[start:end], dtype=np.int32), name=node.output[0])


def _constant_of_shape(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX ConstantOfShape (opset 9) -> MIL ``fill``.

    The ``value`` attribute is a 1-element tensor giving both the fill value and
    the output dtype (default float32 0). We fold it into ``fill``'s scalar value;
    its numpy dtype (narrowed for int64) carries through to the output.
    """
    shape = const_array(ctx, node, 0)
    if shape is None:
        raise ValueError("ConstantOfShape requires a constant 'input' shape")

    value = get_attr(node, "value", None)
    if value is None:
        fill_value = np.float32(0)
    else:
        arr = onnx.numpy_helper.to_array(value)
        scalar = arr.reshape(-1)[0]
        # int64 -> int32 to stay on a Core ML compute dtype.
        if arr.dtype == np.int64:
            scalar = np.int32(scalar)
        fill_value = scalar

    return mb.fill(
        shape=np.array(shape, dtype=np.int32), value=fill_value, name=node.output[0]
    )


REGISTRY: dict[str, Lowering] = {
    "Gather": _gather,
    "GatherND": _gather_nd,
    "GatherElements": _gather_elements,
    "ScatterND": _scatter_nd,
    "ScatterElements": _scatter_elements,
    "NonZero": _non_zero,
    "Slice": _slice,
    "Expand": _expand,
    "Tile": _tile,
    "Shape": _shape,
    "ConstantOfShape": _constant_of_shape,
}

__all__ = ["REGISTRY"]
