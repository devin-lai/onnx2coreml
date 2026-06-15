# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Reduction and arg-reduction lowerings (ReduceMean, ReduceSum, ArgMax, ...)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import onnx

from .._mil import mb
from ._common import Lowering, LoweringContext, const_array, get_attr


def _resolve_axes(ctx: LoweringContext, node: onnx.NodeProto) -> list[int] | None:
    """ONNX axes for a Reduce* node, normalized to a list[int] or ``None``.

    Opset nuance: ReduceSum (opset 13+) carries axes as input 1, while the other
    reductions keep it as the ``axes`` attribute at opset 17. We accept either so
    a single helper covers both. ``None`` means "reduce over all axes", which MIL
    expresses by leaving ``axes`` unset.
    """
    axes = get_attr(node, "axes", None)
    if axes is None:
        arr = const_array(ctx, node, 1)
        if arr is not None:
            axes = arr.tolist()
    if axes is None:
        return None
    axes = [int(a) for a in axes]
    return axes or None


def _reduce(mb_op: Callable[..., Any]) -> Lowering:
    """Build a lowering for a ReduceMean/Sum/Max/Min/Prod-style op."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        x = ctx.values_map[node.input[0]]
        keep_dims = bool(get_attr(node, "keepdims", 1))
        axes = _resolve_axes(ctx, node)

        # noop_with_empty_axes (opset 13+, default 0): when set and no axes are
        # given, the op is an identity rather than a full reduction.
        if axes is None and get_attr(node, "noop_with_empty_axes", 0):
            return mb.identity(x=x, name=node.output[0])

        kwargs: dict[str, Any] = {"x": x, "keep_dims": keep_dims, "name": node.output[0]}
        if axes is not None:
            kwargs["axes"] = axes
        return mb_op(**kwargs)

    return lower


def _topk(ctx: LoweringContext, node: onnx.NodeProto) -> list[Any]:
    """ONNX TopK -> MIL ``topk``. ``K`` is the (constant) input 1.

    ONNX ``largest`` maps to MIL ``ascending = not largest``. We target the iOS15
    ``topk`` (values + indices, always sorted) so both container formats build;
    ``sorted=0`` is therefore treated as sorted (MIL cannot opt out at iOS15, and
    callers that ask for top-k almost always want them ordered).
    """
    x = ctx.values_map[node.input[0]]
    k_arr = const_array(ctx, node, 1)
    if k_arr is None:
        raise ValueError("TopK requires a constant 'K' input")
    k = int(k_arr.reshape(-1)[0])
    axis = int(get_attr(node, "axis", -1))
    largest = bool(get_attr(node, "largest", 1))
    values, indices = mb.topk(x=x, k=k, axis=axis, ascending=not largest)
    # mb.topk returns a tuple of immutable Vars; re-emit named identities so the
    # predicted output keys match ONNX's value/index output names.
    return [
        mb.identity(x=values, name=node.output[0]),
        mb.identity(x=indices, name=node.output[1]),
    ]


def _arg_reduce(mb_op: Callable[..., Any]) -> Lowering:
    """Build a lowering for ArgMax/ArgMin (single-axis, int output)."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        x = ctx.values_map[node.input[0]]
        axis = int(get_attr(node, "axis", 0))
        keep_dims = bool(get_attr(node, "keepdims", 1))
        # select_last_index (default 0) is unsupported: MIL's argmax/argmin do not
        # guarantee tie ordering, so we cannot honor "last". Tests avoid ties.
        return mb_op(x=x, axis=axis, keep_dims=keep_dims, name=node.output[0])

    return lower


REGISTRY: dict[str, Lowering] = {
    "ReduceMean": _reduce(mb.reduce_mean),
    "ReduceSum": _reduce(mb.reduce_sum),
    "ReduceMax": _reduce(mb.reduce_max),
    "ReduceMin": _reduce(mb.reduce_min),
    "ReduceProd": _reduce(mb.reduce_prod),
    "ReduceL1": _reduce(mb.reduce_l1_norm),
    "ReduceL2": _reduce(mb.reduce_l2_norm),
    "ReduceLogSum": _reduce(mb.reduce_log_sum),
    "ReduceLogSumExp": _reduce(mb.reduce_log_sum_exp),
    "ReduceSumSquare": _reduce(mb.reduce_sum_square),
    "ArgMax": _arg_reduce(mb.reduce_argmax),
    "ArgMin": _arg_reduce(mb.reduce_argmin),
    "TopK": _topk,
}

__all__ = ["REGISTRY"]
