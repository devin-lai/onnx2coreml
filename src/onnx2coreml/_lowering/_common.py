# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Reusable lowering primitives shared across operator families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._utils import get_attr, operands
from ._context import Lowering, LoweringContext

__all__ = [
    "Lowering",
    "LoweringContext",
    "binary",
    "const_array",
    "get_attr",
    "operands",
    "unary",
]


def binary(mb_op: Callable[..., Any]) -> Lowering:
    """Build a lowering for a two-input, broadcasting elementwise op.

    The result Var is named with the node's output name so Core ML's predicted
    output keys line up with the ONNX graph's output names.
    """

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        x, y = operands(ctx.values_map, node, [0, 1])
        return mb_op(x=x, y=y, name=node.output[0])

    return lower


def unary(mb_op: Callable[..., Any]) -> Lowering:
    """Build a lowering for a single-input elementwise op."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        (x,) = operands(ctx.values_map, node, [0])
        return mb_op(x=x, name=node.output[0])

    return lower


def const_array(ctx: LoweringContext, node: onnx.NodeProto, idx: int) -> np.ndarray | None:
    """Return the constant numpy value of input ``idx`` if it is known at
    convert time (an initializer or a const op), else ``None``.

    Ops like Reshape/Slice/Reduce read shapes/axes from constant inputs; this is
    how they recover those values from the MIL graph.
    """
    if idx >= len(node.input) or not node.input[idx]:
        return None
    var = ctx.values_map.get(node.input[idx])
    val = getattr(var, "val", None)
    if val is None:
        return None
    return np.asarray(val)
