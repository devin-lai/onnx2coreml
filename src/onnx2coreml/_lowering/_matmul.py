# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Linear-algebra lowerings (MatMul, Gemm)."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb
from ._common import const_array, get_attr, operands
from ._context import Lowering, LoweringContext


def _matmul(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # mb.matmul follows numpy matmul semantics (N-D batching, broadcasting,
    # 1-D vector promotion), matching ai.onnx MatMul exactly.
    x, y = operands(ctx.values_map, node, [0, 1])
    return mb.matmul(x=x, y=y, name=node.output[0])


def _gemm(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # Y = alpha * (A' @ B') + beta * C, with A'/B' optionally transposed.
    a, b, c = operands(ctx.values_map, node, [0, 1, 2])
    alpha = float(get_attr(node, "alpha", 1.0))
    beta = float(get_attr(node, "beta", 1.0))
    trans_a = bool(get_attr(node, "transA", 0))
    trans_b = bool(get_attr(node, "transB", 0))

    name = node.output[0]
    has_bias = c is not None
    scale = alpha != 1.0

    # The op producing the final value must carry the node's output name; passing
    # name=None is rejected by the builder, so only set it on the last op.
    mm_kw = {} if scale or has_bias else {"name": name}
    out = mb.matmul(x=a, y=b, transpose_x=trans_a, transpose_y=trans_b, **mm_kw)
    if scale:
        mul_kw = {} if has_bias else {"name": name}
        out = mb.mul(x=out, y=alpha, **mul_kw)
    if has_bias:
        bias = c if beta == 1.0 else mb.mul(x=c, y=beta)
        out = mb.add(x=out, y=bias, name=name)
    return out


def _inverse(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """``com.microsoft::Inverse`` (batched matrix inverse over the last two dims).

    Core ML has no matrix-inverse op. A constant operand is folded with numpy at
    convert time. A runtime operand is handled for the common 2x2 case (e.g.
    keypoint Jacobians) with the closed-form inverse
    ``[[a, b], [c, d]]^-1 = 1/(ad - bc) * [[d, -b], [-c, a]]``; larger runtime
    matrices raise.
    """
    mat = const_array(ctx, node, 0)
    if mat is not None:
        inv = np.linalg.inv(mat.astype(np.float64)).astype(np.float32)
        return mb.const(val=np.ascontiguousarray(inv), name=node.output[0])

    m = ctx.values_map[node.input[0]]
    batch = [int(d) for d in m.shape[:-2]]
    if [int(d) for d in m.shape[-2:]] != [2, 2]:
        raise ValueError(f"runtime Inverse supports only 2x2 matrices, got {list(m.shape[-2:])}")
    a, b, c, d = mb.split(x=mb.reshape(x=m, shape=[*batch, 4]), num_splits=4, axis=-1)
    inv_det = mb.inverse(x=mb.sub(x=mb.mul(x=a, y=d), y=mb.mul(x=b, y=c)))
    cols = [
        mb.mul(x=d, y=inv_det),
        mb.mul(x=mb.mul(x=b, y=-1.0), y=inv_det),
        mb.mul(x=mb.mul(x=c, y=-1.0), y=inv_det),
        mb.mul(x=a, y=inv_det),
    ]
    return mb.reshape(x=mb.concat(values=cols, axis=-1), shape=[*batch, 2, 2], name=node.output[0])


REGISTRY: dict[str, Lowering] = {
    "MatMul": _matmul,
    "Gemm": _gemm,
    "com.microsoft::Inverse": _inverse,
}

__all__ = ["REGISTRY"]
