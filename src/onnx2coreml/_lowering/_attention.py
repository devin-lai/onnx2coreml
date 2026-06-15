# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Attention lowering (ScaledDotProductAttention), produced by the fusion pass.

The fused node carries inputs ``[Q, K, V]`` and a float ``scale`` attribute. It
lowers to the same MIL primitives the decomposed graph would have, so it runs on
both backends: ``matmul`` (with ``transpose_y`` for ``K^T``), ``mul`` for the
scale, ``softmax`` over the last axis, then ``matmul`` against ``V``. Every op
used exists at iOS15, so no native iOS18 ``scaled_dot_product_attention`` op is
required and the ``.mlmodel`` (NeuralNetwork) backend is supported too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb, types
from ._common import get_attr, operands
from ._context import Lowering, LoweringContext


def _scaled_dot_product_attention(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    q, k, v = operands(ctx.values_map, node, [0, 1, 2])
    # scores = Q @ K^T (transpose only the last two dims of K, matching the
    # decomposed Transpose(K) the fusion replaced).
    scores = mb.matmul(x=q, y=k, transpose_y=True)
    # Scale by the constant the original graph used (1/sqrt(d) for a Mul, or the
    # reciprocal of a Div's divisor). Use Q's dtype so fp16 graphs stay fp16.
    scale = np.dtype(types.nptype_from_builtin(q.dtype)).type(get_attr(node, "scale", 1.0))
    scaled = mb.mul(x=scores, y=scale)
    probs = mb.softmax(x=scaled, axis=-1)
    return mb.matmul(x=probs, y=v, name=node.output[0])


REGISTRY: dict[str, Lowering] = {
    "ScaledDotProductAttention": _scaled_dot_product_attention,
}

__all__ = ["REGISTRY"]
