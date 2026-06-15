# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Activation-function lowerings (Relu, Sigmoid, Gelu, Softmax, ...)."""

from __future__ import annotations

from typing import Any

import onnx

from .._mil import mb
from ._common import get_attr, operands, unary
from ._context import Lowering, LoweringContext


def _leaky_relu(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    (x,) = operands(ctx.values_map, node, [0])
    return mb.leaky_relu(x=x, alpha=get_attr(node, "alpha", 0.01), name=node.output[0])


def _elu(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    (x,) = operands(ctx.values_map, node, [0])
    return mb.elu(x=x, alpha=get_attr(node, "alpha", 1.0), name=node.output[0])


def _prelu(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX PRelu(x) = x if x >= 0 else slope * x, with the slope (input[1])
    # unidirectionally broadcast against x. MIL's prelu op only models a
    # per-channel slope, so lower via the exact elementwise identity
    # relu(x) + slope * min(x, 0) instead — this handles every valid slope shape
    # (scalar, per-channel, or higher-rank) with MIL's native broadcasting.
    x, slope = operands(ctx.values_map, node, [0, 1])
    neg = mb.minimum(x=x, y=0.0)
    return mb.add(x=mb.relu(x=x), y=mb.mul(x=neg, y=slope), name=node.output[0])


def _gelu(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    (x,) = operands(ctx.values_map, node, [0])
    approximate = get_attr(node, "approximate", "none")
    mode = "TANH_APPROXIMATION" if approximate == "tanh" else "EXACT"
    return mb.gelu(x=x, mode=mode, name=node.output[0])


def _softmax(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # opset>=13: axis is the reduction axis (default -1).
    (x,) = operands(ctx.values_map, node, [0])
    return mb.softmax(x=x, axis=get_attr(node, "axis", -1), name=node.output[0])


def _log_softmax(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    (x,) = operands(ctx.values_map, node, [0])
    sm = mb.softmax(x=x, axis=get_attr(node, "axis", -1))
    return mb.log(x=sm, name=node.output[0])


def _hard_sigmoid(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX HardSigmoid = max(0, min(1, alpha*x + beta)); MIL sigmoid_hard matches.
    (x,) = operands(ctx.values_map, node, [0])
    alpha = get_attr(node, "alpha", 0.2)
    beta = get_attr(node, "beta", 0.5)
    return mb.sigmoid_hard(x=x, alpha=alpha, beta=beta, name=node.output[0])


def _hard_swish(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX HardSwish(x) = x * max(0, min(1, x/6 + 0.5)).
    (x,) = operands(ctx.values_map, node, [0])
    affine = mb.add(x=mb.mul(x=x, y=1.0 / 6.0), y=0.5)
    clamped = mb.clip(x=affine, alpha=0.0, beta=1.0)
    return mb.mul(x=x, y=clamped, name=node.output[0])


REGISTRY: dict[str, Lowering] = {
    "Relu": unary(mb.relu),
    "Sigmoid": unary(mb.sigmoid),
    "Tanh": unary(mb.tanh),
    "Softplus": unary(mb.softplus),
    "LeakyRelu": _leaky_relu,
    "PRelu": _prelu,
    "Elu": _elu,
    "Gelu": _gelu,
    "Softmax": _softmax,
    "LogSoftmax": _log_softmax,
    "HardSigmoid": _hard_sigmoid,
    "HardSwish": _hard_swish,
}

__all__ = ["REGISTRY"]
