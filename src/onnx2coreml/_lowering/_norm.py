# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Normalization lowerings (BatchNorm, LayerNorm, InstanceNorm, GroupNorm).

BatchNorm/LayerNorm/InstanceNorm map one-to-one onto the MIL ``batch_norm`` /
``layer_norm`` / ``instance_norm`` ops. MIL has no group-normalization op, so
:func:`_group_norm` composes one: reshape the channel axis into
``(num_groups, channels_per_group * spatial)``, layer-normalize the trailing
per-group axis, reshape back, then apply the affine scale/bias. ONNX
GroupNormalization carries scale/bias of shape ``(num_groups,)`` at opset 18 and
``(C,)`` at opset 21; both layouts are handled by broadcasting the per-channel or
per-group affine over the restored ``(N, C, *D)`` tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb
from ._common import get_attr, operands
from ._context import Lowering, LoweringContext


def _batch_norm(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX inputs: X, scale, B, input_mean, input_var (all per-channel, axis=1).
    x, scale, b, mean, var = operands(ctx.values_map, node, [0, 1, 2, 3, 4])
    epsilon = get_attr(node, "epsilon", 1e-5)
    return mb.batch_norm(
        x=x, mean=mean, variance=var, gamma=scale, beta=b, epsilon=epsilon,
        name=node.output[0],
    )


def _instance_norm(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX inputs: X, scale, B (per-channel); normalize each (N,C) over spatial.
    x, scale, b = operands(ctx.values_map, node, [0, 1, 2])
    epsilon = get_attr(node, "epsilon", 1e-5)
    return mb.instance_norm(
        x=x, gamma=scale, beta=b, epsilon=epsilon, name=node.output[0]
    )


def _layer_norm(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX (opset 17): normalize over axes [axis .. rank-1]; Scale/B are shaped
    # like those normalized axes, which is exactly what mb.layer_norm wants.
    x, scale, b = operands(ctx.values_map, node, [0, 1, 2])
    epsilon = get_attr(node, "epsilon", 1e-5)
    rank = x.rank
    axis = get_attr(node, "axis", -1)
    if axis < 0:
        axis += rank
    axes = list(range(axis, rank))
    return mb.layer_norm(
        x=x, axes=axes, gamma=scale, beta=b, epsilon=epsilon, name=node.output[0]
    )


def _group_norm(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX GroupNormalization: split C into num_groups, normalize each group over
    # (channels_per_group, *spatial), then apply affine scale/bias. MIL has no
    # group_norm, so compose it. scale/bias are (num_groups,) at opset 18 or (C,)
    # at opset 21 -- both reshape to broadcast over the restored (N, C, *D) layout.
    x, scale, b = operands(ctx.values_map, node, [0, 1, 2])
    num_groups = int(get_attr(node, "num_groups"))
    epsilon = get_attr(node, "epsilon", 1e-5)

    shape = x.shape
    n, c = shape[0], shape[1]
    spatial = list(shape[2:])

    # (N, C, *D) -> (N, num_groups, channels_per_group * prod(spatial)).
    grouped = mb.reshape(x=x, shape=[n, num_groups, -1])
    # Per-group normalization == layer_norm over the trailing axis (no affine).
    normed = mb.layer_norm(x=grouped, axes=[-1], epsilon=epsilon)
    normed = mb.reshape(x=normed, shape=list(shape))

    # Affine: scale/bias broadcast over channels. (C,) applies per channel;
    # (num_groups,) repeats each group's value across its channels_per_group.
    bcast = [1, c] + [1] * len(spatial)
    gamma = np.asarray(scale.val)
    beta = np.asarray(b.val)
    if gamma.shape[0] == num_groups:
        reps = c // num_groups
        gamma = np.repeat(gamma, reps)
        beta = np.repeat(beta, reps)
    gamma = gamma.reshape(bcast)
    beta = beta.reshape(bcast)

    scaled = mb.mul(x=normed, y=gamma)
    return mb.add(x=scaled, y=beta, name=node.output[0])


REGISTRY: dict[str, Lowering] = {
    "BatchNormalization": _batch_norm,
    "LayerNormalization": _layer_norm,
    "InstanceNormalization": _instance_norm,
    "GroupNormalization": _group_norm,
}

__all__ = ["REGISTRY"]
