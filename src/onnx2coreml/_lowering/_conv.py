# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Convolution and pooling lowerings.

The shared subtlety is padding. ONNX lays ``pads`` out as all begins followed by
all ends (``[d0_begin, d1_begin, ..., d0_end, d1_end, ...]``), while MIL's
``custom`` pad interleaves them per dimension (``[d0_begin, d0_end, d1_begin,
d1_end, ...]``); :func:`_onnx_pads_to_mil` does that reorder. ONNX ``auto_pad``
maps onto MIL ``pad_type`` strings (NOTSET->custom, VALID->valid, SAME_UPPER->same,
SAME_LOWER->same_lower).
"""

from __future__ import annotations

from typing import Any

import onnx

from .._mil import mb
from ._common import get_attr, operands
from ._context import Lowering, LoweringContext

# ONNX auto_pad -> MIL pad_type. NOTSET means "use explicit pads" -> custom.
_AUTO_PAD_TO_MIL = {
    "VALID": "valid",
    "SAME_UPPER": "same",
    "SAME_LOWER": "same_lower",
}


def _onnx_pads_to_mil(pads: list[int], n_spatial: int) -> list[int]:
    """Reorder ONNX ``pads`` ([all begins, all ends]) into MIL's per-dim
    interleaving ([d0_begin, d0_end, d1_begin, d1_end, ...])."""
    out: list[int] = []
    for i in range(n_spatial):
        out.extend((pads[i], pads[i + n_spatial]))
    return out


def _pad_kwargs(node: onnx.NodeProto, n_spatial: int) -> dict[str, Any]:
    """Translate ONNX padding attrs into MIL ``pad_type`` (+ ``pad`` if custom).

    auto_pad (when not NOTSET) wins over explicit ``pads`` per the ONNX spec.
    """
    auto_pad = get_attr(node, "auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):  # ONNX string attrs decode as bytes
        auto_pad = auto_pad.decode("utf-8")
    if auto_pad != "NOTSET":
        return {"pad_type": _AUTO_PAD_TO_MIL[auto_pad]}

    pads = get_attr(node, "pads")
    if pads is None or not any(pads):
        return {"pad_type": "valid"}
    return {"pad_type": "custom", "pad": _onnx_pads_to_mil(list(pads), n_spatial)}


def _spatial_attr(node: onnx.NodeProto, name: str, n_spatial: int, default: int) -> list[int]:
    """Read a per-spatial-dim int-list attr, defaulting to ``[default] * n_spatial``."""
    val = get_attr(node, name)
    if val is None:
        return [default] * n_spatial
    return list(val)


def _conv(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX Conv weight (C_out, C_in/groups, *K) and bias (C_out,) match MIL conv
    # layout directly, so no transpose is needed. (A cardinality grouped conv fed
    # into an activation is miscomputed by the Core ML *runtime*, not this
    # lowering — see tests/test_ops_conv.py::test_grouped_conv_fused_activation.)
    x, weight, bias = operands(ctx.values_map, node, [0, 1, 2])
    n_spatial = x.rank - 2

    kwargs: dict[str, Any] = {
        "x": x,
        "weight": weight,
        "strides": _spatial_attr(node, "strides", n_spatial, 1),
        "dilations": _spatial_attr(node, "dilations", n_spatial, 1),
        "groups": get_attr(node, "group", 1),
        **_pad_kwargs(node, n_spatial),
    }
    if bias is not None:
        kwargs["bias"] = bias
    return mb.conv(name=node.output[0], **kwargs)


def _convtranspose_spatial(
    node: onnx.NodeProto, x: Any, weight: Any, n_spatial: int,
    strides: list[int], dilations: list[int],
) -> list[int]:
    """ONNX ConvTranspose output spatial size, which folds in ``output_padding``:

    ``out = stride*(in-1) + output_padding + ((kernel-1)*dilation + 1) - pad_begin - pad_end``
    """
    out_pad = _spatial_attr(node, "output_padding", n_spatial, 0)
    pads = list(get_attr(node, "pads") or [0] * (2 * n_spatial))
    in_spatial = [int(d) for d in x.shape[2:]]
    kernel = [int(d) for d in weight.shape[2:]]
    return [
        strides[i] * (in_spatial[i] - 1) + out_pad[i]
        + (kernel[i] - 1) * dilations[i] + 1 - pads[i] - pads[n_spatial + i]
        for i in range(n_spatial)
    ]


def _conv_transpose(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    # ONNX ConvTranspose weight (C_in, C_out/groups, *K) and bias (C_out,) match
    # MIL conv_transpose layout directly.
    x, weight, bias = operands(ctx.values_map, node, [0, 1, 2])
    n_spatial = x.rank - 2
    strides = _spatial_attr(node, "strides", n_spatial, 1)
    dilations = _spatial_attr(node, "dilations", n_spatial, 1)
    groups = get_attr(node, "group", 1)

    kwargs: dict[str, Any] = {
        "x": x, "weight": weight, "strides": strides, "dilations": dilations,
        "groups": groups, **_pad_kwargs(node, n_spatial),
    }
    if bias is not None:
        kwargs["bias"] = bias

    # MIL's output_shape wants [n, C_out, *spatial]. ONNX gives the spatial target
    # either explicitly (output_shape) or implicitly via output_padding, which the
    # deconv size formula must add — otherwise the output is one short per padded
    # axis. Either way, materialize the full output_shape for MIL.
    output_shape = get_attr(node, "output_shape")
    auto_pad = get_attr(node, "auto_pad", b"NOTSET")
    auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
    if output_shape is None and any(_spatial_attr(node, "output_padding", n_spatial, 0)) \
            and auto_pad == "NOTSET":
        output_shape = _convtranspose_spatial(node, x, weight, n_spatial, strides, dilations)
    if output_shape is not None:
        c_out = int(weight.shape[1]) * groups
        kwargs["output_shape"] = [int(x.shape[0]), c_out, *[int(s) for s in output_shape]]
    return mb.conv_transpose(name=node.output[0], **kwargs)


def _pool(mb_op: Any, *, is_avg: bool) -> Lowering:
    """Build a MaxPool/AveragePool lowering around the given MIL pool op."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        (x,) = operands(ctx.values_map, node, [0])
        n_spatial = x.rank - 2

        kwargs: dict[str, Any] = {
            "x": x,
            "kernel_sizes": list(get_attr(node, "kernel_shape")),
            "strides": _spatial_attr(node, "strides", n_spatial, 1),
            "ceil_mode": bool(get_attr(node, "ceil_mode", 0)),
            **_pad_kwargs(node, n_spatial),
        }
        if is_avg:
            # ONNX count_include_pad (default 0) is the inverse of MIL's
            # exclude_padding_from_average.
            kwargs["exclude_padding_from_average"] = not get_attr(node, "count_include_pad", 0)
        return mb_op(name=node.output[0], **kwargs)

    return lower


def _global_pool(mb_reduce: Any) -> Lowering:
    """Build a GlobalAveragePool/GlobalMaxPool lowering: reduce over the spatial
    axes (2..rank-1), keeping dims so the output stays NCHW with 1x1 spatial."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        (x,) = operands(ctx.values_map, node, [0])
        axes = list(range(2, x.rank))
        return mb_reduce(x=x, axes=axes, keep_dims=True, name=node.output[0])

    return lower


REGISTRY: dict[str, Lowering] = {
    "Conv": _conv,
    "ConvTranspose": _conv_transpose,
    "MaxPool": _pool(mb.max_pool, is_avg=False),
    "AveragePool": _pool(mb.avg_pool, is_avg=True),
    "GlobalAveragePool": _global_pool(mb.reduce_mean),
    "GlobalMaxPool": _global_pool(mb.reduce_max),
}

__all__ = ["REGISTRY"]
