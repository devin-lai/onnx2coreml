# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shape and data-movement lowerings (Reshape, Transpose, Concat, Pad, ...).

These ops mostly reshuffle or relabel a tensor's metadata. The recurring chore
here is recovering ONNX's static shape/axes/sizes — which may live in an
``attr`` (older opsets) or as a constant *input* (opset 13+) — and translating
ONNX's conventions into MIL's. The two conventions that differ materially are
``Pad`` (ONNX groups all begins then all ends; MIL interleaves per-axis) and
``Resize`` (ONNX picks interpolation via a coordinate-transformation mode; MIL
picks an ``align_corners`` flag / ``sampling_mode`` string).

Resize/Upsample deliberately target the iOS15 ``upsample_*`` / ``resize_*`` ops
(not the iOS17 ``resize``) so both ``.mlpackage`` and ``.mlmodel`` build. See
``_resize`` for exactly which ONNX modes are honored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from .._mil import mb
from ._common import const_array, get_attr, operands
from ._context import Lowering, LoweringContext


def _static_shape(var: Any) -> list[int]:
    """The fixed integer shape of a MIL Var (inputs are fixed-shape here)."""
    return [int(d) for d in var.shape]


def _reshape(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Reshape. ``shape`` is the constant input 1.

    ``-1`` (infer one dim) is passed straight to MIL, which supports it. ``0``
    means "copy this dim from the input" when ``allowzero == 0`` (the default);
    we resolve those against the input's static shape first, because MIL's own
    ``0`` handling requires ``len(shape) == rank(x)`` which need not hold once a
    ``-1`` is also present.
    """
    (x,) = operands(ctx.values_map, node, [0])
    shape_arr = const_array(ctx, node, 1)
    if shape_arr is None:
        raise ValueError("Reshape requires a constant 'shape' input")
    dims = [int(s) for s in shape_arr.tolist()]
    if get_attr(node, "allowzero", 0) == 0 and 0 in dims:
        in_shape = _static_shape(x)
        dims = [in_shape[i] if s == 0 else s for i, s in enumerate(dims)]
    return mb.reshape(x=x, shape=dims, name=node.output[0])


def _transpose(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Transpose. ``perm`` defaults to the reversed axis order."""
    (x,) = operands(ctx.values_map, node, [0])
    perm = get_attr(node, "perm")
    if perm is None:
        perm = list(reversed(range(len(_static_shape(x)))))
    return mb.transpose(x=x, perm=[int(p) for p in perm], name=node.output[0])


def _flatten(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Flatten: collapse to 2D ``(prod(dims[:axis]), prod(dims[axis:]))``."""
    (x,) = operands(ctx.values_map, node, [0])
    in_shape = _static_shape(x)
    axis = int(get_attr(node, "axis", 1))
    if axis < 0:
        axis += len(in_shape)
    outer = int(np.prod(in_shape[:axis])) if axis > 0 else 1
    inner = int(np.prod(in_shape[axis:])) if axis < len(in_shape) else 1
    return mb.reshape(x=x, shape=[outer, inner], name=node.output[0])


def _axes(ctx: LoweringContext, node: onnx.NodeProto) -> list[int] | None:
    """Squeeze/Unsqueeze axes: ``axes`` attr (opset<13) or const input 1."""
    axes = get_attr(node, "axes")
    if axes is None:
        arr = const_array(ctx, node, 1)
        if arr is not None:
            axes = arr.tolist()
    if axes is None:
        return None
    return [int(a) for a in axes]


def _squeeze(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Squeeze. With no axes, MIL squeezes every size-1 dim (matches ONNX)."""
    (x,) = operands(ctx.values_map, node, [0])
    axes = _axes(ctx, node)
    if axes is None:
        return mb.squeeze(x=x, name=node.output[0])
    return mb.squeeze(x=x, axes=axes, name=node.output[0])


def _unsqueeze(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Unsqueeze -> MIL expand_dims at the given axes."""
    (x,) = operands(ctx.values_map, node, [0])
    axes = _axes(ctx, node)
    if axes is None:
        raise ValueError("Unsqueeze requires 'axes'")
    return mb.expand_dims(x=x, axes=axes, name=node.output[0])


def _concat(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Concat of all inputs along ``axis``."""
    values = operands(ctx.values_map, node, list(range(len(node.input))))
    axis = int(get_attr(node, "axis"))
    return mb.concat(values=values, axis=axis, name=node.output[0])


def _split(ctx: LoweringContext, node: onnx.NodeProto) -> list[Any]:
    """ONNX Split into ``len(node.output)`` pieces along ``axis``.

    Split sizes come from the ``split`` attr (opset<13) or const input 1
    (opset 13+). With neither, ONNX splits evenly into ``num_outputs`` (opset18)
    or ``len(outputs)`` pieces; MIL's ``num_splits`` handles the even case.
    """
    (x,) = operands(ctx.values_map, node, [0])
    axis = int(get_attr(node, "axis", 0))
    n_out = len(node.output)

    sizes = get_attr(node, "split")
    if sizes is None:
        arr = const_array(ctx, node, 1)
        if arr is not None:
            sizes = arr.tolist()

    if sizes is not None:
        outs = mb.split(x=x, split_sizes=[int(s) for s in sizes], axis=axis)
    else:
        outs = mb.split(x=x, num_splits=n_out, axis=axis)

    # mb.split returns a tuple of Vars; rename each to its node output. MIL Vars
    # are immutable, so re-emit a named identity per branch.
    return [mb.identity(x=o, name=name) for o, name in zip(outs, node.output, strict=True)]


def _pad(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Pad -> MIL pad.

    ONNX ``pads`` (input 1, opset 11+) is ``[b0..bN, e0..eN]``; MIL wants the
    per-axis interleave ``[b0, e0, b1, e1, ...]``. ``mode`` maps constant/reflect
    straight through, and ONNX ``edge`` -> MIL ``replicate``. The fill value is
    input 2 (opset 11+) or the ``value`` attr.
    """
    (x,) = operands(ctx.values_map, node, [0])
    pads_arr = const_array(ctx, node, 1)
    if pads_arr is None:
        raise ValueError("Pad requires a constant 'pads' input")
    pads = [int(p) for p in pads_arr.tolist()]
    rank = len(pads) // 2
    interleaved = [pads[i] if half == 0 else pads[rank + i]
                   for i in range(rank) for half in (0, 1)]

    mode = (get_attr(node, "mode", b"constant") or b"constant")
    mode = mode.decode() if isinstance(mode, bytes) else mode
    mode = "replicate" if mode == "edge" else mode

    cval = const_array(ctx, node, 2)
    if cval is None:
        cval = get_attr(node, "value", 0.0)
    fill = float(np.asarray(cval).item())

    return mb.pad(x=x, pad=interleaved, mode=mode, constant_val=fill, name=node.output[0])


# ONNX TensorProto dtype -> MIL cast dtype string, with Core ML's 64->32 policy.
# Core ML compute is 32-bit, so every integer width folds onto int32 (the values
# casts carry — indices, small counts, byte-range image data — fit int32, and the
# narrower-than-32 ONNX integer types have no Core ML compute counterpart).
_ONNX_TO_MIL_CAST: dict[int, str] = {
    TensorProto.FLOAT: "fp32",
    TensorProto.FLOAT16: "fp16",
    TensorProto.DOUBLE: "fp32",  # narrow
    TensorProto.INT8: "int32",
    TensorProto.INT16: "int32",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int32",  # narrow
    TensorProto.UINT8: "int32",
    TensorProto.UINT16: "int32",
    TensorProto.UINT32: "int32",
    TensorProto.UINT64: "int32",  # narrow
    TensorProto.BOOL: "bool",
}


def _cast(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Cast. ``to`` is a TensorProto enum -> MIL dtype string (64->32)."""
    (x,) = operands(ctx.values_map, node, [0])
    to = int(get_attr(node, "to"))
    dtype = _ONNX_TO_MIL_CAST.get(to)
    if dtype is None:
        name = TensorProto.DataType.Name(to)
        raise ValueError(f"Cast to {name} is not supported")
    return mb.cast(x=x, dtype=dtype, name=node.output[0])


def _identity(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Identity -> MIL identity."""
    (x,) = operands(ctx.values_map, node, [0])
    return mb.identity(x=x, name=node.output[0])


def _depth_to_space(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX DepthToSpace -> rearrange depth into spatial blocks.

    MIL ``depth_to_space`` implements the ``DCR`` ordering (depth-column-row, the
    ONNX default). For ``CRD`` (column-row-depth) we build the equivalent
    reshape/transpose/reshape explicitly, since the two modes differ only in how
    the channel axis is unpacked.
    """
    (x,) = operands(ctx.values_map, node, [0])
    bs = int(get_attr(node, "blocksize"))
    mode = get_attr(node, "mode", b"DCR")
    mode = mode.decode() if isinstance(mode, bytes) else mode
    if mode == "DCR":
        return mb.depth_to_space(x=x, block_size=bs, name=node.output[0])

    n, c, h, w = (int(d) for d in x.shape)
    cb = c // (bs * bs)
    # CRD: (N, C/bs^2, bs, bs, H, W) -> (N, C/bs^2, H, bs, W, bs) -> (N, C/bs^2, H*bs, W*bs)
    reshaped = mb.reshape(x=x, shape=[n, cb, bs, bs, h, w])
    transposed = mb.transpose(x=reshaped, perm=[0, 1, 4, 2, 5, 3])
    return mb.reshape(x=transposed, shape=[n, cb, h * bs, w * bs], name=node.output[0])


def _resize(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Resize / Upsample for the common 4D image cases.

    Inputs: X, roi (ignored), scales, sizes — at most one of scales/sizes is
    set, and it must be constant. We map onto the iOS15 spatial ops so both
    container formats build:

    * ``mode='nearest'`` -> ``upsample_nearest_neighbor`` (scales) or
      ``resize_nearest_neighbor`` (sizes). Only ``coordinate_transformation_mode
      ='asymmetric'`` with ``nearest_mode='floor'`` (the ONNX defaults) is
      reproduced exactly; both MIL ops floor source coordinates.
    * ``mode='linear'`` -> ``upsample_bilinear`` (scales) or ``resize_bilinear``
      (sizes), with the coordinate-transformation mode mapped to an
      ``align_corners`` flag / ``sampling_mode``:
        - ``half_pixel`` (ONNX default) -> ``align_corners=False`` /
          ``UNALIGN_CORNERS``
        - ``align_corners``             -> ``align_corners=True`` /
          ``STRICT_ALIGN_CORNERS``

    Other modes (``cubic``; ``pytorch_half_pixel``; nearest with a non-floor
    ``nearest_mode``; non-spatial scales) are not emitted — those test cases are
    skipped rather than silently mismatched.
    """
    (x,) = operands(ctx.values_map, node, [0])
    mode = get_attr(node, "mode", b"nearest")
    mode = mode.decode() if isinstance(mode, bytes) else mode
    coord = get_attr(node, "coordinate_transformation_mode", b"half_pixel")
    coord = coord.decode() if isinstance(coord, bytes) else coord

    def _present(idx: int) -> Any:
        # A const input that is absent OR an empty tensor counts as "not given".
        arr = const_array(ctx, node, idx)
        return arr if arr is not None and arr.size else None

    if node.op_type == "Upsample":
        # Deprecated opset-9 form: inputs are [X, scales]; no roi/sizes.
        scales, sizes = _present(1), None
    else:
        scales, sizes = _present(2), _present(3)

    if mode == "nearest":
        if scales is not None:
            sh, sw = float(scales[-2]), float(scales[-1])
            return mb.upsample_nearest_neighbor(
                x=x, scale_factor_height=sh, scale_factor_width=sw, name=node.output[0]
            )
        if sizes is not None:
            th, tw = int(sizes[-2]), int(sizes[-1])
            return mb.resize_nearest_neighbor(
                x=x, target_size_height=th, target_size_width=tw, name=node.output[0]
            )
        raise ValueError("Resize requires either 'scales' or 'sizes'")

    if mode == "linear":
        align = coord == "align_corners"
        if scales is not None:
            sh, sw = float(scales[-2]), float(scales[-1])
            return mb.upsample_bilinear(
                x=x, scale_factor_height=sh, scale_factor_width=sw,
                align_corners=align, name=node.output[0],
            )
        if sizes is not None:
            th, tw = int(sizes[-2]), int(sizes[-1])
            sampling = "STRICT_ALIGN_CORNERS" if align else "UNALIGN_CORNERS"
            return mb.resize_bilinear(
                x=x, target_size_height=th, target_size_width=tw,
                sampling_mode=sampling, name=node.output[0],
            )
        raise ValueError("Resize requires either 'scales' or 'sizes'")

    raise ValueError(f"Resize mode '{mode}' is not supported")


def _grid_pad_mode(onnx_pad: str, align: bool) -> str:
    """ONNX GridSample padding_mode -> MIL resample padding_mode.

    ``reflection`` is align-dependent: ONNX reflects across the pixel *edge* when
    ``align_corners=False`` (MIL's ``symmetric``) and across the pixel *center*
    when ``align_corners=True`` (MIL's ``reflection``).
    """
    if onnx_pad == "zeros":
        return "constant"
    if onnx_pad == "border":
        return "border"
    if onnx_pad == "reflection":
        return "reflection" if align else "symmetric"
    raise ValueError(f"GridSample padding_mode '{onnx_pad}' is not supported")


def _grid_sample(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX GridSample -> MIL ``resample``.

    ONNX's grid is ``(N, H_out, W_out, 2)`` with each location an ``(x, y)`` pair
    normalized to ``[-1, 1]`` — exactly MIL ``resample`` with
    ``coordinates_mode='normalized_minus_one_to_one'``. ``bicubic`` is not emitted
    (MIL resample has only nearest / bilinear).
    """
    x, grid = operands(ctx.values_map, node, [0, 1])
    mode = get_attr(node, "mode", b"linear")
    mode = mode.decode() if isinstance(mode, bytes) else mode
    # opset 20 renamed the modes: "linear"/"cubic" replace "bilinear"/"bicubic".
    sampling = {"linear": "bilinear", "bilinear": "bilinear", "nearest": "nearest"}.get(mode)
    if sampling is None:
        raise ValueError(f"GridSample mode '{mode}' is not supported")
    pad = get_attr(node, "padding_mode", b"zeros")
    pad = pad.decode() if isinstance(pad, bytes) else pad
    align = bool(get_attr(node, "align_corners", 0))
    return mb.resample(
        x=x,
        coordinates=grid,
        sampling_mode=sampling,
        padding_mode=_grid_pad_mode(pad, align),
        padding_value=0.0,
        coordinates_mode="normalized_minus_one_to_one",
        align_corners=align,
        name=node.output[0],
    )


REGISTRY: dict[str, Lowering] = {
    "Reshape": _reshape,
    "Transpose": _transpose,
    "Flatten": _flatten,
    "Squeeze": _squeeze,
    "Unsqueeze": _unsqueeze,
    "Concat": _concat,
    "Split": _split,
    "Pad": _pad,
    "Cast": _cast,
    "Identity": _identity,
    "DepthToSpace": _depth_to_space,
    "GridSample": _grid_sample,
    "Resize": _resize,
    "Upsample": _resize,
}

__all__ = ["REGISTRY"]
