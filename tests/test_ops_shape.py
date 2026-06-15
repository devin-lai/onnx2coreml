# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for shape / data-movement ops on both container formats."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model
from onnx import TensorProto

pytestmark = pytest.mark.ops

FMTS = ["mlpackage", "mlmodel"]


@pytest.fixture
def x4() -> dict[str, np.ndarray]:
    """A small deterministic NCHW float tensor reused across many ops."""
    return {"X": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)}


@pytest.mark.parametrize("fmt", FMTS)
def test_reshape_neg_one(fmt, x4):
    m = single_op_model(
        "Reshape", x4, initializers={"shape": np.array([2, 3, -1], dtype=np.int64)}
    )
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_reshape_zero_dim(fmt, x4):
    # 0 == "copy dim 0 from input" (allowzero defaults to 0).
    m = single_op_model(
        "Reshape", x4, initializers={"shape": np.array([0, 3, 20], dtype=np.int64)}
    )
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_transpose(fmt, x4):
    m = single_op_model("Transpose", x4, attrs={"perm": [0, 2, 3, 1]})
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("axis", [1, 2])
def test_flatten(fmt, axis, x4):
    m = single_op_model("Flatten", x4, attrs={"axis": axis})
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_squeeze(fmt):
    x = {"X": np.arange(3 * 4, dtype=np.float32).reshape(1, 3, 1, 4)}
    m = single_op_model(
        "Squeeze", x, initializers={"axes": np.array([0, 2], dtype=np.int64)}
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_unsqueeze(fmt):
    x = {"X": np.arange(3 * 4, dtype=np.float32).reshape(3, 4)}
    m = single_op_model(
        "Unsqueeze", x, initializers={"axes": np.array([0, 2], dtype=np.int64)}
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("axis", [0, 1])
def test_concat(fmt, axis):
    a = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    b = (np.arange(2 * 3, dtype=np.float32) + 100).reshape(2, 3)
    inputs = {"A": a, "B": b}
    m = single_op_model("Concat", inputs, attrs={"axis": axis})
    assert_parity(m, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_split_even_into_3(fmt):
    # 6 along axis 1 -> three pieces of 2 via num_outputs (opset-18 attribute).
    x = {"X": np.arange(2 * 6, dtype=np.float32).reshape(2, 6)}
    m = single_op_model(
        "Split", x, n_outputs=3, attrs={"axis": 1, "num_outputs": 3}, opset=18
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_split_sizes_into_2(fmt):
    # Uneven split via a `split` initializer (opset 13+ input form).
    x = {"X": np.arange(2 * 5, dtype=np.float32).reshape(2, 5)}
    m = single_op_model(
        "Split",
        x,
        n_outputs=2,
        attrs={"axis": 1},
        initializers={"split": np.array([2, 3], dtype=np.int64)},
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_pad_constant(fmt, x4):
    # Pad H/W only; ONNX layout [b0,b1,b2,b3, e0,e1,e2,e3].
    pads = np.array([0, 0, 1, 2, 0, 0, 1, 2], dtype=np.int64)
    m = single_op_model(
        "Pad",
        x4,
        attrs={"mode": "constant"},
        initializers={"pads": pads, "value": np.array(3.5, dtype=np.float32)},
    )
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_pad_reflect(fmt, x4):
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    m = single_op_model(
        "Pad", x4, attrs={"mode": "reflect"}, initializers={"pads": pads}
    )
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_cast_float_to_int(fmt):
    from onnx import TensorProto

    x = {"X": (np.arange(2 * 3, dtype=np.float32).reshape(2, 3) * 1.7)}
    m = single_op_model("Cast", x, attrs={"to": int(TensorProto.INT32)})
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_cast_int_to_float(fmt):
    from onnx import TensorProto

    x = {"X": np.arange(2 * 3, dtype=np.int32).reshape(2, 3)}
    m = single_op_model("Cast", x, attrs={"to": int(TensorProto.FLOAT)})
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_identity(fmt, x4):
    m = single_op_model("Identity", x4)
    assert_parity(m, x4, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_resize_nearest_scales(fmt):
    # nearest x2 via scales; ONNX defaults: asymmetric coord + floor nearest_mode.
    x = {"X": np.arange(1 * 1 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2)}
    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    m = single_op_model(
        "Resize",
        x,
        attrs={"mode": "nearest"},
        initializers={"roi": roi, "scales": scales},
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize(
    "fmt",
    [
        "mlpackage",
        pytest.param(
            "mlmodel",
            marks=pytest.mark.xfail(
                raises=NotImplementedError,
                strict=True,
                reason=(
                    "linear + half_pixel via sizes needs resize_bilinear with "
                    "sampling_mode=UNALIGN_CORNERS, which the iOS15 (.mlmodel) "
                    "neuralnetwork backend does not implement (it supports only "
                    "STRICT_ALIGN/ALIGN/DEFAULT/OFFSET corners). The scales path "
                    "(upsample_bilinear) covers half_pixel on both formats — see "
                    "test_resize_linear_half_pixel_scales."
                ),
            ),
        ),
    ],
)
def test_resize_linear_sizes(fmt):
    # linear via sizes; ONNX default coordinate_transformation_mode is half_pixel.
    x = {"X": np.arange(1 * 1 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2)}
    roi = np.array([], dtype=np.float32)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 4, 4], dtype=np.int64)
    m = single_op_model(
        "Resize",
        x,
        attrs={"mode": "linear"},
        initializers={"roi": roi, "scales": scales, "sizes": sizes},
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_resize_linear_half_pixel_scales(fmt):
    # linear + half_pixel via scales -> upsample_bilinear(align_corners=False),
    # which both the mlprogram and neuralnetwork backends support.
    x = {"X": np.arange(1 * 1 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2)}
    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    m = single_op_model(
        "Resize",
        x,
        attrs={"mode": "linear"},
        initializers={"roi": roi, "scales": scales},
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_resize_linear_align_corners_sizes(fmt):
    x = {"X": np.arange(1 * 1 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2)}
    roi = np.array([], dtype=np.float32)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 1, 4, 4], dtype=np.int64)
    m = single_op_model(
        "Resize",
        x,
        attrs={"mode": "linear", "coordinate_transformation_mode": "align_corners"},
        initializers={"roi": roi, "scales": scales, "sizes": sizes},
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
def test_upsample_nearest_scales(fmt):
    # Deprecated opset-9 Upsample; scales path, nearest.
    x = {"X": np.arange(1 * 1 * 2 * 2, dtype=np.float32).reshape(1, 1, 2, 2)}
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    m = single_op_model(
        "Upsample",
        x,
        attrs={"mode": "nearest"},
        initializers={"scales": scales},
        opset=9,
    )
    assert_parity(m, x, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space(fmt, mode):
    # C=8, blocksize=2 -> output channels 8/4 = 2, spatial doubled.
    x = np.arange(1 * 8 * 3 * 4, dtype=np.float32).reshape(1, 8, 3, 4) * 0.1
    model = single_op_model("DepthToSpace", {"X": x}, attrs={"blocksize": 2, "mode": mode})
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("to", [TensorProto.UINT8, TensorProto.INT32])
def test_cast_integer(fmt, to):
    # Values in [0, 200] truncate identically under uint8 and int32.
    x = (np.arange(2 * 3, dtype=np.float32).reshape(2, 3) * 30.0)
    model = single_op_model("Cast", {"X": x}, attrs={"to": to})
    assert_parity(model, {"X": x}, fmt=fmt)


def _grid_sample_model(mode, padding_mode, align_corners):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
    # grid is (N, H_out, W_out, 2), (x, y) order. Range [-1.2, 1.2] deliberately
    # exceeds [-1, 1] so the padding_mode is actually exercised at the borders.
    grid = (rng.random((1, 3, 4, 2)).astype(np.float32) * 2.4 - 1.2)
    model = single_op_model(
        "GridSample", {"X": x, "grid": grid},
        attrs={"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners},
    )
    return model, {"X": x, "grid": grid}


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(mode, padding_mode, align_corners):
    # ML Program path (resample). reflection is align-dependent (symmetric vs
    # reflection in MIL); zeros/border map directly.
    model, ins = _grid_sample_model(mode, padding_mode, align_corners)
    assert_parity(model, ins, fmt="mlpackage")


# GridSample lowers to MIL ``resample``, an ML Program op the iOS15 NeuralNetwork
# (.mlmodel) backend does not implement — so .mlmodel cannot carry GridSample.
@pytest.mark.xfail(reason="NeuralNetwork (.mlmodel) backend has no resample op", strict=True)
def test_grid_sample_mlmodel_unsupported():
    model, ins = _grid_sample_model("bilinear", "zeros", 0)
    assert_parity(model, ins, fmt="mlmodel")


def test_grid_sample_linear_alias():
    # opset-20 renamed bilinear -> "linear"; it must map to the same sampler.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
    grid = (rng.random((1, 3, 4, 2)).astype(np.float32) * 2.0 - 1.0)
    model = single_op_model(
        "GridSample", {"X": x, "grid": grid},
        attrs={"mode": "linear", "padding_mode": "zeros", "align_corners": 0},
        opset=20,
    )
    assert_parity(model, {"X": x, "grid": grid}, fmt="mlpackage")
