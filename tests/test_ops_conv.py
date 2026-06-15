# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for convolution and pooling lowerings."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity, single_op_model
from onnx import TensorProto, helper, numpy_helper

pytestmark = pytest.mark.ops

FORMATS = ["mlpackage", "mlmodel"]

# Each weight/input array gets a distinct seed. The Core ML runtime caches
# compiled models by weight content within a process, so two models that share
# identical const weights can collide (the second prediction is served the first
# model). Distinct constants per test keep every model content-unique while
# still feeding ONNX and Core ML identical data within a test (parity holds).
_counter = iter(range(10_000))


def _rand(*shape: int) -> np.ndarray:
    return np.random.default_rng(next(_counter)).standard_normal(shape).astype(np.float32)


# --- Conv -------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_stride(fmt: str, stride: int) -> None:
    x = _rand(1, 3, 8, 8)
    w = _rand(4, 3, 3, 3)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [stride, stride]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_explicit_pads(fmt: str) -> None:
    x = _rand(1, 3, 8, 8)
    w = _rand(4, 3, 3, 3)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_asymmetric_pads(fmt: str) -> None:
    # Exercises the ONNX [begins..., ends...] -> MIL interleave reorder with
    # distinct begin/end values per spatial dim.
    x = _rand(1, 3, 8, 8)
    w = _rand(4, 3, 3, 3)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "pads": [0, 1, 2, 1]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "VALID"])
def test_conv_auto_pad(fmt: str, auto_pad: str) -> None:
    x = _rand(1, 3, 8, 8)
    w = _rand(4, 3, 3, 3)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [2, 2], "auto_pad": auto_pad},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_groups_depthwise(fmt: str) -> None:
    # Depthwise: groups == C_in, weight (C_out=6, C_in/groups=1, kh, kw).
    x = _rand(1, 3, 8, 8)
    w = _rand(6, 1, 3, 3)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "group": 3, "pads": [1, 1, 1, 1]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


def _conv_activation_model(
    x: np.ndarray, w: np.ndarray, group: int, *, act: str = "Relu", b: np.ndarray | None = None
) -> onnx.ModelProto:
    """A two-node ``Conv -> activation`` model.

    The Core ML runtime fuses a convolution with a following activation; the
    fused *grouped* kernel miscomputes, so the bug only appears when an activation
    follows the conv (a lone grouped conv is fine). This builder exists to
    exercise that fused path, which ``single_op_model`` cannot.
    """
    conv_inputs = ["X", "W"] + (["B"] if b is not None else [])
    inits = {"W": w} if b is None else {"W": w, "B": b}
    nodes = [
        helper.make_node(
            "Conv", conv_inputs, ["c"],
            kernel_shape=[w.shape[2], w.shape[3]], group=group, pads=[1, 1, 1, 1],
        ),
        helper.make_node(act, ["c"], ["out0"]),
    ]
    graph = helper.make_graph(
        nodes, "conv_act",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
        initializer=[numpy_helper.from_array(v, n) for n, v in inits.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model)


# Verified Core ML platform limitation (coremltools 9.0, macOS 26/Tahoe Core ML
# runtime): a *cardinality* grouped conv (1 < groups < C_in) fused with a following
# activation uses a miscomputing kernel — PSNR collapses to ~10-15 dB vs ONNX
# Runtime even in fp32, while the same conv with no activation is exact and an
# ungrouped/depthwise conv + activation is exact. It is the *runtime's* fused
# kernel, not the lowering: the emitted MIL is correct, and decomposing the conv
# (Split/Conv*/Concat) does not help because the on-device compiler re-fuses it.
# Tracked as a platform bug; xfail rather than mask a real lowering issue.
_GROUPED_CONV_FUSION_XFAIL = pytest.mark.xfail(
    reason="Core ML runtime miscomputes a fused cardinality-grouped-conv + activation",
    strict=False,
)


@_GROUPED_CONV_FUSION_XFAIL
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("group", [2, 4, 8])
def test_grouped_conv_fused_activation(fmt: str, group: int) -> None:
    # Cardinality grouped conv (1 < groups < C_in) followed by an activation.
    x = _rand(1, 16, 8, 8)
    w = _rand(16, 16 // group, 3, 3)
    model = _conv_activation_model(x, w, group)
    assert_parity(model, {"X": x}, fmt=fmt)


@_GROUPED_CONV_FUSION_XFAIL
@pytest.mark.parametrize("fmt", FORMATS)
def test_grouped_conv_fused_activation_with_bias(fmt: str) -> None:
    x = _rand(1, 16, 8, 8)
    w = _rand(16, 4, 3, 3)
    b = _rand(16)
    model = _conv_activation_model(x, w, 4, b=b)
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_with_bias(fmt: str) -> None:
    x = _rand(1, 3, 8, 8)
    w = _rand(4, 3, 3, 3)
    b = _rand(4)
    model = single_op_model(
        "Conv",
        {"X": x},
        attrs={"kernel_shape": [3, 3]},
        initializers={"W": w, "B": b},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


# --- ConvTranspose ----------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_transpose_upsample(fmt: str) -> None:
    # Basic 2x upsampling. ONNX weight is (C_in, C_out/groups, kh, kw).
    x = _rand(1, 3, 4, 4)
    w = _rand(3, 5, 3, 3)
    model = single_op_model(
        "ConvTranspose",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [2, 2]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conv_transpose_with_bias(fmt: str) -> None:
    x = _rand(1, 3, 4, 4)
    w = _rand(3, 5, 3, 3)
    b = _rand(5)
    model = single_op_model(
        "ConvTranspose",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [2, 2]},
        initializers={"W": w, "B": b},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


# --- MaxPool / AveragePool --------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_max_pool_2x2(fmt: str) -> None:
    x = _rand(1, 3, 8, 8)
    model = single_op_model(
        "MaxPool", {"X": x}, attrs={"kernel_shape": [2, 2], "strides": [2, 2]}
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_max_pool_padded(fmt: str) -> None:
    x = _rand(1, 3, 7, 7)
    model = single_op_model(
        "MaxPool",
        {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_avg_pool_2x2(fmt: str) -> None:
    x = _rand(1, 3, 8, 8)
    model = single_op_model(
        "AveragePool", {"X": x}, attrs={"kernel_shape": [2, 2], "strides": [2, 2]}
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("count_include_pad", [0, 1])
def test_avg_pool_count_include_pad(fmt: str, count_include_pad: int) -> None:
    x = _rand(1, 3, 7, 7)
    model = single_op_model(
        "AveragePool",
        {"X": x},
        attrs={
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [1, 1, 1, 1],
            "count_include_pad": count_include_pad,
        },
    )
    assert_parity(model, {"X": x}, fmt=fmt)


# --- GlobalAveragePool / GlobalMaxPool --------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_global_average_pool(fmt: str) -> None:
    x = _rand(1, 4, 5, 5)
    model = single_op_model("GlobalAveragePool", {"X": x})
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_global_max_pool(fmt: str) -> None:
    x = _rand(1, 4, 5, 5)
    model = single_op_model("GlobalMaxPool", {"X": x})
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("output_padding", [0, 1])
def test_conv_transpose_output_padding(request, fmt: str, output_padding: int) -> None:
    # output_padding adds to the output size (stride 2 -> +output_padding per axis);
    # dropping it makes the output one short, which breaks downstream residual adds.
    # The mlprogram backend honors the resulting output_shape exactly; the iOS15
    # NeuralNetwork backend miscomputes a deconv given an explicit output_shape.
    if fmt == "mlmodel" and output_padding:
        request.applymarker(pytest.mark.xfail(
            reason="NeuralNetwork backend miscomputes conv_transpose with output_shape",
            strict=True,
        ))
    x = _rand(1, 3, 8, 8)
    w = _rand(3, 4, 3, 3)
    model = single_op_model(
        "ConvTranspose", {"X": x},
        attrs={"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1],
               "output_padding": [output_padding, output_padding]},
        initializers={"W": w},
    )
    assert_parity(model, {"X": x}, fmt=fmt)
