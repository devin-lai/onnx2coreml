# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end integration tests: small but realistic multi-node models that
exercise the passes, fusion, and many lowerings together, on both formats."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity
from onnx import TensorProto, helper, numpy_helper

pytestmark = [pytest.mark.integration, pytest.mark.ops]

_RNG = np.random.default_rng(7)


def _f32(shape, scale=1.0):
    return (_RNG.standard_normal(shape) * scale).astype(np.float32)


def _cnn_block() -> onnx.ModelProto:
    """Conv -> BatchNorm -> Relu -> GlobalAveragePool -> Flatten -> Gemm."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.UNDEFINED, None)
    inits = {
        "cw": _f32((4, 3, 3, 3), 0.2),
        "bn_s": (_RNG.random(4).astype(np.float32) + 0.5),
        "bn_b": _f32(4, 0.1),
        "bn_m": _f32(4, 0.1),
        "bn_v": (_RNG.random(4).astype(np.float32) + 0.5),  # variance > 0
        "gw": _f32((4, 3), 0.3),
        "gb": _f32(3, 0.1),
    }
    nodes = [
        helper.make_node("Conv", ["X", "cw"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node(
            "BatchNormalization", ["c", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], epsilon=1e-5
        ),
        helper.make_node("Relu", ["bn"], ["r"]),
        helper.make_node("GlobalAveragePool", ["r"], ["g"]),
        helper.make_node("Flatten", ["g"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "gw", "gb"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "cnn_block", [x], [y],
        initializer=[numpy_helper.from_array(a, n) for n, a in inits.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model)


def _attention_block() -> onnx.ModelProto:
    """Self-attention (decomposed) + residual + LayerNorm — the SDPA fusion target.

    scores = X @ Xᵀ; scaled = scores / sqrt(d); probs = softmax(scaled);
    ctx = probs @ X; out = LayerNorm(X + ctx).
    """
    seq, d = 4, 8
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq, d])
    y = helper.make_tensor_value_info("Y", TensorProto.UNDEFINED, None)
    inits = {
        "scale": np.array(1.0 / np.sqrt(d), dtype=np.float32),
        "ln_s": (_RNG.random(d).astype(np.float32) + 0.5),
        "ln_b": _f32(d, 0.1),
    }
    nodes = [
        helper.make_node("Transpose", ["X"], ["Kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["X", "Kt"], ["scores"]),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "X"], ["ctx"]),
        helper.make_node("Add", ["X", "ctx"], ["res"]),
        helper.make_node(
            "LayerNormalization", ["res", "ln_s", "ln_b"], ["Y"], axis=-1, epsilon=1e-5
        ),
    ]
    graph = helper.make_graph(
        nodes, "attention_block", [x], [y],
        initializer=[numpy_helper.from_array(a, n) for n, a in inits.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_cnn_block(fmt):
    x = _f32((1, 3, 8, 8))
    assert_parity(_cnn_block(), {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("fuse", [True, False])
def test_attention_block(fmt, fuse):
    # fuse=True exercises the SDPA fusion; fuse=False the decomposed primitives.
    # Both must match ONNX Runtime on both formats.
    x = _f32((1, 4, 8))
    assert_parity(_attention_block(), {"X": x}, fmt=fmt, fuse=fuse)
