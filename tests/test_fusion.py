# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the graph-fusion pass and the ScaledDotProductAttention lowering.

Builds the decomposed attention subgraph (Transpose + MatMul + scale + Softmax +
MatMul) and checks two things: the fusion collapses it to one node and converts
with parity (fuse on), and the unfused primitive chain converts with parity too
(fuse off). Parity is asserted on both container formats.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity
from onnx import TensorProto, helper, numpy_helper

import onnx2coreml as o2c
from onnx2coreml._utils import get_attr

pytestmark = pytest.mark.ops

FORMATS = ["mlpackage", "mlmodel"]

_SEQ = 4
_D = 8


def _decomposed_attention(scale_op: str = "Mul") -> onnx.ModelProto:
    """Build the canonical decomposed scaled-dot-product-attention graph.

    Q, K, V are float inputs of shape ``(1, _SEQ, _D)``. ``scale_op`` selects how
    the scores are scaled: ``"Mul"`` by ``1/sqrt(d)`` or ``"Div"`` by ``sqrt(d)``.
    """
    if scale_op == "Div":
        const = np.sqrt(_D).astype(np.float32)
    else:
        const = (1.0 / np.sqrt(_D)).astype(np.float32)

    vis = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, _SEQ, _D])
        for n in ("Q", "K", "V")
    ]
    out = helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)
    nodes = [
        helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["Q", "Kt"], ["scores"]),
        helper.make_node(scale_op, ["scores", "scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "V"], ["out0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "decomposed_attention",
        vis,
        [out],
        initializer=[numpy_helper.from_array(np.array(const, np.float32), "scale")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _attention_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {n: rng.standard_normal((1, _SEQ, _D)).astype(np.float32) for n in ("Q", "K", "V")}


# --- The fusion rewrite itself ------------------------------------------------


@pytest.mark.parametrize("scale_op", ["Mul", "Div"])
def test_fusion_collapses_attention(scale_op: str) -> None:
    """run() rewrites the 5-node decomposition into one ScaledDotProductAttention."""
    model = _decomposed_attention(scale_op)
    assert len(model.graph.node) == 5

    fused = o2c._fusion.run(model)
    nodes = list(fused.graph.node)
    assert len(nodes) == 1, [n.op_type for n in nodes]

    (sdpa,) = nodes
    assert sdpa.op_type == "ScaledDotProductAttention"
    assert list(sdpa.input) == ["Q", "K", "V"]
    assert sdpa.output[0] == "out0"  # final output name preserved
    # The carried scale is the effective multiplier, 1/sqrt(d), either way.
    assert get_attr(sdpa, "scale") == pytest.approx(1.0 / np.sqrt(_D))


def test_fusion_noop_on_single_op() -> None:
    """A single-op model is passed through structurally unchanged."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)
    node = helper.make_node("Relu", ["X"], ["out0"])
    graph = helper.make_graph([node], "single", [x], [y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    fused = o2c._fusion.run(model)
    assert [n.op_type for n in fused.graph.node] == ["Relu"]


def test_fusion_noop_when_softmax_axis_wrong() -> None:
    """A softmax over a non-final axis is not an attention block; leave it alone."""
    model = _decomposed_attention("Mul")
    for n in model.graph.node:
        if n.op_type == "Softmax":
            del n.attribute[:]
            n.attribute.append(helper.make_attribute("axis", 1))
    fused = o2c._fusion.run(model)
    assert "ScaledDotProductAttention" not in [n.op_type for n in fused.graph.node]
    assert len(fused.graph.node) == 5  # untouched


# --- End-to-end conversion + parity ------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("scale_op", ["Mul", "Div"])
def test_attention_parity_fuse_on(fmt: str, scale_op: str) -> None:
    """Test A: with fusion on, the fused node converts with parity on both formats."""
    model = _decomposed_attention(scale_op)
    inputs = _attention_inputs()

    # The fusion actually fires for this graph.
    assert [n.op_type for n in o2c._fusion.run(model).graph.node] == [
        "ScaledDotProductAttention"
    ]
    # convert(..., fuse=True) takes the fused path and matches onnxruntime.
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_attention_parity_fuse_off(fmt: str) -> None:
    """Test B: with fusion off, the primitive chain converts with parity too."""
    model = _decomposed_attention("Mul")
    inputs = _attention_inputs()

    # fuse=False must still convert (via MatMul/Mul/Softmax/MatMul) and match.
    mlmodel = o2c.convert(
        model, format=fmt, compute_precision="fp32", compute_units="cpu_only", fuse=False
    )
    assert mlmodel.get_spec() is not None
    _assert_predict_parity(mlmodel, model, inputs, fmt)


def _assert_predict_parity(mlmodel, model, inputs, fmt) -> None:
    """Run prediction parity for an already-converted model (fuse=off path).

    Mirrors helpers.assert_parity's numeric leg but for a model we converted
    ourselves with ``fuse=False``; off-device it degrades to the build-only check.
    """
    import platform

    from helpers import _predict, run_onnxruntime

    if platform.system() != "Darwin":
        return
    expected = run_onnxruntime(model, inputs)
    got = _predict(mlmodel, model, inputs)
    assert len(got) == len(expected)
    for e, g in zip(expected, got, strict=True):
        np.testing.assert_allclose(
            np.asarray(g, np.float64),
            np.asarray(e, np.float64),
            rtol=1e-3,
            atol=1e-4,
        )
