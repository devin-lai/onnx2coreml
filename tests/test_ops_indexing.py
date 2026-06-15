# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for indexing / gather lowerings.

Index and shape operands are passed as int64 initializers (Slice's
starts/ends/axes/steps, Gather's indices, Expand/Tile/ConstantOfShape's shapes);
the converter narrows them to int32 on the way in. Integer-valued outputs (Shape,
and Gather of an int tensor) are compared exactly by the harness; float outputs use
the default fp32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model

pytestmark = pytest.mark.ops

_FMTS = ["mlpackage", "mlmodel"]

_X45 = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize(
    "indices",
    [
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 2, -1], dtype=np.int64),  # negative index wraps
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_gather(fmt, axis, indices):
    """Gather along axis 0/1, including a negative index (MIL handles natively)."""
    model = single_op_model(
        "Gather",
        {"x": _X45},
        attrs={"axis": axis},
        initializers={"indices": indices},
    )
    assert_parity(model, {"x": _X45}, fmt=fmt)


def test_gather_int_exact():
    """Gather of an int tensor: output is int and compared exactly.

    ``x`` is int32 (Core ML's native integer compute dtype) so it can be a model
    input directly; the result stays int and is compared exactly by the harness.
    """
    x = (np.arange(4 * 5, dtype=np.int32).reshape(4, 5) * 2) - 11
    model = single_op_model(
        "Gather",
        {"x": x},
        attrs={"axis": 1},
        initializers={"indices": np.array([4, 0, 2], dtype=np.int64)},
    )
    assert_parity(model, {"x": x})


# Slice cases: (starts, ends, axes, steps). axes/steps may be None (omitted).
_SLICE_CASES = [
    ([1], [3], [0], None),  # basic [1:3] on axis 0
    ([1, 0], [3, 4], [0, 1], None),  # multi-axis
    ([-3], [-1], [0], None),  # negative start/end
    ([0], [100], [0], None),  # end overflow -> clamps to dim
    ([0], [5], [1], [2]),  # step 2 on axis 1
    ([1, 1], [4, 5], [0, 1], [2, 2]),  # multi-axis with step 2
    ([4], [1], [1], [-1]),  # negative step: partial reverse [4:1:-1]
    ([4], [-6], [1], [-1]),  # negative step to the start (end_mask branch)
    ([3], [-5], [0], [-1]),  # full reverse on axis 0
]


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize(("starts", "ends", "axes", "steps"), _SLICE_CASES)
def test_slice(fmt, starts, ends, axes, steps):
    """ONNX Slice (opset 10+) with starts/ends/axes/steps as int64 initializers."""
    inits = {
        "starts": np.array(starts, dtype=np.int64),
        "ends": np.array(ends, dtype=np.int64),
    }
    if axes is not None:
        inits["axes"] = np.array(axes, dtype=np.int64)
    if steps is not None:
        inits["steps"] = np.array(steps, dtype=np.int64)
    model = single_op_model("Slice", {"x": _X45}, initializers=inits)
    assert_parity(model, {"x": _X45}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize(
    ("in_shape", "target"),
    [
        ((3, 1), [3, 4]),  # broadcast last dim
        ((1,), [2, 3]),  # bidirectional broadcast adds a dim
    ],
)
def test_expand(fmt, in_shape, target):
    """ONNX Expand: bidirectional broadcast of x to `shape` (int64 initializer)."""
    x = np.arange(int(np.prod(in_shape)), dtype=np.float32).reshape(in_shape)
    model = single_op_model(
        "Expand", {"x": x}, initializers={"shape": np.array(target, dtype=np.int64)}
    )
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("reps", [[2, 1], [1, 3], [2, 3]])
def test_tile(fmt, reps):
    """ONNX Tile: replicate (2,3) input by `repeats` (int64 initializer)."""
    x = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    model = single_op_model(
        "Tile", {"x": x}, initializers={"repeats": np.array(reps, dtype=np.int64)}
    )
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize(
    "attrs",
    [
        {},  # full shape -> [2, 3, 4]
        {"start": 1},  # [3, 4]
        {"start": 0, "end": 2},  # [2, 3]
        {"start": -2},  # [3, 4]
    ],
)
def test_shape(fmt, attrs):
    """ONNX Shape (opset 15): int64 shape vector, optionally sliced by start/end."""
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    model = single_op_model("Shape", {"x": x}, attrs=attrs)
    assert_parity(model, {"x": x}, fmt=fmt)


def _constant_of_shape_model(out_shape, value):
    """Build a ConstantOfShape model with the shape as an int64 initializer.

    Core ML requires every model to have at least one input, so we declare an
    unused float ``dummy`` input — ConstantOfShape itself is input-less in ONNX.
    """
    import onnx
    from onnx import TensorProto, helper

    val_tensor = helper.make_tensor("value", TensorProto.FLOAT, [1], [value])
    node = helper.make_node(
        "ConstantOfShape", ["in_shape"], ["out0"], value=val_tensor
    )
    graph = helper.make_graph(
        [node],
        "test_ConstantOfShape",
        [helper.make_tensor_value_info("dummy", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
        initializer=[
            onnx.numpy_helper.from_array(
                np.array(out_shape, dtype=np.int64), name="in_shape"
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("value", [0.0, 2.5])
def test_constant_of_shape_float(fmt, value):
    """ConstantOfShape with a float `value` attr (default 0.0 and a nonzero)."""
    model = _constant_of_shape_model([2, 3], value)
    assert_parity(model, {"dummy": np.zeros((1,), dtype=np.float32)}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axis", [0, 1])
def test_gather_elements(fmt, axis):
    dim = _X45.shape[axis]
    idx = (np.arange(20).reshape(4, 5) % dim).astype(np.int64)
    model = single_op_model(
        "GatherElements", {"data": _X45}, attrs={"axis": axis}, initializers={"indices": idx}
    )
    assert_parity(model, {"data": _X45}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_gather_nd(fmt):
    # Full 2-D coordinates -> one gathered scalar per index row.
    idx = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.int64)
    model = single_op_model("GatherND", {"data": _X45}, initializers={"indices": idx})
    assert_parity(model, {"data": _X45}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("reduction", ["none", "add"])
def test_scatter_nd(fmt, reduction):
    idx = np.array([[0], [2]], dtype=np.int64)  # distinct rows -> unambiguous
    upd = np.full((2, 5), 7.0, dtype=np.float32)
    model = single_op_model(
        "ScatterND", {"data": _X45}, attrs={"reduction": reduction},
        initializers={"indices": idx, "updates": upd},
    )
    assert_parity(model, {"data": _X45}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("reduction", ["none", "add"])
def test_scatter_elements(fmt, reduction):
    # Each row indexes a full permutation of columns -> no duplicate targets.
    idx = (np.arange(20).reshape(4, 5) % 5).astype(np.int64)
    upd = np.full((4, 5), 3.0, dtype=np.float32)
    model = single_op_model(
        "ScatterElements", {"data": _X45}, attrs={"axis": 1, "reduction": reduction},
        initializers={"indices": idx, "updates": upd},
    )
    assert_parity(model, {"data": _X45}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_nonzero(fmt):
    x = np.array([[0, 1, 0, 2], [3, 0, 0, 0], [0, 0, 5, 0]], dtype=np.float32)
    assert_parity(single_op_model("NonZero", {"X": x}), {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_expand_bool(fmt):
    # A bool tensor (produced inside the graph, since Core ML I/O has no bool) is
    # Expand-ed, then consumed by Where so the output is float.
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    a = np.array([[1.0], [2.0]], dtype=np.float32)
    b = np.array([[2.0], [1.0]], dtype=np.float32)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = (np.arange(6, dtype=np.float32) + 100).reshape(2, 3)
    nodes = [
        helper.make_node("Greater", ["a", "b"], ["cond"]),
        helper.make_node("Expand", ["cond", "shape"], ["cond_e"]),
        helper.make_node("Where", ["cond_e", "x", "y"], ["out0"]),
    ]
    graph = helper.make_graph(
        nodes, "expand_bool",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
         for n, s in [("a", (2, 1)), ("b", (2, 1)), ("x", (2, 3)), ("y", (2, 3))]],
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
        initializer=[numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "shape")],
    )
    model = onnx.shape_inference.infer_shapes(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    )
    assert_parity(model, {"a": a, "b": b, "x": x, "y": y}, fmt=fmt)
