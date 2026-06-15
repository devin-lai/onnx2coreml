# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for reduction and arg-reduction lowerings.

One combination is an inherent Core ML limitation rather than a lowering bug, and
is marked ``xfail`` with the reason inline: reducing over *all* axes with
``keepdims=0`` produces an ONNX rank-0 scalar ``()``, but Core ML has no rank-0
tensor type and emits shape ``(1,)`` instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model

pytestmark = pytest.mark.ops

_FMTS = ["mlpackage", "mlmodel"]
# 3-D input; distinct values keep argmax/argmin tie-free.
_X = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) * 0.5 - 7.0
# Strictly positive variant so ReduceProd stays in a sane numeric range.
_XPOS = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) % 5) + 1.0

# axes variants: single axis, multiple axes, all axes.
_AXES = [[1], [0, 2], [0, 1, 2]]

# Core ML cannot represent a rank-0 scalar: a full reduction with keepdims=0
# collapses to () in ONNX but (1,) in Core ML.
_RANK0_XFAIL = pytest.mark.xfail(
    reason="Core ML has no rank-0 tensor: all-axes keepdims=0 yields (1,) not ()",
    strict=True,
)


def _maybe_rank0(axes, keepdims, rank=3):
    """Mark the all-axes + keepdims=0 case, which has no Core ML rank-0 output."""
    reduces_all = len(axes) == rank
    return [_RANK0_XFAIL] if reduces_all and keepdims == 0 else []


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axes", _AXES)
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("op", ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd"])
def test_reduce_axes_attr(request, op, keepdims, axes, fmt):
    """ReduceMean/Max/Min/Prod take axes via the opset-17 ``axes`` attribute."""
    for mark in _maybe_rank0(axes, keepdims):
        request.applymarker(mark)
    x = _XPOS if op == "ReduceProd" else _X
    model = single_op_model(op, {"x": x}, attrs={"axes": axes, "keepdims": keepdims})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axes", _AXES)
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_sum_axes_input(request, keepdims, axes, fmt):
    """ReduceSum takes axes via input 1 (opset-13 form), passed as initializer."""
    for mark in _maybe_rank0(axes, keepdims):
        request.applymarker(mark)
    model = single_op_model(
        "ReduceSum",
        {"x": _X},
        attrs={"keepdims": keepdims},
        initializers={"axes": np.array(axes, dtype=np.int64)},
    )
    assert_parity(model, {"x": _X}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("keepdims", [1])
@pytest.mark.parametrize("op", ["ReduceMean", "ReduceSum"])
def test_reduce_all_axes_default(op, keepdims, fmt):
    """Omitted axes reduces over all dimensions (attr form and input form).

    Only keepdims=1 is exercised; keepdims=0 would yield a rank-0 scalar, which
    Core ML cannot represent (covered by the strict xfail above).
    """
    model = single_op_model(op, {"x": _X}, attrs={"keepdims": keepdims})
    assert_parity(model, {"x": _X}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("op", ["ArgMax", "ArgMin"])
def test_arg_reduce(op, keepdims, axis, fmt):
    """ArgMax/ArgMin: int64 indices, compared exactly by the harness."""
    model = single_op_model(op, {"x": _X}, attrs={"axis": axis, "keepdims": keepdims})
    assert_parity(model, {"x": _X}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axes", _AXES)
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("op", ["ReduceL1", "ReduceL2", "ReduceLogSumExp", "ReduceSumSquare"])
def test_reduce_norms(request, op, keepdims, axes, fmt):
    """L1/L2/LogSumExp/SumSquare reductions; signed input is fine for all."""
    for mark in _maybe_rank0(axes, keepdims):
        request.applymarker(mark)
    model = single_op_model(op, {"x": _X}, attrs={"axes": axes, "keepdims": keepdims})
    assert_parity(model, {"x": _X}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axes", _AXES)
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_logsum(request, keepdims, axes, fmt):
    """ReduceLogSum needs a strictly positive sum to stay real-valued."""
    for mark in _maybe_rank0(axes, keepdims):
        request.applymarker(mark)
    model = single_op_model("ReduceLogSum", {"x": _XPOS}, attrs={"axes": axes, "keepdims": keepdims})
    assert_parity(model, {"x": _XPOS}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("largest", [1, 0])
def test_topk(fmt, largest):
    # _X has distinct values along the last axis, so top-k ordering is unambiguous.
    model = single_op_model(
        "TopK", {"x": _X}, n_outputs=2,
        attrs={"axis": -1, "largest": largest, "sorted": 1},
        initializers={"K": np.array([2], dtype=np.int64)},
    )
    assert_parity(model, {"x": _X}, fmt=fmt)
