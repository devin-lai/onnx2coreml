# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the structured error hierarchy."""

from __future__ import annotations

from onnx2coreml.errors import ConversionError, UnsupportedOpError


def test_unsupported_op_message_aggregates_and_truncates():
    err = UnsupportedOpError({"Foo": ["n1", "n2", "n3", "n4"], "Bar": ["b1"]})
    msg = str(err)
    assert "Foo" in msg
    assert "Bar" in msg
    assert "n1" in msg
    assert "n3" in msg
    assert "+1 more" in msg  # 4 nodes -> shows 3 + "(+1 more)"
    assert err.missing["Bar"] == ["b1"]


def test_conversion_error_carries_context():
    cause = ValueError("boom")
    err = ConversionError("nodeA", "Bar", cause)
    assert err.node_name == "nodeA"
    assert err.op_key == "Bar"
    assert err.cause is cause
    msg = str(err)
    assert "nodeA" in msg
    assert "Bar" in msg
    assert "boom" in msg
