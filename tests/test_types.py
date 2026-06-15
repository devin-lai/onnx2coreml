# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the 64->32-bit narrowing policy."""

from __future__ import annotations

import numpy as np

from onnx2coreml._types import narrow_array


def test_narrow_int64_in_range_unchanged() -> None:
    out = narrow_array(np.array([1, 2, 3], dtype=np.int64))
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, [1, 2, 3])


def test_narrow_int64_saturates_sentinels() -> None:
    # ONNX exporters emit INT64_MAX/MIN as "to the end" sentinels (Slice bounds,
    # mask fills). Core ML is 32-bit, so these must saturate to the int32 range
    # rather than raise — the clamped value is still an out-of-bounds sentinel.
    i64 = np.iinfo(np.int64)
    i32 = np.iinfo(np.int32)
    out = narrow_array(np.array([i64.max, i64.min, 7], dtype=np.int64))
    assert out.dtype == np.int32
    assert out[0] == i32.max
    assert out[1] == i32.min
    assert out[2] == 7


def test_narrow_float64_to_float32() -> None:
    out = narrow_array(np.array([1.5, 2.5], dtype=np.float64))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.5, 2.5])


def test_saturate_fp16_clamps_overflow() -> None:
    from onnx2coreml._types import saturate_fp16

    out = saturate_fp16(np.array([3.4e38, -3.4e38, 1.5, -2.0], dtype=np.float32))
    assert out[0] == 65504.0
    assert out[1] == -65504.0
    assert out[2] == 1.5
    assert out[3] == -2.0
    assert out.dtype == np.float32


def test_saturate_fp16_noop_in_range() -> None:
    from onnx2coreml._types import saturate_fp16

    a = np.array([1.0, -2.0, 100.0], dtype=np.float32)
    assert saturate_fp16(a) is a  # untouched when already within range


def test_saturate_fp16_ignores_non_float() -> None:
    from onnx2coreml._types import saturate_fp16

    a = np.array([1, 2, 3], dtype=np.int32)
    assert saturate_fp16(a) is a
