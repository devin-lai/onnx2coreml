# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the numerical-verification module."""

from __future__ import annotations

import numpy as np
import onnx
from helpers import _has_coreml_runtime, requires_predict, single_op_model

import onnx2coreml as o2c
from onnx2coreml._verify import VerifyReport, generate_inputs, verify_model


def _add_model() -> onnx.ModelProto:
    return single_op_model(
        "Add",
        {
            "a": np.zeros((2, 3), dtype=np.float32),
            "b": np.zeros((2, 3), dtype=np.float32),
        },
    )


def test_generate_inputs_is_deterministic():
    model = _add_model()
    a = generate_inputs(model)
    b = generate_inputs(model)
    assert set(a) == {"a", "b"}
    for name in a:
        assert a[name].shape == (2, 3)
        assert a[name].dtype == np.float32
        np.testing.assert_array_equal(a[name], b[name])  # seeded -> reproducible


@requires_predict
def test_verify_model_passes_for_faithful_conversion():
    model = _add_model()
    mlmodel = o2c.convert(model, compute_precision="fp32", compute_units="cpu_only")
    report = verify_model(model, mlmodel)
    assert isinstance(report, VerifyReport)
    assert report.passed is True
    assert len(report.outputs) == 1
    m = report.outputs[0]
    assert m.max_abs_err >= 0.0
    assert m.max_rel_err >= 0.0
    assert m.psnr > 0.0  # finite-or-inf, but positive
    assert "PASS" in str(report)


@requires_predict
def test_verify_model_fails_against_wrong_reference():
    # Convert Add, but verify against a Sub reference over the same inputs: same
    # shapes/dtypes, different math -> parity must fail.
    add = _add_model()
    sub = single_op_model(
        "Sub",
        {
            "a": np.zeros((2, 3), dtype=np.float32),
            "b": np.zeros((2, 3), dtype=np.float32),
        },
    )
    mlmodel = o2c.convert(add, compute_precision="fp32", compute_units="cpu_only")
    report = verify_model(sub, mlmodel)
    assert report.passed is False
    assert "FAIL" in str(report)


@requires_predict
def test_verify_public_entry_accepts_saved_path(tmp_path):
    model = _add_model()
    mlmodel = o2c.convert(model, compute_precision="fp32", compute_units="cpu_only")
    out = tmp_path / "add.mlpackage"
    mlmodel.save(str(out))
    report = o2c.verify(model, str(out))
    assert report.passed is True


def test_verify_report_str_off_device():
    # __str__ must work without the Core ML runtime (no predict involved).
    report = VerifyReport(passed=True, rtol=1e-3, atol=1e-4)
    text = str(report)
    assert "PASS" in text
    assert "rtol=0.001" in text
    if not _has_coreml_runtime():
        # Smoke-check that the module imports and the dataclass is usable.
        assert report.as_dict()["passed"] is True
