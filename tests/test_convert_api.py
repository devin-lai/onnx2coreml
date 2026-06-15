# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Public-API and coverage-gate tests."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import single_op_model

import onnx2coreml as o2c
from onnx2coreml.errors import TargetError, UnsupportedOpError


def test_supported_ops_includes_add():
    assert "Add" in o2c.supported_ops()


def test_unsupported_op_raises_aggregated_error():
    # Det has no lowering — the unsupported-op probe.
    model = single_op_model("Det", {"X": np.zeros((3, 3), dtype=np.float32)})
    with pytest.raises(UnsupportedOpError) as excinfo:
        o2c.convert(model)
    assert "Det" in str(excinfo.value)


def test_analyze_reports_supported_and_unsupported():
    model = single_op_model("Det", {"X": np.zeros((3, 3), dtype=np.float32)})
    report = o2c.analyze(model)
    assert "Det" in report.unsupported
    assert not report.convertible


def test_bad_format_raises_target_error():
    model = single_op_model("Add", {"a": np.zeros((2,), np.float32), "b": np.zeros((2,), np.float32)})
    with pytest.raises(TargetError):
        o2c.convert(model, format="onnx")


def test_fp16_saturates_overflowing_constant():
    # A constant beyond fp16 range (FLT_MAX, as used for attention-mask fills)
    # becomes inf once cast to fp16; 0 * inf = NaN. Saturating the constant into
    # the fp16 range keeps it finite. Build everywhere; predict only on macOS.
    import platform

    x = np.zeros((1, 4), dtype=np.float32)
    big = np.array(3.4e38, dtype=np.float32)
    model = single_op_model("Mul", {"x": x}, initializers={"big": big})
    mlmodel = o2c.convert(
        model, format="mlpackage", compute_precision="fp16",
        minimum_deployment_target="iOS17",
    )
    assert mlmodel.get_spec() is not None
    if platform.system() != "Darwin":
        return
    spec = mlmodel.get_spec()
    out = mlmodel.predict({spec.description.input[0].name: x})
    v = np.asarray(out[spec.description.output[0].name])
    assert np.isfinite(v).all(), "fp16 constant overflowed to inf -> NaN"
    np.testing.assert_allclose(v, 0.0)


@pytest.mark.parametrize("fp32_ops", [None, {"sigmoid"}, {"nonexistent_op"}])
def test_fp32_op_types_accepted_and_correct(fp32_ops):
    # fp32_op_types must be honored without changing correctness (the kept op runs
    # fp32, the rest fp16); an unknown op name is a harmless no-op.
    import platform

    x = ((np.arange(12, dtype=np.float32).reshape(3, 4) - 6.0) * 0.5)
    model = single_op_model("Sigmoid", {"x": x})
    ml = o2c.convert(
        model, format="mlpackage", compute_precision="fp16",
        minimum_deployment_target="iOS17", fp32_op_types=fp32_ops,
    )
    assert ml.get_spec() is not None
    if platform.system() != "Darwin":
        return
    spec = ml.get_spec()
    v = np.asarray(ml.predict({spec.description.input[0].name: x})[spec.description.output[0].name])
    np.testing.assert_allclose(v, 1.0 / (1.0 + np.exp(-x)), rtol=2e-2)
