# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ``onnx2coreml`` command-line interface."""

from __future__ import annotations

import json

import numpy as np
import onnx
from helpers import single_op_model

from onnx2coreml._cli import main


def _write_add_model(tmp_path) -> str:
    model = single_op_model(
        "Add",
        {
            "a": np.zeros((2, 3), dtype=np.float32),
            "b": np.zeros((2, 3), dtype=np.float32),
        },
    )
    path = tmp_path / "add.onnx"
    onnx.save_model(model, str(path))
    return str(path)


def test_schema_json(capsys):
    rc = main(["schema", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "version" in out
    assert isinstance(out["supported_op_count"], int)
    assert out["supported_op_count"] > 0
    assert "error_codes" in out
    assert "UnsupportedOpError" in out["error_codes"]


def test_inspect_json(tmp_path, capsys):
    model_path = _write_add_model(tmp_path)
    rc = main(["inspect", model_path, "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["convertible"] is True
    assert "Add" in out["supported"]
    assert out["formats"]["mlpackage"] is True
    assert out["formats"]["mlmodel"] is True


def test_inspect_json_unsupported_exits_nonzero(tmp_path, capsys):
    model = single_op_model("Det", {"X": np.zeros((3, 3), dtype=np.float32)})
    path = tmp_path / "det.onnx"
    onnx.save_model(model, str(path))
    rc = main(["inspect", str(path), "--json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["convertible"] is False
    assert "Det" in out["unsupported"]


def test_convert_json(tmp_path, capsys):
    model_path = _write_add_model(tmp_path)
    out_path = tmp_path / "add.mlpackage"
    rc = main(["convert", model_path, "-o", str(out_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["output"] == str(out_path)
    assert out["format"] == "mlpackage"
    assert out_path.exists()


def test_convert_human_output(tmp_path, capsys):
    model_path = _write_add_model(tmp_path)
    out_path = tmp_path / "add2.mlpackage"
    rc = main(["convert", model_path, "-o", str(out_path)])
    assert rc == 0
    assert "wrote" in capsys.readouterr().out
    assert out_path.exists()


def test_no_command_returns_usage(capsys):
    rc = main([])
    assert rc == 2


def test_convert_bad_model_reports_error_json(tmp_path, capsys):
    bad = tmp_path / "missing.onnx"
    rc = main(["convert", str(bad), "-o", str(tmp_path / "x.mlpackage"), "--json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out
    assert "message" in out
