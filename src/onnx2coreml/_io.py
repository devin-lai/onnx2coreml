# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Loading ONNX models and saving Core ML models."""

from __future__ import annotations

from pathlib import Path

import onnx

from .errors import ModelValidationError


def load(model: onnx.ModelProto | str | Path | bytes) -> onnx.ModelProto:
    """Load and validate an ONNX model from a proto, path, or serialized bytes."""
    try:
        if isinstance(model, onnx.ModelProto):
            proto = model
        elif isinstance(model, (bytes, bytearray)):
            proto = onnx.load_model_from_string(bytes(model))
        else:
            proto = onnx.load_model(str(model))
    except Exception as exc:
        raise ModelValidationError(f"failed to load ONNX model: {exc}") from exc

    try:
        onnx.checker.check_model(proto)
    except Exception as exc:
        raise ModelValidationError(f"ONNX model failed validation: {exc}") from exc
    return proto


def save(mlmodel, path: str | Path) -> str:
    """Save a coremltools ``MLModel`` to disk; returns the path."""
    mlmodel.save(str(path))
    return str(path)
