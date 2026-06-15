# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model-level passes: default-domain opset normalization and shape inference.

Both passes are best-effort. Lowerings are written against a modern opset
(>= ``baseline``); normalizing low-opset models up front lets them reuse those
lowerings. Shape inference populates ``value_info`` so downstream passes and
lowerings can read static shapes/dtypes. Either pass returns the input model
unchanged if it cannot run cleanly.
"""

from __future__ import annotations

import onnx

_DEFAULT_DOMAINS = ("", "ai.onnx")


def _default_opset_version(model: onnx.ModelProto) -> int | None:
    """Return the model's default-domain (``ai.onnx``) opset version, or ``None``
    if it declares no default-domain import."""
    for op in model.opset_import:
        if op.domain in _DEFAULT_DOMAINS:
            return op.version
    return None


def normalize_opset(model: onnx.ModelProto, baseline: int = 13) -> onnx.ModelProto:
    """Best-effort upgrade of the default-domain opset to at least ``baseline``.

    Only *upgrades*: if the model already imports the default domain at
    ``>= baseline`` (or imports no default domain at all) it is returned
    untouched. Downgrading is never attempted — ``convert_version`` lacks
    down-adapters for many ops and would raise. Any failure of the underlying
    ``onnx.version_converter`` is swallowed and the original model returned, so
    this pass can never break a conversion that already worked.
    """
    try:
        current = _default_opset_version(model)
        if current is None or current >= baseline:
            return model
        return onnx.version_converter.convert_version(model, baseline)
    except Exception:
        return model


def infer(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run ONNX shape/type inference, returning the annotated model.

    Inference is advisory here: on any failure (unsupported op, malformed graph)
    the original model is returned unchanged.
    """
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception:
        return model
