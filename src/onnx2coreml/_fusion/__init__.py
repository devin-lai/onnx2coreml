# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Semantics-preserving graph fusion passes run after preprocessing.

``run`` rewrites recognizable subgraph patterns and returns a possibly-new
model. The v1 pipeline fuses the canonical decomposed scaled-dot-product
attention chain into a single ScaledDotProductAttention node.

Every fusion is applied conservatively: a graph that does not match a precise
pattern passes through unchanged, and each fusion runs on a private copy of the
model behind a guard, so an unexpected failure skips that fusion rather than
breaking the conversion. ``run`` executes on every ``fuse=True`` conversion, so
this safety is load-bearing.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import onnx

from . import _attention

# Fusions run in order; each takes a model and returns a (possibly mutated) model.
_FUSIONS: tuple[Callable[[onnx.ModelProto], onnx.ModelProto], ...] = (
    _attention.fuse,
)


def run(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply the fusion pipeline and return the transformed model.

    Each step is attempted on a deep copy; only a successful, exception-free
    step is committed. Any failure leaves the pre-step model in place, so a
    buggy or partially-applied fusion can never corrupt the conversion.
    """
    for fuse in _FUSIONS:
        try:
            candidate = fuse(copy.deepcopy(model))
        except Exception:
            # Defensive: a fusion must never break an otherwise-valid conversion.
            continue
        if candidate is not None:
            model = candidate
    return model
