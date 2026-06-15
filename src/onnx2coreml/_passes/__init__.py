# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""ONNX graph preprocessing passes run before lowering.

``run`` applies the pass pipeline (opset normalization, shape/type inference,
constant folding, cleanup) and returns a possibly-new model. Every pass is
semantics-preserving and defensive: it is invoked through ``_safe`` so that if
it raises, the failure is swallowed and the pass is skipped, leaving the model
exactly as it entered. A preprocessing pass can never break a conversion that
already worked.
"""

from __future__ import annotations

from collections.abc import Callable

import onnx

from ._cleanup import (
    eliminate_dead_nodes,
    prune_initializers,
    remove_dropout,
    remove_identity,
)
from ._fold import fold_constants
from ._model import infer, normalize_opset

# Pipeline order: bring the opset up to the lowerings' baseline, annotate shapes,
# fold constants (which may create dead nodes / orphan initializers), then clean
# up — identity/dropout removal, dead-node elimination, initializer pruning.
_PIPELINE: tuple[Callable[[onnx.ModelProto], onnx.ModelProto], ...] = (
    normalize_opset,
    infer,
    fold_constants,
    remove_identity,
    remove_dropout,
    eliminate_dead_nodes,
    prune_initializers,
)


def _safe(
    pass_fn: Callable[[onnx.ModelProto], onnx.ModelProto], model: onnx.ModelProto
) -> onnx.ModelProto:
    """Run ``pass_fn`` defensively: any exception skips the pass unchanged."""
    try:
        return pass_fn(model)
    except Exception:
        return model


def run(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply the preprocessing pipeline and return the transformed model."""
    for pass_fn in _PIPELINE:
        model = _safe(pass_fn, model)
    return model
