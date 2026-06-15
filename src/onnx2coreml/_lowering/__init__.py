# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Operator-lowering registry.

Each ``_lowering/_*.py`` module exposes a ``REGISTRY: dict[str, Lowering]``.
This package merges them all into one resolver, raising at import time on any
duplicate op key so two modules can never silently shadow each other.
"""

from __future__ import annotations

from . import (
    _activations,
    _attention,
    _conv,
    _elementwise,
    _indexing,
    _matmul,
    _norm,
    _recurrent,
    _reduce,
    _shape,
)
from ._context import Lowering, LoweringContext

# Every operator-family module contributes its REGISTRY here.
_MODULES = (
    _elementwise,
    _activations,
    _conv,
    _norm,
    _matmul,
    _shape,
    _reduce,
    _indexing,
    _attention,
    _recurrent,
)

RESOLVER: dict[str, Lowering] = {}
for _module in _MODULES:
    for _key, _fn in _module.REGISTRY.items():
        if _key in RESOLVER:
            raise RuntimeError(f"duplicate ONNX op lowering registered: {_key!r}")
        RESOLVER[_key] = _fn


def resolve(op_key: str) -> Lowering | None:
    """Return the lowering for an op key, or ``None`` if unsupported."""
    return RESOLVER.get(op_key)


def supported_keys() -> set[str]:
    """The set of op keys with a built-in lowering."""
    return set(RESOLVER)


__all__ = ["Lowering", "LoweringContext", "resolve", "supported_keys"]
