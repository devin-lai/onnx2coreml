# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The single seam between onnx2coreml and coremltools.

Every coremltools import funnels through this module. If coremltools
reorganizes its internals or bumps the Core ML spec, this file (and ``_backend``)
absorb the change — the operator lowerings, passes, and fusions never import
coremltools directly. This is the maintainability contract from the design spec.
"""

from __future__ import annotations

import coremltools as ct
from coremltools import ComputeUnit, precision
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types

__all__ = [
    "ComputeUnit",
    "Function",
    "Program",
    "ct",
    "mb",
    "precision",
    "types",
]
