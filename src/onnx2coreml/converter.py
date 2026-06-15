# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The conversion orchestrator: ONNX graph -> MIL Program.

Runs the coverage gate, materializes initializers as constants, then walks the
graph in topological order dispatching each node to its registered lowering.
"""

from __future__ import annotations

from typing import Any

import onnx
from onnx import numpy_helper

from . import _fusion, _lowering, _passes
from ._lowering import LoweringContext
from ._mil import Function, Program, ct, mb
from ._types import narrow_array, onnx_dtype_to_mil, saturate_fp16
from ._utils import iter_graph_nodes, op_key, topo_iter
from .errors import ConversionError, Onnx2CoreMLError, UnsupportedOpError

# MIL functions are always built at a concrete opset; the backend then targets
# the requested container. iOS17 is a safe, broadly-supported authoring opset.
_DEFAULT_BUILD_TARGET = ct.target.iOS17


def _static_shape(vi: onnx.ValueInfoProto) -> tuple[int, ...]:
    tt = vi.type.tensor_type
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            raise Onnx2CoreMLError(
                f"input '{vi.name}' has a dynamic or unknown dimension; "
                "fixed input shapes are required in this version"
            )
    return tuple(dims)


class Converter:
    """Lowers ONNX models to MIL. Reusable; holds per-instance custom lowerings."""

    def __init__(self) -> None:
        self._custom: dict[str, _lowering.Lowering] = {}

    def register(self, op_type: str):
        """Decorator registering a custom lowering for ``op_type`` on this
        converter instance. The escape hatch for ops without a built-in lowering.
        """

        def deco(fn: _lowering.Lowering) -> _lowering.Lowering:
            self._custom[op_type] = fn
            return fn

        return deco

    def to_mil(self, model: onnx.ModelProto, *, target=None, precision=None, fuse=True) -> Program:
        """Convert an ONNX model to a MIL ``Program`` with a single ``main``
        function. ``precision`` is honored only to saturate constants into the
        fp16 range (so the backend's fp16 cast cannot create ``inf``); the backend
        otherwise applies precision during serialization."""
        model = _passes.run(model)
        if fuse:
            model = _fusion.run(model)
        graph = model.graph
        self._check_coverage(graph)

        build_target = target if target is not None else _DEFAULT_BUILD_TARGET
        fp16 = precision is not None and precision == ct.precision.FLOAT16
        init_names = {i.name for i in graph.initializer}
        input_specs: dict[str, Any] = {}
        for vi in graph.input:
            if vi.name in init_names:
                continue  # initializer-as-input: handled as a constant below
            dtype = onnx_dtype_to_mil(vi.type.tensor_type.elem_type)
            input_specs[vi.name] = mb.placeholder(shape=_static_shape(vi), dtype=dtype)

        prog = Program()
        func = self._build_function(graph, input_specs, build_target, fp16=fp16)
        prog.add_function("main", func)
        return prog

    def _build_function(
        self, graph: onnx.GraphProto, input_specs: dict[str, Any], target, *, fp16: bool = False
    ):
        try:
            func_ctx = Function(input_specs, opset_version=target)
        except TypeError:  # older coremltools without the kwarg
            func_ctx = Function(input_specs)
        with func_ctx as func:
            values_map: dict[str, Any] = dict(func.inputs)
            for init in graph.initializer:
                arr = narrow_array(
                    numpy_helper.to_array(init), context=f"initializer '{init.name}'"
                )
                if fp16:
                    arr = saturate_fp16(arr)
                values_map[init.name] = mb.const(val=arr, name=init.name)
            ctx = LoweringContext(values_map=values_map, target=target, converter=self)
            for node in topo_iter(graph):
                self._lower_node(ctx, node)
            func.set_outputs([values_map[o.name] for o in graph.output])
        return func

    def _lower_node(self, ctx: LoweringContext, node: onnx.NodeProto) -> None:
        key = op_key(node)
        fn = self._custom.get(key) or _lowering.resolve(key)
        if fn is None:  # coverage gate should have caught this
            raise UnsupportedOpError({key: [node.name or node.op_type]})
        try:
            result = fn(ctx, node)
        except Onnx2CoreMLError:
            raise
        except Exception as exc:
            raise ConversionError(node.name or node.op_type, key, exc) from exc

        outs = list(result) if isinstance(result, (list, tuple)) else [result]
        if len(outs) != len(node.output):
            raise ConversionError(
                node.name or node.op_type,
                key,
                ValueError(f"produced {len(outs)} output(s), expected {len(node.output)}"),
            )
        for name, var in zip(node.output, outs, strict=True):
            ctx.values_map[name] = var

    def _check_coverage(self, graph: onnx.GraphProto) -> None:
        missing: dict[str, list[str]] = {}
        for i, node in enumerate(iter_graph_nodes(graph)):
            key = op_key(node)
            if key not in self._custom and _lowering.resolve(key) is None:
                missing.setdefault(key, []).append(node.name or f"node_{i}")
        if missing:
            raise UnsupportedOpError(missing)
