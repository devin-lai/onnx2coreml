# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constant folding: evaluate all-constant nodes into initializers.

A node every input of which is an initializer (or an omitted optional) computes
a value that never changes, so it can be replaced by that value as a new
initializer. Evaluation is delegated to ONNX Runtime, which is an *optional*
dependency: if it is not importable, or a node cannot be evaluated, that node is
left in place. The pass thus only ever removes work; it never changes results.
"""

from __future__ import annotations

import onnx
from onnx import helper, numpy_helper

# Nodes carrying these attribute types embed subgraphs (If/Loop/Scan). Folding
# them standalone is fragile, so we never treat them as constant.
_SUBGRAPH_ATTR_TYPES = (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)


def _has_subgraph(node: onnx.NodeProto) -> bool:
    return any(attr.type in _SUBGRAPH_ATTR_TYPES for attr in node.attribute)


def _is_foldable(node: onnx.NodeProto, const_names: set[str]) -> bool:
    """A node is foldable when it has no subgraph body, every (non-omitted)
    input is a known constant, and it produces at least one named output."""
    if _has_subgraph(node):
        return False
    if not any(node.output):
        return False
    return all(name in const_names for name in node.input if name)


def _evaluate(model: onnx.ModelProto, node: onnx.NodeProto, pool: dict[str, onnx.TensorProto]):
    """Run a single node over the constant ``pool`` via ONNX Runtime and return
    the list of output arrays, or ``None`` if evaluation is unavailable/fails."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    try:
        inits = [pool[name] for name in node.input if name]
        out_names = [name for name in node.output if name]
        outputs = [helper.make_empty_tensor_value_info(name) for name in out_names]
        sub_graph = helper.make_graph([node], "fold", [], outputs, initializer=inits)
        sub_model = helper.make_model(sub_graph, opset_imports=list(model.opset_import))
        sub_model.ir_version = model.ir_version
        sess = ort.InferenceSession(
            sub_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(out_names, {})
    except Exception:
        return None


def fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace nodes whose inputs are all constant with precomputed initializers.

    Folds to a fixpoint so constant chains collapse fully. Operates on a copy;
    graph input/output names and order are preserved. Returns the input model
    unchanged if ONNX Runtime is unavailable or nothing can be folded.
    """
    folded = onnx.ModelProto()
    folded.CopyFrom(model)
    graph = folded.graph

    const_names = {init.name for init in graph.initializer}
    # The runnable constant pool, keyed by tensor name -> initializer proto.
    pool = {init.name: init for init in graph.initializer}

    changed_any = False
    progressed = True
    while progressed:
        progressed = False
        for node in list(graph.node):
            if not _is_foldable(node, const_names):
                continue
            results = _evaluate(folded, node, pool)
            if results is None:
                continue

            out_names = [name for name in node.output if name]
            for name, array in zip(out_names, results, strict=True):
                tensor = numpy_helper.from_array(array, name=name)
                graph.initializer.append(tensor)
                const_names.add(name)
                pool[name] = tensor
            graph.node.remove(node)
            progressed = True
            changed_any = True

    return folded if changed_any else model
