# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Elementwise arithmetic, comparison, and unary-math lowerings.

Most ops map one-to-one onto a MIL builder op via the :func:`binary` / :func:`unary`
helpers. The exceptions handled inline here are: ``Neg`` (MIL has no ``neg``, so it
becomes ``x * -1``), the variadic ``Min``/``Max`` (folded pairwise), ``Clip`` (whose
bounds may arrive as opset-11 inputs or opset-6 attributes, and may be absent), and
``Where`` (which maps to MIL ``select``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb, types
from ._common import binary, const_array, get_attr, operands, unary
from ._context import Lowering, LoweringContext


def _np_dtype(var: Any) -> np.dtype:
    """Return the numpy dtype backing a MIL Var, for building matching scalars."""
    return np.dtype(types.nptype_from_builtin(var.dtype))


def _neg(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """Negate ``x``. MIL has no ``neg`` op, so multiply by ``-1`` of x's dtype."""
    (x,) = operands(ctx.values_map, node, [0])
    return mb.mul(x=x, y=_np_dtype(x).type(-1), name=node.output[0])


def _variadic(mb_op: Any) -> Lowering:
    """Build a lowering for an ONNX variadic op (N inputs) by folding pairwise."""

    def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
        idxs = list(range(len(node.input)))
        vars_ = operands(ctx.values_map, node, idxs)
        acc = vars_[0]
        last = len(vars_) - 1
        for i, rhs in enumerate(vars_[1:], start=1):
            # Name only the final fold op with the output name; intermediate ops
            # must omit ``name`` entirely (passing None breaks the builder).
            if i == last:
                acc = mb_op(x=acc, y=rhs, name=node.output[0])
            else:
                acc = mb_op(x=acc, y=rhs)
        return acc

    return lower


def _clip(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """Clip ``x`` to ``[min, max]``.

    Bounds come from inputs 1/2 (opset>=11) or ``min``/``max`` attributes
    (opset 6). For floats we use MIL ``clip`` (which wants both bounds as
    constants, so a missing one falls back to the dtype's representable limit).
    MIL ``clip`` is float-only, so integer tensors are clamped with
    ``maximum``/``minimum`` against whichever bounds are present instead.
    """
    (x,) = operands(ctx.values_map, node, [0])
    dtype = _np_dtype(x)

    lo = const_array(ctx, node, 1)
    if lo is None:
        lo = get_attr(node, "min")
    hi = const_array(ctx, node, 2)
    if hi is None:
        hi = get_attr(node, "max")

    if dtype.kind == "f":
        finfo = np.finfo(dtype)
        alpha = finfo.min if lo is None else np.asarray(lo).item()
        beta = finfo.max if hi is None else np.asarray(hi).item()
        return mb.clip(x=x, alpha=dtype.type(alpha), beta=dtype.type(beta), name=node.output[0])

    # Integer clamp: compose max(lo, .) and min(hi, .); name only the final op.
    result = x
    ops = ([(mb.maximum, lo)] if lo is not None else []) + (
        [(mb.minimum, hi)] if hi is not None else []
    )
    if not ops:
        return mb.identity(x=x, name=node.output[0])
    for i, (mb_op, bound) in enumerate(ops):
        kw = {"name": node.output[0]} if i == len(ops) - 1 else {}
        result = mb_op(x=result, y=dtype.type(np.asarray(bound).item()), **kw)
    return result


def _where(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """Select from ``a``/``b`` by boolean ``cond`` (ONNX Where -> MIL select)."""
    cond, a, b = operands(ctx.values_map, node, [0, 1, 2])
    return mb.select(cond=cond, a=a, b=b, name=node.output[0])


def _isnan(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX IsNaN: NaN is the only value not equal to itself, so ``x != x``."""
    (x,) = operands(ctx.values_map, node, [0])
    return mb.not_equal(x=x, y=x, name=node.output[0])


def _pow(ctx: LoweringContext, node: onnx.NodeProto) -> Any:
    """ONNX Pow -> MIL ``pow``.

    The NeuralNetwork backend wants a scalar for constant exponents. A rank-1
    single-value initializer is equivalent under ONNX broadcasting, so collapse it
    before handing the op to MIL.
    """
    x, y = operands(ctx.values_map, node, [0, 1])
    exponent = const_array(ctx, node, 1)
    if exponent is not None and exponent.size == 1:
        y = _np_dtype(x).type(exponent.reshape(()).item())
    return mb.pow(x=x, y=y, name=node.output[0])


REGISTRY: dict[str, Lowering] = {
    # Arithmetic (two-input, broadcasting).
    "Add": binary(mb.add),
    "Sub": binary(mb.sub),
    "Mul": binary(mb.mul),
    "Div": binary(mb.real_div),
    "Pow": _pow,
    # Unary math.
    "Sqrt": unary(mb.sqrt),
    "Exp": unary(mb.exp),
    "Log": unary(mb.log),
    "Abs": unary(mb.abs),
    "Erf": unary(mb.erf),
    "Reciprocal": unary(mb.inverse),
    "Neg": _neg,
    "Floor": unary(mb.floor),
    "Ceil": unary(mb.ceil),
    "Round": unary(mb.round),  # both ONNX and MIL round half-to-even
    "Sign": unary(mb.sign),
    # Trigonometric / hyperbolic.
    "Sin": unary(mb.sin),
    "Cos": unary(mb.cos),
    "Tan": unary(mb.tan),
    "Asin": unary(mb.asin),
    "Acos": unary(mb.acos),
    "Atan": unary(mb.atan),
    "Sinh": unary(mb.sinh),
    "Cosh": unary(mb.cosh),
    "Atanh": unary(mb.atanh),
    # Variadic min/max.
    "Min": _variadic(mb.minimum),
    "Max": _variadic(mb.maximum),
    "Clip": _clip,
    "Mod": binary(mb.mod),
    # Comparison (bool output).
    "Equal": binary(mb.equal),
    "Greater": binary(mb.greater),
    "Less": binary(mb.less),
    "GreaterOrEqual": binary(mb.greater_equal),
    "LessOrEqual": binary(mb.less_equal),
    # Logical (bool in/out).
    "And": binary(mb.logical_and),
    "Or": binary(mb.logical_or),
    "Xor": binary(mb.logical_xor),
    "Not": unary(mb.logical_not),
    "IsNaN": _isnan,
    # Selection.
    "Where": _where,
}

__all__ = ["REGISTRY"]
