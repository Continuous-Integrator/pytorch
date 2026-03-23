"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
comparison dispatch machinery that is independent of any specific type.
Per-type richcompare_impl hooks live in their respective VT files.
"""

from typing import TYPE_CHECKING

from ..exc import raise_observed_exception, unimplemented
from ..utils import istype, richcmp_op_str
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator


reflected_richcompare_op: dict[str, str] = {
    "__eq__": "__eq__",
    "__ne__": "__ne__",
    "__lt__": "__gt__",
    "__le__": "__ge__",
    "__gt__": "__lt__",
    "__ge__": "__le__",
}


def is_richcompare_not_implemented(vt: "VariableTracker") -> bool:
    from .constant import ConstantVariable

    return isinstance(vt, ConstantVariable) and vt.value is NotImplemented


def type_overrides_richcompare(tp: type, op: str) -> bool:
    method = getattr(tp, op, None)
    obj_method = getattr(object, op, None)
    return method is not None and method is not obj_method


def python_constant_richcompare_impl(
    self: "VariableTracker",
    tx: "InstructionTranslator",
    other: "VariableTracker",
    op: str,
) -> "VariableTracker":
    """richcompare_impl for VTs whose identity is their as_python_constant() value.

    Suitable for function VTs (UserFunctionVariable, TorchInGraphFunctionVariable,
    BuiltinVariable, etc.) where equality is Python object identity.
    Returns ConstantVariable(NotImplemented) if either side can't be reduced
    to a Python constant, letting generic_richcompare try the reflected op or
    fall back to identity comparison.
    """
    from .constant import ConstantVariable

    if not other.is_python_constant():
        return ConstantVariable.create(NotImplemented)
    try:
        self_val = self.as_python_constant()
    except NotImplementedError:
        return ConstantVariable.create(NotImplemented)
    # Call the dunder directly (not operator.* which dispatches both sides)
    # so that unsupported comparisons return NotImplemented instead of raising
    # TypeError, matching CPython's tp_richcompare slot semantics.
    result = getattr(type(self_val), op)(self_val, other.as_python_constant())
    return ConstantVariable.create(result)


def object_richcompare(
    self: "VariableTracker",
    tx: "InstructionTranslator",
    other: "VariableTracker",
    op: str,
) -> "VariableTracker":
    """richcompare_impl for VTs whose CPython type's tp_richcompare is object_richcompare.

    Mirrors PyBaseObject_Type.tp_richcompare (Objects/typeobject.c): identity
    for __eq__/__ne__, NotImplemented for ordering ops. Used for types where
    tp_richcompare is NULL/0 (inheriting object_richcompare), e.g., PyType_Type,
    PyModule_Type. Delegates to python_constant_richcompare_impl, which uses
    the backing Python object's __eq__/__ne__ (identity-based for these types)
    and returns NotImplemented for ordering, matching object_richcompare behavior.
    """
    return python_constant_richcompare_impl(self, tx, other, op)


def vt_identity_compare(
    left: VariableTracker,
    right: VariableTracker,
) -> "VariableTracker | None":
    """Try to determine Python identity (left is right) at trace time.

    Returns ConstantVariable(True/False) if determinable, else None.
    Mirrors the logic in BuiltinVariable's handle_is handler.
    """
    if left is right:
        return CONSTANT_VARIABLE_TRUE

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            CONSTANT_VARIABLE_TRUE if left_val is right_val else CONSTANT_VARIABLE_FALSE
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return CONSTANT_VARIABLE_FALSE

    # Mutable containers created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable

    if isinstance(left, (ConstDictVariable, ListVariable)):
        return CONSTANT_VARIABLE_FALSE

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return CONSTANT_VARIABLE_FALSE
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    from .. import variables

    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return CONSTANT_VARIABLE_FALSE

    return None


def generic_richcompare(
    tx: "InstructionTranslator",
    lhs: "VariableTracker",
    rhs: "VariableTracker",
    op: str,
) -> "VariableTracker":
    """Implement CPython's PyObject_RichCompare algorithm.

    https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L972

    Steps:
      1. If type(rhs) is a proper subclass of type(lhs) with overriding reflected op: try rhs first
      2. Try lhs.__op__(rhs)
      3. If NotImplemented, try rhs.__reflected_op__(lhs)
      4. If still NotImplemented: for __eq__/__ne__ use identity; for others TypeError
    """
    from .. import graph_break_hints

    reflected = reflected_richcompare_op[op]

    # Step 1: subclass priority
    try:
        lhs_type = lhs.python_type()
        rhs_type = rhs.python_type()
        rhs_first = (
            lhs_type is not rhs_type
            and issubclass(rhs_type, lhs_type)
            and type_overrides_richcompare(rhs_type, reflected)
        )
    except NotImplementedError:
        rhs_first = False

    if rhs_first:
        result = rhs.richcompare_impl(tx, lhs, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 2: forward
    result = lhs.richcompare_impl(tx, rhs, op)
    if not is_richcompare_not_implemented(result):
        return result

    # Step 3: reflected (if not already tried in step 1)
    if not rhs_first:
        result = rhs.richcompare_impl(tx, lhs, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 4: fallback
    if op in ("__eq__", "__ne__"):
        # CPython: fall back to identity (a is b)
        identity = vt_identity_compare(lhs, rhs)
        if identity is None:
            unimplemented(
                gb_type="Cannot determine object identity at trace time",
                context=f"comparing {type(lhs).__name__} and {type(rhs).__name__}",
                explanation="Dynamo cannot resolve identity of these objects at trace time.",
                hints=[*graph_break_hints.FUNDAMENTAL],
            )
        is_same = identity.as_python_constant()
        from .constant import ConstantVariable

        return ConstantVariable.create(is_same if op == "__eq__" else not is_same)
    else:
        lhs_name = lhs.python_type_name()
        rhs_name = rhs.python_type_name()
        op_str = richcmp_op_str[op]
        msg = VariableTracker.build(
            tx,
            f"'{op_str}' not supported between instances of '{lhs_name}' and '{rhs_name}'",
        )
        raise_observed_exception(TypeError, tx, args=[msg])
