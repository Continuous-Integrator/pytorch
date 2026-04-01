"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c and abstract.c, this module holds
the general comparison and binary-op dispatch machinery that is independent
of any specific type. Per-type slot hooks live in their respective VT files.
"""

from typing import TYPE_CHECKING

from ..exc import raise_observed_exception
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator


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


# ---------------------------------------------------------------------------
# Binary-op dispatch (CPython's abstract.c: binary_op1 / binary_op)
# https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L927 (binary_op1)
# ---------------------------------------------------------------------------


def is_nb_not_implemented(result: VariableTracker) -> bool:
    return result.is_constant_match(NotImplemented)


def _is_python_subtype(w: VariableTracker, v: VariableTracker) -> bool:
    """Check if w's underlying Python type is a proper subtype of v's."""
    try:
        return issubclass(w.python_type(), v.python_type())
    except NotImplementedError:
        return False


def binary_op1(
    tx: "InstructionTranslator",
    v: VariableTracker,
    w: VariableTracker,
    slot_name: str,
) -> VariableTracker:
    """CPython's binary_op1: try v's slot, then w's slot with subclass priority.

    Each VT that participates provides a ``<slot_name>_impl(self, tx, other,
    reverse)`` method. ``reverse=False`` means "self is left operand" (forward,
    e.g. ``__or__``), ``reverse=True`` means "self is right operand" (reverse,
    e.g. ``__ror__``). For built-in types the flag is ignored because their
    slots check both operands symmetrically.

    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L926-L977
    """
    impl_attr = f"{slot_name}_impl"
    v_slot = getattr(type(v), impl_attr, None)
    w_slot = getattr(type(w), impl_attr, None)

    # Same class → only call once (CPython: slotw = NULL if same type)
    if type(v) is type(w):
        w_slot = None
    # Same implementation (inherited) → skip w
    elif v_slot is w_slot:
        w_slot = None

    if v_slot is not None:
        # Subclass priority: if w's Python type is a proper subtype of v's
        # Python type and overrides the slot, try w first (CPython abstract.c:952-960).
        if w_slot is not None and _is_python_subtype(w, v):
            result = w_slot(w, tx, v, True)
            if not is_nb_not_implemented(result):
                return result
            w_slot = None
        result = v_slot(v, tx, w, False)
        if not is_nb_not_implemented(result):
            return result
    if w_slot is not None:
        result = w_slot(w, tx, v, True)
        if not is_nb_not_implemented(result):
            return result
    from .constant import ConstantVariable

    return ConstantVariable.create(NotImplemented)


def binary_op(
    tx: "InstructionTranslator",
    v: VariableTracker,
    w: VariableTracker,
    slot_name: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_op: binary_op1 + TypeError fallback.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L997-L1020
    """

    result = binary_op1(tx, v, w, slot_name)
    if is_nb_not_implemented(result):
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                f"unsupported operand type(s) for {op_symbol}: "
                f"'{v.python_type_name()}' and '{w.python_type_name()}'"
            ],
        )
    return result


def binary_iop(
    tx: "InstructionTranslator",
    v: VariableTracker,
    w: VariableTracker,
    islot_name: str,
    slot_name: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_iop: try inplace slot, fallback to binary_op1.

    Combines binary_iop1 + TypeError fallback from binary_iop.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1229-L1270 (binary_iop1, binary_iop)
    """
    islot_impl = getattr(type(v), f"{islot_name}_impl", None)
    if islot_impl is not None:
        result = islot_impl(v, tx, w, False)
        if not is_nb_not_implemented(result):
            return result
    result = binary_op1(tx, v, w, slot_name)
    if is_nb_not_implemented(result):
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                f"unsupported operand type(s) for {op_symbol}: "
                f"'{v.python_type_name()}' and '{w.python_type_name()}'"
            ],
        )
    return result
