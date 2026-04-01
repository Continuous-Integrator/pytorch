"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c and abstract.c, this module holds
the general comparison and binary-op dispatch machinery that is independent
of any specific type. Per-type slot hooks live in their respective VT files.
"""

from ..exc import raise_observed_exception
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


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
# ---------------------------------------------------------------------------


def is_nb_not_implemented(result: VariableTracker) -> bool:
    return result.is_constant_match(NotImplemented)


def binary_op1(
    tx: "InstructionTranslator",  # noqa: F821
    v: VariableTracker,
    w: VariableTracker,
    slot_name: str,
) -> VariableTracker:
    """CPython's binary_op1: try v's slot, then w's slot (with subclass priority).

    Each VT that participates provides a ``<slot_name>_impl(self, tx, other)``
    method where *self* is the slot owner and *other* is the second operand.
    The generic algorithm calls ``v.<slot>_impl(tx, w)`` for the left slot
    and ``w.<slot>_impl(tx, v)`` for the right slot (reversed args, matching
    CPython's ``slotw(v, w)`` / ``slotv(v, w)`` pattern — for the types we
    support the slots check both operands symmetrically so the result is the
    same).
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
        result = v_slot(v, tx, w)
        if not is_nb_not_implemented(result):
            return result
    if w_slot is not None:
        result = w_slot(w, tx, v)
        if not is_nb_not_implemented(result):
            return result
    from .constant import ConstantVariable

    return ConstantVariable.create(NotImplemented)


def binary_op(
    tx: "InstructionTranslator",  # noqa: F821
    v: VariableTracker,
    w: VariableTracker,
    slot_name: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_op: binary_op1 + TypeError fallback."""
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
    tx: "InstructionTranslator",  # noqa: F821
    v: VariableTracker,
    w: VariableTracker,
    islot_name: str,
    slot_name: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_iop: try inplace slot, fallback to binary_op1."""
    islot_impl = getattr(type(v), f"{islot_name}_impl", None)
    if islot_impl is not None:
        result = islot_impl(v, tx, w)
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
