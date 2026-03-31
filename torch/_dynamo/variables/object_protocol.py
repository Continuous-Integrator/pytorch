"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
comparison dispatch machinery that is independent of any specific type.
Per-type richcompare_impl hooks live in their respective VT files.
"""

from typing import TYPE_CHECKING

from .. import polyfills
from ..exc import raise_observed_exception
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, raise_type_error_exc, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE
from .functions import UserFunctionVariable


type_error = raise_type_error_exc


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


def vt_implements_slot(
    obj: "VariableTracker",
    dunder: str,
    impl_method: str,
) -> bool:
    """
    Check whether obj implements a CPython slot, identified by both its Python
    dunder name and the corresponding VT impl method name.

    - UserDefinedObjectVariable: check whether the underlying class defines dunder.
    - ConstantVariable: check hasattr on the wrapped value.
    - All other VTs: check whether the subclass overrides impl_method.
    """
    from .base import VariableTracker
    from .constant import ConstantVariable
    from .user_defined import UserDefinedObjectVariable

    if istype(obj, UserDefinedObjectVariable):
        return obj._maybe_get_baseclass_method(dunder) is not None
    elif istype(obj, ConstantVariable):
        return hasattr(obj.value, dunder)
    else:
        m1 = getattr(obj.__class__, impl_method)
        m2 = getattr(VariableTracker, impl_method)
        return m1 is not m2


def vt_implements_sq_length(obj: "VariableTracker") -> bool:
    return vt_implements_slot(obj, "__len__", "sq_length")


def vt_implements_mp_length(obj: "VariableTracker") -> bool:
    return vt_implements_slot(obj, "__len__", "mp_length")


def vt_mapping_size(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    if vt_implements_mp_length(obj):
        return obj.mp_length(tx)

    if vt_implements_sq_length(obj):
        type_error(tx, f"{obj.python_type_name()} is not a mapping")

    type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def vt_implements_tp_iter(obj: "VariableTracker") -> bool:
    return vt_implements_slot(obj, "__iter__", "iter_impl")


def vt_sequence_check(obj: "VariableTracker") -> bool:
    """Implements PySequence_Check semantics for VariableTracker objects."""
    from .dicts import ConstDictVariable

    if istype(obj, ConstDictVariable):
        return False

    # needs generic_getitem to be implemented in Dynamo
    return True
    # return vt_implements_method(obj, " getitem_impl")


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L53-L69
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Dispatches to sq_length (sequences) or mp_length (mappings) depending on the VT type.
    """

    if vt_implements_sq_length(obj):
        return obj.sq_length(tx)
    return vt_mapping_size(tx, obj)


def generic_getitem(
    tx: "InstructionTranslator", obj: "VariableTracker", item: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_GetItem semantics for VariableTracker objects.
    Routes to obj.getitem_impl(tx, item)
    """
    return obj.getitem_impl(tx, item)


def vt_implements_tp_iternext(obj: "VariableTracker") -> bool:
    """Helper function to check if a VariableTracker implements the tp_iternext slot."""
    from .constant import ConstantVariable
    from .user_defined import UserDefinedObjectVariable

    if istype(obj, UserDefinedObjectVariable):
        return obj._maybe_get_baseclass_method("__next__") is not None
    elif istype(obj, ConstantVariable):
        return hasattr(obj.value, "__next__")
    else:
        return vt_implements_method(obj, "iternext_impl")


def generic_iternext(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyIter_Next / tp_iternext semantics for VariableTracker objects.

    Calls obj.iternext_impl(tx) if the object is an iterator, otherwise raises
    TypeError. StopIteration propagation is left to the caller (mirrors
    CPython's iternext contract where NULL return signals exhaustion).
    """
    from .base import VariableTracker

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2865
    if not vt_implements_tp_iternext(obj):
        msg = VariableTracker.build(
            tx, f"'{obj.python_type_name()}' object is not an iterator"
        )
        raise_observed_exception(TypeError, tx, args=[msg])

    return obj.iternext_impl(tx)


# TODO(guilhermeleobas): should we narrow the return type to IteratorVariable?
def generic_getiter(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_GetIter semantics for VariableTracker objects.
    Routes to obj.iter_impl(tx), the tp_iter slot on the object's type.
    """
    from .base import VariableTracker

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#2848
    # The algorithm for PyObject_GetIter is as follows: Steps:
    # 1. If the object has tp_iter slot, call it and return the result The
    #    return object must be an iterator (it must have a tp_iternext slot)
    # 2. If the object implements the sequence protocol - implements __getitem__
    #    and __len__, then create a sequence iterator for the object and return
    #    it.
    # 3. Otherwise, raise a TypeError

    if vt_implements_tp_iter(obj):
        res = obj.iter_impl(tx)
        if not vt_implements_tp_iternext(res):
            msg = VariableTracker.build(
                tx,
                f"{obj.python_type_name()}.__iter__() must return an iterator, not {res.python_type_name()}",
            )
            raise_observed_exception(TypeError, tx, args=[msg])
        return res
    elif vt_sequence_check(obj):
        return UserFunctionVariable(polyfills.builtins.sequence_iterator).call_function(
            tx, [obj], {}
        )
    else:
        msg = VariableTracker.build(
            tx, f"'{obj.python_type_name()}' object is not iterable"
        )
        raise_observed_exception(
            TypeError,
            tx,
            args=[msg],
        )
