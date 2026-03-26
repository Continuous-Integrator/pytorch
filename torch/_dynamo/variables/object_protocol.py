"""
Implements CPython's type protocol for VariableTracker objects.

Following the TypeVariableTracker design doc, we mirror CPython's
PyTypeObject slots (e.g., tp_iter, tp_richcompare) by providing
generic dispatcher functions that route to per-type hook methods.

For example:
  - generic_getiter(tx, obj) corresponds to PyObject_GetIter(obj)
  - obj.iter_impl(tx) corresponds to type(obj)->tp_iter(obj)
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator
    from .base import VariableTracker


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    return obj.len_impl(tx)
