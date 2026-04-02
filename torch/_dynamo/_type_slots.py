from enum import IntEnum


class PySequenceSlots(IntEnum):
    """PySequenceMethods slot bit positions (bits 0-7 in get_pysequence_slots)"""

    SQ_LENGTH = 0
    SQ_CONCAT = 1
    SQ_REPEAT = 2
    SQ_ITEM = 3
    SQ_CONTAINS = 4
    SQ_ASS_ITEM = 5
    SQ_INPLACE_CONCAT = 6
    SQ_INPLACE_REPEAT = 7


class PyMappingSlots(IntEnum):
    """PyMappingMethods slot bit positions (bits 0-2 in get_pymapping_slots)"""

    MP_LENGTH = 0
    MP_SUBSCRIPT = 1
    MP_ASS_SUBSCRIPT = 2


class PyNumberSlots(IntEnum):
    """PyNumberMethods slot bit positions (bits 0-33 in get_pynumber_slots)"""

    NB_ADD = 0
    NB_SUBTRACT = 1
    NB_MULTIPLY = 2
    NB_REMAINDER = 3
    NB_POWER = 4
    NB_NEGATIVE = 5
    NB_POSITIVE = 6
    NB_ABSOLUTE = 7
    NB_BOOL = 8
    NB_INVERT = 9
    NB_LSHIFT = 10
    NB_RSHIFT = 11
    NB_AND = 12
    NB_XOR = 13
    NB_OR = 14
    NB_INT = 15
    NB_FLOAT = 16
    NB_INPLACE_ADD = 17
    NB_INPLACE_SUBTRACT = 18
    NB_INPLACE_MULTIPLY = 19
    NB_INPLACE_REMAINDER = 20
    NB_INPLACE_POWER = 21
    NB_INPLACE_LSHIFT = 22
    NB_INPLACE_RSHIFT = 23
    NB_INPLACE_AND = 24
    NB_INPLACE_XOR = 25
    NB_INPLACE_OR = 26
    NB_FLOOR_DIVIDE = 27
    NB_TRUE_DIVIDE = 28
    NB_INPLACE_FLOOR_DIVIDE = 29
    NB_INPLACE_TRUE_DIVIDE = 30
    NB_INDEX = 31
    NB_MATRIX_MULTIPLY = 32
    NB_INPLACE_MATRIX_MULTIPLY = 33


class PyTypeSlots(IntEnum):
    """PyTypeObject slot bit positions (bits 0-9 in get_pytype_slots)"""

    TP_HASH = 0
    TP_ITER = 1
    TP_ITERNEXT = 2
    TP_CALL = 3
    TP_REPR = 4
    TP_RICHCOMPARE = 5
    TP_GETATTRO = 6
    TP_SETATTRO = 7
    TP_DESCR_GET = 8
    TP_DESCR_SET = 9


def has_slot(slots: int, slot_bit: int) -> bool:
    """Check if a slot is present in the bitmask.

    Args:
        slots: The int64 bitmask returned by get_type_slots()
        slot_bit: The bit position (e.g., PySequenceSlots.SQ_LENGTH)

    Returns:
        True if the slot is present (bit is set)
    """
    return (slots & (1 << slot_bit)) != 0


def dbg_slot(slot_bitmask: int, slot_enum: type) -> None:
    """Get the names of all slots set in a bitmask for a given slot enum."""
    names = []
    for slot in slot_enum:
        if has_slot(slot_bitmask, slot):
            names.append(slot.name)
    print(names)
