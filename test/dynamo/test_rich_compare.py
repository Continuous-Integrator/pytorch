"""Tests for generic_richcompare: unified comparison protocol in Dynamo."""
import typing

import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, TestCase


class RichCompareTests(TestCase):
    def _compile(self, fn, *args, **kwargs):
        return torch.compile(fn, backend="eager", fullgraph=True)(*args, **kwargs)

    # --- SetVariable ---

    def test_set_eq_non_set_returns_false(self):
        def fn(x):
            s = {1, 2, 3}
            return s == "not a set"

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_set_ne_non_set_returns_true(self):
        def fn(x):
            s = {1, 2}
            return s != "foo"

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_set_eq_equal_sets(self):
        def fn(x):
            a = {1, 2}
            b = {1, 2}
            return a == b

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_set_eq_unequal_sets(self):
        def fn(x):
            a = {1, 2}
            b = {1, 3}
            return a == b

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- ConstDictVariable ---

    def test_dict_eq_equal(self):
        def fn(x):
            return {"a": 1} == {"a": 1}

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_eq_unequal(self):
        def fn(x):
            return {"a": 1} == {"b": 2}

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_ne_equal_dicts(self):
        def fn(x):
            return {"a": 1} != {"a": 1}

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_ne_unequal_dicts(self):
        def fn(x):
            return {"a": 1} != {"b": 2}

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_eq_non_dict_returns_false(self):
        # dict == non-dict: CPython returns False (via NotImplemented → identity fallback)
        def fn(x):
            return {"a": 1} == [1, 2]

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- BaseListVariable ---

    def test_list_eq_equal(self):
        def fn(x):
            return [1, 2, 3] == [1, 2, 3]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_list_eq_unequal(self):
        def fn(x):
            return [1, 2] == [1, 3]

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_list_lt(self):
        def fn(x):
            return [1, 2] < [1, 3]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_tuple_eq_equal(self):
        def fn(x):
            return (1, 2) == (1, 2)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_tuple_eq_unequal(self):
        def fn(x):
            return (1, 2) == (1, 3)

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_list_eq_non_list_returns_false(self):
        def fn(x):
            return [1, 2] == "foo"

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- RangeVariable ---

    def test_range_eq_equal(self):
        def fn(x):
            return range(3) == range(3)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_range_eq_unequal(self):
        def fn(x):
            return range(3) == range(4)

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_range_ne_equal(self):
        def fn(x):
            return range(3) != range(3)

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_range_eq_non_range_returns_false(self):
        def fn(x):
            return range(3) == [0, 1, 2]

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- TypingVariable ---

    def test_typing_eq_equal(self):
        def fn(x):
            return typing.List[int] == typing.List[int]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_typing_eq_unequal(self):
        def fn(x):
            return typing.List[int] == typing.List[str]

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_typing_ne(self):
        def fn(x):
            return typing.List[int] != typing.List[str]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- ConstantVariable ---

    def test_constant_eq(self):
        def fn(x):
            return 1 == 1

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_ne(self):
        def fn(x):
            return 1 != 2

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_lt(self):
        def fn(x):
            return 1 < 2

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_eq_cross_type_int_str(self):
        def fn(x):
            return 1 == "1"

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- Subclass priority (step 1) ---

    def test_subclass_priority_eq_native_types(self):
        """rhs side is tried first for subclass, but result is identity for unknown types."""
        # A list of tuples is not equal to a tuple of the same items in CPython
        # because list.__eq__ returns NotImplemented for non-list, and
        # tuple.__eq__ returns NotImplemented for non-tuple → identity → False
        def fn(x):
            a = [1, 2]
            b = (1, 2)
            return a == b

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- DictItemsVariable ---

    def test_dict_items_eq_equal(self):
        def fn(x):
            d1 = {"a": 1, "b": 2}
            d2 = {"a": 1, "b": 2}
            return d1.items() == d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_eq_unequal(self):
        def fn(x):
            d1 = {"a": 1}
            d2 = {"a": 2}
            return d1.items() == d2.items()

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_items_eq_non_items_returns_false(self):
        def fn(x):
            d = {"a": 1}
            return d.items() == [("a", 1)]

        self.assertFalse(self._compile(fn, torch.tensor(0)))


if __name__ == "__main__":
    run_tests()
