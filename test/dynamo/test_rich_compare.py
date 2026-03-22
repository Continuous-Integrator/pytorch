# Owner(s): ["module: dynamo"]
"""Tests for generic_richcompare: unified comparison protocol in Dynamo."""

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
            return list[int] == list[int]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_typing_eq_unequal(self):
        def fn(x):
            return list[int] == list[str]

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_typing_ne(self):
        def fn(x):
            return list[int] != list[str]

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- ConstantVariable ---

    def test_constant_eq(self):
        def fn(x):
            return 1 == 1  # noqa: PLR0133

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_ne(self):
        def fn(x):
            return 1 != 2  # noqa: PLR0133

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_lt(self):
        def fn(x):
            return 1 < 2  # noqa: PLR0133

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_constant_eq_cross_type_int_str(self):
        def fn(x):
            return 1 == "1"  # noqa: PLR0133

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

    def test_dict_items_ne_equal(self):
        def fn(x):
            d1 = {"a": 1}
            d2 = {"b": 2}
            return d1.items() != d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_ne_same(self):
        def fn(x):
            d = {"a": 1}
            return d.items() != d.items()

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_items_lt_proper_subset(self):
        def fn(x):
            d1 = {"a": 1}
            d2 = {"a": 1, "b": 2}
            return d1.items() < d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_lt_same(self):
        def fn(x):
            d = {"a": 1}
            return d.items() < d.items()

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_items_le_subset(self):
        def fn(x):
            d1 = {"a": 1}
            d2 = {"a": 1, "b": 2}
            return d1.items() <= d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_le_equal(self):
        def fn(x):
            d = {"a": 1}
            return d.items() <= d.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_gt_proper_superset(self):
        def fn(x):
            d1 = {"a": 1, "b": 2}
            d2 = {"a": 1}
            return d1.items() > d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_ge_equal(self):
        def fn(x):
            d = {"a": 1}
            return d.items() >= d.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_eq_set_matching_tuples(self):
        # d.items() == {("a", 1)} — items are (k,v) tuples, set contains matching tuples
        def fn(x):
            d = {"a": 1}
            s = {("a", 1)}
            return d.items() == s

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_items_eq_set_non_matching(self):
        def fn(x):
            d = {"a": 1}
            s = {"a"}
            return d.items() == s

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- DictKeysVariable vs DictItemsVariable ---

    def test_dict_keys_eq_items_non_matching(self):
        # keys are single values, items are (k,v) tuples — almost always False
        def fn(x):
            d = {"a": 1}
            return d.keys() == d.items()

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_dict_keys_eq_items_matching_tuple_keys(self):
        # dict whose keys are themselves (k,v) tuples matching the items of another dict
        def fn(x):
            d1 = {("a", 1): None}  # key is the tuple ("a", 1)
            d2 = {"a": 1}  # items are {("a", 1)}
            # d1.keys() = {("a", 1)}, d2.items() = {("a", 1)} → True
            return d1.keys() == d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_dict_keys_lt_items(self):
        # d1.keys() is a proper subset of d2.items() when keys are (k,v) tuples
        def fn(x):
            d1 = {("a", 1): None}
            d2 = {"a": 1, "b": 2}
            # d1.keys() = {("a",1)}, d2.items() = {("a",1), ("b",2)} → True (proper subset)
            return d1.keys() < d2.items()

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- SliceVariable ---

    def test_slice_eq_equal(self):
        def fn(x):
            return slice(1, 5, 2) == slice(1, 5, 2)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_slice_eq_unequal(self):
        def fn(x):
            return slice(1, 5) == slice(1, 6)

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_slice_ne(self):
        def fn(x):
            return slice(1, 5) != slice(1, 6)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_slice_lt(self):
        def fn(x):
            return slice(1, 3) < slice(1, 5)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_slice_eq_non_slice_returns_false(self):
        def fn(x):
            return slice(1, 5) == (1, 5, None)

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- UserDefinedClassVariable ---

    def test_user_class_eq_same(self):
        class MyClass:
            pass

        def fn(x):
            return MyClass == MyClass

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_user_class_eq_different(self):
        class A:
            pass

        class B:
            pass

        def fn(x):
            return A == B

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- UserDefinedObjectVariable ---

    def test_user_object_eq_same_identity(self):
        class MyObj:
            pass

        obj = MyObj()

        def fn(x):
            return obj == obj

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_user_object_eq_custom_eq(self):
        class MyObj:
            def __init__(self, v):
                self.v = v

            def __eq__(self, other):
                return isinstance(other, MyObj) and self.v == other.v

        a = MyObj(1)
        b = MyObj(1)

        def fn(x):
            return a == b

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- NNModuleVariable ---

    def test_nn_module_eq_same(self):
        import torch.nn as nn

        m = nn.Linear(2, 2)

        def fn(x):
            return m == m

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_nn_module_eq_different(self):
        import torch.nn as nn

        m1 = nn.Linear(2, 2)
        m2 = nn.Linear(2, 2)

        def fn(x):
            return m1 == m2

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- BaseUserFunctionVariable ---

    def test_function_eq_same(self):
        def foo():
            pass

        def fn(x):
            return foo == foo

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_function_eq_different(self):
        def foo():
            pass

        def bar():
            pass

        def fn(x):
            return foo == bar

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- MappingProxyVariable ---

    def test_mappingproxy_eq_equal(self):
        import types

        def fn(x):
            p1 = types.MappingProxyType({"a": 1})
            p2 = types.MappingProxyType({"a": 1})
            return p1 == p2

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_mappingproxy_eq_unequal(self):
        import types

        def fn(x):
            p1 = types.MappingProxyType({"a": 1})
            p2 = types.MappingProxyType({"b": 2})
            return p1 == p2

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    def test_mappingproxy_ne(self):
        import types

        def fn(x):
            p1 = types.MappingProxyType({"a": 1})
            p2 = types.MappingProxyType({"b": 2})
            return p1 != p2

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- ExceptionVariable ---

    def test_exception_eq_same(self):
        exc = ValueError("hello")

        def fn(x):
            return exc == exc

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_exception_eq_different(self):
        exc1 = ValueError("a")
        exc2 = ValueError("b")

        def fn(x):
            return exc1 == exc2

        self.assertFalse(self._compile(fn, torch.tensor(0)))

    # --- FakeIdVariable ---

    def test_fake_id_eq(self):
        def fn(x):
            d = {}
            return id(d) == id(d)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    def test_fake_id_ne(self):
        def fn(x):
            d1 = {}
            d2 = {}
            return id(d1) != id(d2)

        self.assertTrue(self._compile(fn, torch.tensor(0)))

    # --- Exception propagation ---

    def test_ordering_unsupported_dict_lt_raises_type_error(self):
        def fn(x):
            try:
                return {"a": 1} < {"b": 2}
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)
        self.assertIn("<", result)

    def test_ordering_unsupported_range_lt_raises_type_error(self):
        def fn(x):
            try:
                return range(3) < range(4)
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)

    def test_ordering_unsupported_set_lt_int_raises_type_error(self):
        def fn(x):
            s = {1, 2}
            try:
                return s < 5
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)

    def test_cross_type_ordering_int_lt_str_raises_type_error(self):
        def fn(x):
            try:
                return 1 < "abc"  # noqa: PLR0133
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)

    def test_custom_eq_that_raises_propagates(self):
        class BadEq:
            def __eq__(self, other):
                raise ValueError("cannot compare")

        obj = BadEq()

        def fn(x):
            try:
                return obj == obj
            except ValueError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("cannot compare", result)

    def test_custom_lt_that_raises_propagates(self):
        class BadLt:
            def __lt__(self, other):
                raise NotImplementedError("no ordering for BadLt")

        a = BadLt()
        b = BadLt()

        def fn(x):
            try:
                return a < b
            except NotImplementedError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("no ordering for BadLt", result)

    def test_list_lt_dict_raises_type_error(self):
        def fn(x):
            try:
                return [1, 2] < {"a": 1}
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)

    def test_exception_lt_exception_raises_type_error(self):
        e1 = ValueError("a")
        e2 = ValueError("b")

        def fn(x):
            try:
                return e1 < e2
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.tensor(0))
        self.assertIn("not supported", result)


if __name__ == "__main__":
    run_tests()
