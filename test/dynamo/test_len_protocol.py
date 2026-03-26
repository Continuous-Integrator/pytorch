# Owner(s): ["module: dynamo"]

"""
Comprehensive tests for len() builtin and __len__() method protocol in PyTorch Dynamo.

Tests cover:
- len(obj) builtin calls
- obj.__len__() method calls
- type(obj).__len__(obj) unbound method calls (marked as expectedFailure - Dynamo limitation)
- Various container types: list, tuple, dict, set, frozenset, range, str, Tensor, nn.Module
- Dict views: keys(), values(), items()
- User-defined classes with __len__
"""

import collections

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class _BaseSequenceLen:
    """Base class for testing len() on sequence types (list, tuple)"""

    thetype = None  # Override in subclass

    def setUp(self):
        if self.thetype is None:
            self.skipTest("Base class - not meant to be run directly")
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        seq = self.thetype([1, 2, 3])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        seq = self.thetype([])
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        seq = self.thetype([42])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq.__len__(), 1)

    @make_dynamo_test
    def test_len_nested(self):
        inner = self.thetype([1, 2])
        seq = self.thetype([inner, inner, inner])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_tensors(self):
        seq = self.thetype([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_mixed_types(self):
        seq = self.thetype([1, "hello", 3.14, torch.tensor(4)])
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.__len__(), 4)

    @make_dynamo_test
    def test_len_large(self):
        seq = self.thetype(range(100))
        self.assertEqual(len(seq), 100)
        self.assertEqual(seq.__len__(), 100)


class TestListLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on list objects"""

    thetype = list


class TestTupleLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on tuple objects"""

    thetype = tuple


class _BaseMappingLen:
    """Base class for testing len() on mapping types (dict, OrderedDict)"""

    def get_mapping(self, items):
        """Override in subclass to return appropriate mapping type"""
        return dict(items)

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        d = self.get_mapping({1: "a", 2: "b", 3: "c"}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        d = self.get_mapping({}.items())
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_entry(self):
        d = self.get_mapping({"key": "value"}.items())
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_string_keys(self):
        d = self.get_mapping({"one": 1, "two": 2, "three": 3}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_int_keys(self):
        d = self.get_mapping({1: "one", 2: "two", 3: "three", 4: "four"}.items())
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_with_tensor_values(self):
        d = self.get_mapping(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}.items()
        )
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        d = self.get_mapping({i: i * 2 for i in range(20)}.items())
        self.assertEqual(len(d), 20)
        self.assertEqual(d.__len__(), 20)


class TestDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on dict objects"""


class TestOrderedDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on OrderedDict objects"""

    def get_mapping(self, items):
        return collections.OrderedDict(items)


class _BaseSetLen:
    """Base class for testing len() on set types"""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def get_set(self, items):
        """Override in subclass to return appropriate set type"""
        return set(items)

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        s = self.get_set([1, 2, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        s = self.get_set([])
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        s = self.get_set([42])
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)

    @make_dynamo_test
    def test_len_with_strings(self):
        s = self.get_set(["a", "b", "c", "d"])
        self.assertEqual(len(s), 4)
        self.assertEqual(s.__len__(), 4)

    @make_dynamo_test
    def test_len_with_duplicates(self):
        # Set constructor deduplicates
        s = self.get_set([1, 2, 2, 3, 3, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        s = self.get_set(range(50))
        self.assertEqual(len(s), 50)
        self.assertEqual(s.__len__(), 50)


class TestSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on set objects"""

    def get_set(self, items):
        return set(items)


class TestFrozenSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on frozenset objects"""

    def get_set(self, items):
        return frozenset(items)


class TestRangeLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on range objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        r = range(5)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_with_start_stop(self):
        r = range(5, 15)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_with_step(self):
        r = range(0, 10, 2)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_negative_step(self):
        r = range(10, 0, -1)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_empty(self):
        r = range(5, 5)
        self.assertEqual(len(r), 0)
        self.assertEqual(r.__len__(), 0)
        self.assertEqual(range.__len__(r), 0)


class TestStringLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on string objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_empty(self):
        s = ""
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)
        self.assertEqual(str.__len__(s), 0)

    @make_dynamo_test
    def test_len_single_char(self):
        s = "a"
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)
        self.assertEqual(str.__len__(s), 1)

    @make_dynamo_test
    def test_len_multiple_chars(self):
        s = "hello"
        self.assertEqual(len(s), 5)
        self.assertEqual(s.__len__(), 5)
        self.assertEqual(str.__len__(s), 5)

    @make_dynamo_test
    def test_len_with_spaces(self):
        s = "hello world"
        self.assertEqual(len(s), 11)
        self.assertEqual(s.__len__(), 11)
        self.assertEqual(str.__len__(s), 11)


class TestTensorLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on torch.Tensor objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_1d(self):
        t = torch.tensor([1, 2, 3, 4, 5])
        self.assertEqual(len(t), 5)
        self.assertEqual(t.__len__(), 5)
        self.assertEqual(torch.Tensor.__len__(t), 5)

    @make_dynamo_test
    def test_len_2d(self):
        t = torch.tensor([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)

    @make_dynamo_test
    def test_len_3d(self):
        t = torch.randn(4, 5, 6)
        self.assertEqual(len(t), 4)
        self.assertEqual(t.__len__(), 4)
        self.assertEqual(torch.Tensor.__len__(t), 4)

    @make_dynamo_test
    def test_len_empty(self):
        t = torch.tensor([])
        self.assertEqual(len(t), 0)
        self.assertEqual(t.__len__(), 0)
        self.assertEqual(torch.Tensor.__len__(t), 0)

    @make_dynamo_test
    def test_len_large_batch(self):
        t = torch.randn(100, 5, 5)
        self.assertEqual(len(t), 100)
        self.assertEqual(t.__len__(), 100)
        self.assertEqual(torch.Tensor.__len__(t), 100)

    @make_dynamo_test
    def test_len_different_dtypes(self):
        t = torch.tensor([1, 2, 3], dtype=torch.float32)
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)


class TestNNModuleLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on torch.nn module containers"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

        # Pre-construct nn.Module instances outside compiled regions
        self.seq_3layers = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        self.empty_seq = torch.nn.Sequential()
        self.ml_2modules = torch.nn.ModuleList(
            [torch.nn.Linear(10, 20), torch.nn.Linear(20, 30)]
        )
        self.empty_ml = torch.nn.ModuleList()
        self.md_2modules = torch.nn.ModuleDict(
            {"layer1": torch.nn.Linear(10, 20), "layer2": torch.nn.Linear(20, 5)}
        )
        self.empty_md = torch.nn.ModuleDict()
        self.seq_5layers = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10) for _ in range(5)]
        )

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_sequential(self):
        seq = self.seq_3layers
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty_sequential(self):
        seq = self.empty_seq
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_module_list(self):
        ml = self.ml_2modules
        self.assertEqual(len(ml), 2)
        self.assertEqual(ml.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_list(self):
        ml = self.empty_ml
        self.assertEqual(len(ml), 0)
        self.assertEqual(ml.__len__(), 0)

    @make_dynamo_test
    def test_len_module_dict(self):
        md = self.md_2modules
        self.assertEqual(len(md), 2)
        self.assertEqual(md.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_dict(self):
        md = self.empty_md
        self.assertEqual(len(md), 0)
        self.assertEqual(md.__len__(), 0)

    @make_dynamo_test
    def test_len_sequential_with_multiple_layers(self):
        seq = self.seq_5layers
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.__len__(), 5)


class TestDictViewLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on dict view objects (keys, values, items)"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_dict_keys_view(self):
        d = {"a": 1, "b": 2, "c": 3}
        keys = d.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_values_view(self):
        d = {"a": 1, "b": 2, "c": 3}
        values = d.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_items_view(self):
        d = {"x": 10, "y": 20, "z": 30}
        items = d.items()
        self.assertEqual(len(items), 3)
        self.assertEqual(items.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_keys_empty(self):
        d = {}
        keys = d.keys()
        self.assertEqual(len(keys), 0)
        self.assertEqual(keys.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_values_empty(self):
        d = {}
        values = d.values()
        self.assertEqual(len(values), 0)
        self.assertEqual(values.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_items_empty(self):
        d = {}
        items = d.items()
        self.assertEqual(len(items), 0)
        self.assertEqual(items.__len__(), 0)


# User-defined classes for TestUserDefinedLen
class CustomList:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)


class CustomContainer:
    def __len__(self):
        return 0


class Container:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class FixedSize:
    def __len__(self):
        return 10


class ListWrapper:
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.data)


class TestUserDefinedLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on user-defined classes with __len__"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_simple_custom_class(self):
        obj = CustomList([1, 2, 3])
        self.assertEqual(len(obj), 3)
        self.assertEqual(obj.__len__(), 3)
        self.assertEqual(CustomList.__len__(obj), 3)

    @make_dynamo_test
    def test_len_custom_class_empty(self):
        obj = CustomContainer()
        self.assertEqual(len(obj), 0)
        self.assertEqual(obj.__len__(), 0)
        self.assertEqual(CustomContainer.__len__(obj), 0)

    @make_dynamo_test
    def test_len_custom_class_with_properties(self):
        obj = Container(42)
        self.assertEqual(len(obj), 42)
        self.assertEqual(obj.__len__(), 42)
        self.assertEqual(Container.__len__(obj), 42)

    @make_dynamo_test
    def test_len_custom_class_constant_return(self):
        obj = FixedSize()
        self.assertEqual(len(obj), 10)
        self.assertEqual(obj.__len__(), 10)
        self.assertEqual(FixedSize.__len__(obj), 10)

    @make_dynamo_test
    def test_len_custom_class_with_list_attr(self):
        obj = ListWrapper()
        self.assertEqual(len(obj), 5)
        self.assertEqual(obj.__len__(), 5)
        self.assertEqual(ListWrapper.__len__(obj), 5)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
