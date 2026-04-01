# Owner(s): ["module: dynamo"]
"""Tests for mp_subscript_impl: unified __getitem__ dispatch via vt_getitem in Dynamo.

Each test uses operator.getitem() to exercise the vt_getitem → mp_subscript_impl
path (BuiltinVariable.call_getitem → vt_getitem → VT.mp_subscript_impl), which
is distinct from the call_method("__getitem__") path.
"""

import collections
import operator
import types
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_GPU


requires_gpu_and_triton = unittest.skipUnless(
    HAS_GPU and HAS_CUDA_AND_TRITON, "requires gpu and triton"
)


class GetItemTests(torch._dynamo.test_case.TestCase):
    def _compile(self, fn, *args):
        return torch.compile(fn, backend="eager", fullgraph=True)(*args)

    # --- BaseListVariable (ListVariable) ---

    def test_list_int_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_slice(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, slice(0, 2))

        x = torch.randn(4)
        ref = fn(x)
        res = self._compile(fn, x)
        for r, e in zip(res, ref):
            self.assertEqual(r, e)

    def test_list_negative_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_bool_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, True)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_invalid_index_type(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- BaseListVariable (TupleVariable) ---

    def test_tuple_int_index(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- RangeVariable ---

    def test_range_int_index(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, 3)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_slice(self):
        def fn(x):
            r = range(0, 10, 2)
            result = operator.getitem(r, slice(1, 3))
            return x + result[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- SizeVariable ---

    def test_size_int_index(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, 0)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- ConstDictVariable ---

    def test_dict_str_key(self):
        def fn(x):
            d = {"a": x, "b": x + 1}
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_int_key(self):
        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- DefaultDictVariable ---

    def test_defaultdict_existing_key(self):
        def fn(x):
            d = collections.defaultdict(lambda: x + 99)
            d["a"] = x
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TensorVariable ---

    def test_tensor_int_index(self):
        def fn(x):
            return operator.getitem(x, 0)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_slice(self):
        def fn(x):
            return operator.getitem(x, slice(0, 2))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_tuple_index(self):
        def fn(x):
            return operator.getitem(x, (0, slice(1, 3)))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- NamedTupleVariable (via UserDefinedTupleVariable) ---

    def test_namedtuple_int_index(self):
        def fn(x):
            result = torch.topk(x, 2)
            return operator.getitem(result, 1)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_namedtuple_values_index(self):
        def fn(x):
            result = torch.topk(x, 2)
            return operator.getitem(result, 0)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TypingVariable ---

    def test_typing_subscript(self):
        def fn(x):
            operator.getitem(list, int)
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- MappingProxyVariable ---

    def test_mappingproxy_getitem(self):
        def fn(x):
            d = {"a": 1, "b": 2}
            proxy = types.MappingProxyType(d)
            return x + operator.getitem(proxy, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- NNModuleVariable (ModuleList) ---

    def test_nn_module_list_int_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

            def forward(self, x):
                return operator.getitem(self.layers, 1)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (ModuleDict) ---

    def test_nn_module_dict_str_key(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleDict({"fc": torch.nn.Linear(4, 4)})

            def forward(self, x):
                return operator.getitem(self.layers, "fc")(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (Sequential) ---

    def test_nn_sequential_int_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return operator.getitem(self.seq, 0)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- UserDefinedObjectVariable ---

    def test_user_defined_object_getitem(self):
        class Container:
            def __init__(self, items):
                self.items = items

            def __getitem__(self, key):
                return self.items[key]

        def fn(x):
            c = Container([x, x + 1])
            return operator.getitem(c, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_object_without_getitem(self):
        class NoGetItem:
            pass

        def fn(x):
            obj = NoGetItem()
            return operator.getitem(obj, 0)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- UserDefinedListVariable ---

    def test_user_defined_list_getitem(self):
        class MyList(list):
            pass

        def fn(x):
            items = MyList([x, x + 1, x + 2])
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefinedDictVariable ---

    def test_user_defined_dict_getitem(self):
        class MyDict(dict):
            pass

        def fn(x):
            d = MyDict(a=x, b=x + 1)
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_missing(self):
        class MyDict(dict):
            def __missing__(self, key):
                return 42

        def fn(x):
            d = MyDict(a=1)
            return x + operator.getitem(d, "b")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_custom_missing(self):
        class DefaultDict(dict):
            def __missing__(self, key):
                self[key] = len(self)
                return self[key]

        def fn(x):
            d = DefaultDict()
            d["a"] = 1
            val = operator.getitem(d, "b")
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_collections_counter_getitem(self):
        def fn(x):
            c = collections.Counter({"a": 1, "b": 2})
            return x + operator.getitem(c, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- GetAttrVariable (__dict__ access) ---

    def test_getattr_dict_getitem(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                layer = operator.getitem(self.__dict__["_modules"], "linear")
                return layer(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- TorchScriptObjectVariable ---

    def test_opaque_object_getitem(self):
        from torch._library.opaque_object import (
            MemberType,
            OpaqueBase,
            register_opaque_type,
        )

        class OpaqueScaler(OpaqueBase):
            def __init__(self, scale):
                self.scale = scale

            def apply(self, x):
                return x * self.scale

        class OpaqueContainer(OpaqueBase):
            def __init__(self, items):
                self.items = items

            def __getitem__(self, idx):
                return self.items[idx]

        register_opaque_type(
            OpaqueScaler,
            typ="reference",
            members={
                "scale": MemberType.USE_REAL,
                "apply": MemberType.INLINED,
            },
        )
        register_opaque_type(
            OpaqueContainer,
            typ="reference",
            members={
                "items": MemberType.USE_REAL,
                "__getitem__": MemberType.INLINED,
            },
        )

        def fn(x, c):
            scaler = operator.getitem(c, 0)
            return scaler.apply(x)

        x = torch.randn(4)
        c = OpaqueContainer([OpaqueScaler(2.0), OpaqueScaler(3.0)])
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, c), compiled(x, c))

    # --- TritonKernelVariable ---

    @requires_gpu_and_triton
    def test_triton_kernel_getitem_grid(self):
        from torch.testing._internal.triton_utils import add_kernel

        def fn(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements // 256,)
            bound = operator.getitem(add_kernel, grid)
            bound(x, y, output, n_elements, BLOCK_SIZE=256)
            return output

        x = torch.randn(256, device="cuda")
        y = torch.randn(256, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, y), compiled(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
