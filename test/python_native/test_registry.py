# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import torch.library
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRegistry(TestCase):
    """Tests for the torch._native.registry module."""

    def setUp(self):
        """Clean up registry state before each test."""
        self.registry = self._get_registry_module()

        # Clear global state
        self.registry._libs.clear()
        self.registry._graphs.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()

    def _get_registry_module(self):
        """Import registry module directly to avoid importing all ops."""
        # Calculate path to registry.py relative to this test file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        pytorch_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels
        registry_path = os.path.join(pytorch_root, "torch", "_native", "registry.py")

        # Import registry module directly to avoid importing all ops
        spec = importlib.util.spec_from_file_location(
            "torch._native.registry", registry_path
        )
        registry = importlib.util.module_from_spec(spec)
        sys.modules["torch._native.registry"] = registry
        spec.loader.exec_module(registry)
        return registry

    def test_override_node_dataclass(self):
        """Test _OverrideNode dataclass creation and defaults."""

        def test_fn(x):
            return x

        # Test with minimal arguments
        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.override_fn, test_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

        # Test with all arguments
        def override_fn(x):
            return x * 2

        node = self.registry._OverrideNode(
            "another_dsl",
            "mul.Tensor",
            "CUDA",
            override_fn,
            unconditional_override=True,
            active=False,
        )
        self.assertEqual(node.dsl_name, "another_dsl")
        self.assertEqual(node.op_symbol, "mul.Tensor")
        self.assertEqual(node.dispatch_key, "CUDA")
        self.assertEqual(node.override_fn, override_fn)
        self.assertTrue(node.unconditional_override)
        self.assertFalse(node.active)

    def test_get_or_create_library_caching(self):
        """Test _get_or_create_library creates and caches Library instances."""
        # Test library creation and caching
        lib1 = self.registry._get_or_create_library("add.Tensor", "CPU")
        self.assertIsInstance(lib1, torch.library.Library)

        # Should return cached instance
        lib2 = self.registry._get_or_create_library("add.Tensor", "CPU")
        self.assertIs(lib1, lib2)

        # Different key should create different instance
        lib3 = self.registry._get_or_create_library("add.Tensor", "CUDA")
        self.assertIsNot(lib1, lib3)

        # Check that libraries are stored in _libs
        self.assertIn(("add.Tensor", "CPU"), self.registry._libs)
        self.assertIn(("add.Tensor", "CUDA"), self.registry._libs)

    def test_resolve_iterable_none_input(self):
        """Test _resolve_iterable with None input."""
        result = list(self.registry._resolve_iterable(None))
        self.assertEqual(result, [])

    def test_resolve_iterable_string_input(self):
        """Test _resolve_iterable with string input."""
        result = list(self.registry._resolve_iterable("single_string"))
        self.assertEqual(result, ["single_string"])

    def test_resolve_iterable_list_input(self):
        """Test _resolve_iterable with list input."""
        test_list = ["item1", "item2", "item3"]
        result = list(self.registry._resolve_iterable(test_list))
        self.assertEqual(result, test_list)

    def test_filter_no_filters_raises_error(self):
        """Test _filter raises ValueError when no filters provided."""
        with self.assertRaises(ValueError) as cm:
            self.registry._filter("dsl", "op", "key")
        self.assertIn("Must pass 1+ of filter_", str(cm.exception))

    def test_filter_dsl_name_match(self):
        """Test _filter returns True for matching DSL name."""
        self.assertTrue(
            self.registry._filter("test_dsl", "op", "key", filter_dsl_names="test_dsl")
        )
        self.assertTrue(
            self.registry._filter(
                "test_dsl", "op", "key", filter_dsl_names=["test_dsl", "other"]
            )
        )
        self.assertFalse(
            self.registry._filter("test_dsl", "op", "key", filter_dsl_names="other_dsl")
        )

    def test_filter_op_symbol_match(self):
        """Test _filter returns True for matching op symbol."""
        self.assertTrue(
            self.registry._filter("dsl", "test_op", "key", filter_op_symbols="test_op")
        )
        self.assertTrue(
            self.registry._filter(
                "dsl", "test_op", "key", filter_op_symbols=["test_op", "other"]
            )
        )
        self.assertFalse(
            self.registry._filter("dsl", "test_op", "key", filter_op_symbols="other_op")
        )

    def test_filter_dispatch_key_match(self):
        """Test _filter returns True for matching dispatch key."""
        self.assertTrue(
            self.registry._filter(
                "dsl", "op", "test_key", filter_dispatch_keys="test_key"
            )
        )
        self.assertTrue(
            self.registry._filter(
                "dsl", "op", "test_key", filter_dispatch_keys=["test_key", "other"]
            )
        )
        self.assertFalse(
            self.registry._filter(
                "dsl", "op", "test_key", filter_dispatch_keys="other_key"
            )
        )

    def test_update_registration_maps(self):
        """Test _update_registration_maps updates all mapping dictionaries."""
        key = ("test_op", "CPU")
        self.registry._update_registration_maps("test_dsl", "test_op", "CPU", key)

        # Check that all mappings were updated
        self.assertIn("test_dsl", self.registry._dsl_name_to_lib_graph)
        self.assertIn("test_op", self.registry._op_symbol_to_lib_graph)
        self.assertIn("CPU", self.registry._dispatch_key_to_lib_graph)

        # Check that the key is in the mapping values
        self.assertIn(key, self.registry._dsl_name_to_lib_graph["test_dsl"])
        self.assertIn(key, self.registry._op_symbol_to_lib_graph["test_op"])
        self.assertIn(key, self.registry._dispatch_key_to_lib_graph["CPU"])

        # Test appending to existing entry
        key2 = ("test_op2", "CUDA")
        self.registry._update_registration_maps("test_dsl", "test_op2", "CUDA", key2)

        # Should have both keys for test_dsl
        self.assertEqual(len(self.registry._dsl_name_to_lib_graph["test_dsl"]), 2)
        self.assertIn(key, self.registry._dsl_name_to_lib_graph["test_dsl"])
        self.assertIn(key2, self.registry._dsl_name_to_lib_graph["test_dsl"])

    def test__build_key_set(self):
        """Test __build_key_set creates correct set of keys."""
        # Set up test data
        key1 = ("op1", "CPU")
        key2 = ("op2", "CUDA")
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1]
        self.registry._op_symbol_to_lib_graph["op2"] = [key2]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1]

        # Test single filter
        result = self.registry._build_key_set("dsl1", None, None)
        self.assertEqual(result, {key1})

        # Test multiple filters
        result = self.registry._build_key_set("dsl1", "op2", None)
        self.assertEqual(result, {key1, key2})

    @patch("torch.library.Library")
    def testregister_op_override_basic(self, mock_library_cls):
        """Test basic register_op_override functionality."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def impl_fn(x):
            return x

        self.registry.register_op_override(
            backend="test_backend",
            lib_symbol="aten",
            op_symbol="add.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
        )

        # Check that graph was updated
        key = ("add.Tensor", "CPU")
        self.assertIn(key, self.registry._graphs)
        self.assertEqual(len(self.registry._graphs[key]), 1)

        node = self.registry._graphs[key][0]
        self.assertIsInstance(node, self.registry._OverrideNode)
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.override_fn, impl_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

        # Check that library.impl was called
        mock_lib.impl.assert_called_once_with(
            "add.Tensor", impl_fn, "CPU", with_keyset=True, allow_override=False
        )

    @patch("torch.library.Library")
    def testregister_op_override_unconditional(self, mock_library_cls):
        """Test register_op_override with unconditional_override=True."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def impl_fn(x):
            return x

        self.registry.register_op_override(
            backend="test_backend",
            lib_symbol="aten",
            op_symbol="add.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
            unconditional_override=True,
        )

        # Check node properties
        key = ("add.Tensor", "CPU")
        node = self.registry._graphs[key][0]
        self.assertTrue(node.unconditional_override)

        # Check that library.impl was called with correct parameters
        mock_lib.impl.assert_called_once_with(
            "add.Tensor",
            impl_fn,
            "CPU",
            with_keyset=False,  # Should be False for unconditional override
            allow_override=False,
        )

    @patch("torch.library.Library")
    def testregister_op_override_allow_multiple(self, mock_library_cls):
        """Test register_op_override with allow_multiple_override=True."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def impl_fn(x):
            return x

        self.registry.register_op_override(
            backend="test_backend",
            lib_symbol="aten",
            op_symbol="add.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
            allow_multiple_override=True,
        )

        # Check that library.impl was called with allow_override=True
        mock_lib.impl.assert_called_once_with(
            "add.Tensor", impl_fn, "CPU", with_keyset=True, allow_override=True
        )

    @patch("torch.library.Library")
    def testderegister_op_overrides(self, mock_library_cls):
        """Test deregister_op_overrides functionality."""
        # Set up test data
        key = ("add.Tensor", "CPU")

        def test_fn(x):
            return x

        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn, active=True
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "add.Tensor", "CPU", test_fn, active=True
        )
        self.registry._graphs[key] = [node1, node2]

        mock_old_lib = MagicMock()
        mock_new_lib = MagicMock()
        self.registry._libs[key] = mock_old_lib
        mock_library_cls.return_value = mock_new_lib

        # Set up mappings for __build_key_set to work
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key]

        self.registry.deregister_op_overrides(disable_dsl_names="dsl1")

        # Check that old library was removed and new one created
        self.assertIn(key, self.registry._libs)  # Should have new library

        # Check node states - dsl1 should be inactive, dsl2 should be active
        self.assertFalse(node1.active)
        self.assertTrue(node2.active)

        # Check that only non-filtered node was re-registered
        mock_new_lib.impl.assert_called_once_with(
            "add.Tensor",
            node2.override_fn,
            "CPU",
            with_keyset=True,
            allow_override=True,
        )

    def test_print_override_graphs_active_only(self):
        """Test _print_override_graphs with default settings."""
        # Set up test data
        key = ("add.Tensor", "CPU")

        def test_fn(x):
            return x

        active_node = self.registry._OverrideNode(
            "active_dsl", "add.Tensor", "CPU", test_fn, active=True
        )
        inactive_node = self.registry._OverrideNode(
            "inactive_dsl", "add.Tensor", "CPU", test_fn, active=False
        )
        self.registry._graphs[key] = [active_node, inactive_node]

        with patch("builtins.print") as mock_print:
            self.registry._print_override_graphs()

        # Should print header and only active node
        self.assertTrue(
            any("op='add.Tensor'" in str(call) for call in mock_print.call_args_list)
        )
        self.assertTrue(
            any("active_dsl" in str(call) for call in mock_print.call_args_list)
        )
        # Should not print inactive node
        self.assertFalse(
            any("inactive_dsl" in str(call) for call in mock_print.call_args_list)
        )

    def test_print_override_graphs_include_inactive(self):
        """Test _print_override_graphs with print_inactive=True."""
        # Set up test data
        key = ("add.Tensor", "CPU")

        def test_fn(x):
            return x

        active_node = self.registry._OverrideNode(
            "active_dsl", "add.Tensor", "CPU", test_fn, active=True
        )
        inactive_node = self.registry._OverrideNode(
            "inactive_dsl", "add.Tensor", "CPU", test_fn, active=False
        )
        self.registry._graphs[key] = [active_node, inactive_node]

        with patch("builtins.print") as mock_print:
            self.registry._print_override_graphs(print_inactive=True)

        # Should print both nodes with active status
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("active_dsl" in call for call in print_calls))
        self.assertTrue(any("inactive_dsl" in call for call in print_calls))
        self.assertTrue(any("active=True" in call for call in print_calls))
        self.assertTrue(any("active=False" in call for call in print_calls))

    @patch("torch.library.Library")
    def test_integration_register_and_deregister(self, mock_library_cls):
        """Integration test for register and deregister workflow."""

        def impl_fn1(x):
            return x + 1

        def impl_fn2(x):
            return x + 2

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register two overrides
        self.registry.register_op_override(
            "dsl1", "aten", "add.Tensor", "CPU", impl_fn1
        )
        self.registry.register_op_override(
            "dsl2", "aten", "add.Tensor", "CPU", impl_fn2
        )

        # Check both are registered
        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 2)
        self.assertTrue(all(node.active for node in self.registry._graphs[key]))

        # Deregister one
        self.registry.deregister_op_overrides(disable_dsl_names="dsl1")

        # Check that dsl1 is inactive, dsl2 is still active
        nodes = self.registry._graphs[key]
        dsl1_node = next(n for n in nodes if n.dsl_name == "dsl1")
        dsl2_node = next(n for n in nodes if n.dsl_name == "dsl2")

        self.assertFalse(dsl1_node.active)
        self.assertTrue(dsl2_node.active)

    @patch("torch.library.Library")
    def test_multiple_override_chain_basic(self, mock_library_cls):
        """Test multiple overrides on same operation form correct chain."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register multiple overrides for the same operation
        def impl_fn1(dispatch_keys, x, y):
            return ("backend1", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("backend2", x, y)

        def impl_fn3(dispatch_keys, x, y):
            return ("backend3", x, y)

        self.registry.register_op_override(
            "backend1",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend2",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend3",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn3,
            allow_multiple_override=True,
        )

        key = ("mul.Tensor", "CUDA")
        self.assertEqual(len(self.registry._graphs[key]), 3)

        # Check that all overrides are active and in correct order
        nodes = self.registry._graphs[key]
        self.assertEqual(nodes[0].dsl_name, "backend1")
        self.assertEqual(nodes[1].dsl_name, "backend2")
        self.assertEqual(nodes[2].dsl_name, "backend3")
        self.assertTrue(all(node.active for node in nodes))

    @patch("torch.library.Library")
    def test_remove_override_from_middle_of_chain(self, mock_library_cls):
        """Test removing override from middle of chain preserves correct order."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register three overrides
        def impl_fn1(dispatch_keys, x, y):
            return ("backend1", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("backend2", x, y)

        def impl_fn3(dispatch_keys, x, y):
            return ("backend3", x, y)

        self.registry.register_op_override(
            "backend1",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend2",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend3",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn3,
            allow_multiple_override=True,
        )

        # Remove middle override (backend2)
        self.registry.deregister_op_overrides(disable_dsl_names="backend2")

        key = ("mul.Tensor", "CUDA")
        nodes = self.registry._graphs[key]

        # Check that backend2 is inactive, others are active
        backend1_node = next(n for n in nodes if n.dsl_name == "backend1")
        backend2_node = next(n for n in nodes if n.dsl_name == "backend2")
        backend3_node = next(n for n in nodes if n.dsl_name == "backend3")

        self.assertTrue(backend1_node.active)
        self.assertFalse(backend2_node.active)  # Should be inactive
        self.assertTrue(backend3_node.active)

        # The mock should have been called for re-registering active overrides
        # Note: the exact number of calls depends on how many times _get_or_create_library creates new instances
        self.assertTrue(mock_lib.impl.call_count >= 2)

    @patch("torch.library.Library")
    def test_remove_first_override_in_chain(self, mock_library_cls):
        """Test removing first override in chain."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register three overrides
        def impl_fn1(dispatch_keys, x, y):
            return ("backend1", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("backend2", x, y)

        def impl_fn3(dispatch_keys, x, y):
            return ("backend3", x, y)

        self.registry.register_op_override(
            "backend1",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend2",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend3",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn3,
            allow_multiple_override=True,
        )

        # Remove first override (backend1)
        self.registry.deregister_op_overrides(disable_dsl_names="backend1")

        key = ("mul.Tensor", "CUDA")
        nodes = self.registry._graphs[key]

        # Check states
        backend1_node = next(n for n in nodes if n.dsl_name == "backend1")
        backend2_node = next(n for n in nodes if n.dsl_name == "backend2")
        backend3_node = next(n for n in nodes if n.dsl_name == "backend3")

        self.assertFalse(backend1_node.active)  # Should be inactive
        self.assertTrue(backend2_node.active)
        self.assertTrue(backend3_node.active)

    @patch("torch.library.Library")
    def test_remove_last_override_in_chain(self, mock_library_cls):
        """Test removing last override in chain."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register three overrides
        def impl_fn1(dispatch_keys, x, y):
            return ("backend1", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("backend2", x, y)

        def impl_fn3(dispatch_keys, x, y):
            return ("backend3", x, y)

        self.registry.register_op_override(
            "backend1",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend2",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "backend3",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn3,
            allow_multiple_override=True,
        )

        # Remove last override (backend3)
        self.registry.deregister_op_overrides(disable_dsl_names="backend3")

        key = ("mul.Tensor", "CUDA")
        nodes = self.registry._graphs[key]

        # Check states
        backend1_node = next(n for n in nodes if n.dsl_name == "backend1")
        backend2_node = next(n for n in nodes if n.dsl_name == "backend2")
        backend3_node = next(n for n in nodes if n.dsl_name == "backend3")

        self.assertTrue(backend1_node.active)
        self.assertTrue(backend2_node.active)
        self.assertFalse(backend3_node.active)  # Should be inactive

    @patch("torch.library.Library")
    def test_remove_multiple_overrides_from_chain(self, mock_library_cls):
        """Test removing multiple overrides from chain."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register five overrides
        impl_fns = []
        for i in range(5):

            def make_impl_fn(backend_num):
                def impl_fn(dispatch_keys, x, y):
                    return (f"backend{backend_num}", x, y)

                return impl_fn

            impl_fn = make_impl_fn(i + 1)
            impl_fns.append(impl_fn)
            self.registry.register_op_override(
                f"backend{i + 1}",
                "aten",
                "mul.Tensor",
                "CUDA",
                impl_fn,
                allow_multiple_override=True,
            )

        # Remove multiple overrides (backend2 and backend4)
        self.registry.deregister_op_overrides(
            disable_dsl_names=["backend2", "backend4"]
        )

        key = ("mul.Tensor", "CUDA")
        nodes = self.registry._graphs[key]

        # Check states - backend2 and backend4 should be inactive, others active
        expected_active = {"backend1", "backend3", "backend5"}
        expected_inactive = {"backend2", "backend4"}

        for node in nodes:
            if node.dsl_name in expected_active:
                self.assertTrue(node.active, f"{node.dsl_name} should be active")
            elif node.dsl_name in expected_inactive:
                self.assertFalse(node.active, f"{node.dsl_name} should be inactive")

    @patch("torch.library.Library")
    def test_remove_all_overrides_from_chain(self, mock_library_cls):
        """Test removing all overrides from chain."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register three overrides
        for i in range(3):

            def make_impl_fn(backend_num):
                def impl_fn(dispatch_keys, x, y):
                    return (f"backend{backend_num}", x, y)

                return impl_fn

            impl_fn = make_impl_fn(i + 1)
            self.registry.register_op_override(
                f"backend{i + 1}",
                "aten",
                "mul.Tensor",
                "CUDA",
                impl_fn,
                allow_multiple_override=True,
            )

        # Remove all overrides
        self.registry.deregister_op_overrides(
            disable_dsl_names=["backend1", "backend2", "backend3"]
        )

        key = ("mul.Tensor", "CUDA")
        nodes = self.registry._graphs[key]

        # All should be inactive
        self.assertTrue(all(not node.active for node in nodes))

        # No re-registrations should occur since all are filtered out
        # Note: _get_or_create_library might still be called to create the new lib instance

    @patch("torch.library.Library")
    def test_deregister_by_op_symbol_affects_all_backends(self, mock_library_cls):
        """Test deregistering by op_symbol affects all backends for that operation."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register multiple backends for same operation
        def impl_fn1(dispatch_keys, x, y):
            return ("triton", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("cutedsl", x, y)

        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "cutedsl",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )

        # Also register same backends for different operation
        def add_impl_fn1(dispatch_keys, x, y):
            return ("triton", x, y)

        def add_impl_fn2(dispatch_keys, x, y):
            return ("cutedsl", x, y)

        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CUDA",
            add_impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "cutedsl",
            "aten",
            "add.Tensor",
            "CUDA",
            add_impl_fn2,
            allow_multiple_override=True,
        )

        # Remove by op_symbol should affect all backends for that op only
        self.registry.deregister_op_overrides(disable_op_symbols="mul.Tensor")

        # Check mul.Tensor overrides are inactive
        mul_key = ("mul.Tensor", "CUDA")
        mul_nodes = self.registry._graphs[mul_key]
        self.assertTrue(all(not node.active for node in mul_nodes))

        # Check add.Tensor overrides are still active
        add_key = ("add.Tensor", "CUDA")
        add_nodes = self.registry._graphs[add_key]
        self.assertTrue(all(node.active for node in add_nodes))

    @patch("torch.library.Library")
    def test_deregister_by_dispatch_key_affects_all_operations(self, mock_library_cls):
        """Test deregistering by dispatch_key affects all operations for that key."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register same backend for multiple operations and dispatch keys
        def impl_fn_cuda(dispatch_keys, x, y):
            return ("triton_cuda", x, y)

        def impl_fn_cpu(dispatch_keys, x, y):
            return ("triton_cpu", x, y)

        # mul.Tensor on CUDA and CPU
        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn_cuda,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CPU",
            impl_fn_cpu,
            allow_multiple_override=True,
        )

        # add.Tensor on CUDA and CPU
        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CUDA",
            impl_fn_cuda,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CPU",
            impl_fn_cpu,
            allow_multiple_override=True,
        )

        # Remove by dispatch_key should affect all operations for CUDA only
        self.registry.deregister_op_overrides(disable_dispatch_keys="CUDA")

        # Check CUDA overrides are inactive
        mul_cuda_key = ("mul.Tensor", "CUDA")
        add_cuda_key = ("add.Tensor", "CUDA")

        mul_cuda_nodes = self.registry._graphs[mul_cuda_key]
        add_cuda_nodes = self.registry._graphs[add_cuda_key]

        self.assertTrue(all(not node.active for node in mul_cuda_nodes))
        self.assertTrue(all(not node.active for node in add_cuda_nodes))

        # Check CPU overrides are still active
        mul_cpu_key = ("mul.Tensor", "CPU")
        add_cpu_key = ("add.Tensor", "CPU")

        mul_cpu_nodes = self.registry._graphs[mul_cpu_key]
        add_cpu_nodes = self.registry._graphs[add_cpu_key]

        self.assertTrue(all(node.active for node in mul_cpu_nodes))
        self.assertTrue(all(node.active for node in add_cpu_nodes))

    @patch("torch.library.Library")
    def test_complex_multi_criteria_deregistration(self, mock_library_cls):
        """Test deregistration with multiple criteria (dsl_names + op_symbols + dispatch_keys)."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Create a complex scenario with multiple backends, ops, and dispatch keys
        backends = ["triton", "cutedsl", "openvino"]
        ops = ["mul.Tensor", "add.Tensor", "div.Tensor"]
        dispatch_keys = ["CUDA", "CPU"]

        # Register all combinations
        for backend in backends:
            for op in ops:
                for dispatch_key in dispatch_keys:

                    def make_impl_fn(b, o, d):
                        def impl_fn(dk, x, y):
                            return (f"{b}_{o}_{d}", x, y)

                        return impl_fn

                    impl_fn = make_impl_fn(backend, op, dispatch_key)
                    self.registry.register_op_override(
                        backend,
                        "aten",
                        op,
                        dispatch_key,
                        impl_fn,
                        allow_multiple_override=True,
                    )

        # Complex deregistration:
        # - Disable triton backend (affects all ops/dispatch_keys for triton)
        # - Disable mul.Tensor operation (affects all backends/dispatch_keys for mul.Tensor)
        # - Disable CPU dispatch key (affects all backends/ops for CPU)
        self.registry.deregister_op_overrides(
            disable_dsl_names="triton",
            disable_op_symbols="mul.Tensor",
            disable_dispatch_keys="CPU",
        )

        # Check results:
        # 1. All triton overrides should be inactive
        # 2. All mul.Tensor overrides should be inactive
        # 3. All CPU overrides should be inactive
        # 4. Only cutedsl/openvino + add.Tensor/div.Tensor + CUDA should remain active

        for op in ops:
            for dispatch_key in dispatch_keys:
                key = (op, dispatch_key)
                if key in self.registry._graphs:
                    nodes = self.registry._graphs[key]
                    for node in nodes:
                        should_be_active = (
                            node.dsl_name != "triton"  # triton should be inactive
                            and op != "mul.Tensor"  # mul.Tensor should be inactive
                            and dispatch_key != "CPU"  # CPU should be inactive
                        )

                        self.assertEqual(
                            node.active,
                            should_be_active,
                            f"Node {node.dsl_name}/{op}/{dispatch_key} active state incorrect",
                        )

    def test_integration_with_real_torch_library(self):
        """Integration test using real torch.library.Library to ensure actual PyTorch integration works."""
        # Use a unique operation name to avoid conflicts
        test_op = "test_registry_integration.Tensor"

        try:
            # Create real implementation functions
            def impl1(dispatch_keys, x):
                return x + 1

            def impl2(dispatch_keys, x):
                return x + 2

            # Register with real torch.library.Library (no mocking)
            self.registry.register_op_override(
                "test_backend1",
                "aten",
                test_op,
                "CPU",
                impl1,
                allow_multiple_override=True,
            )

            self.registry.register_op_override(
                "test_backend2",
                "aten",
                test_op,
                "CPU",
                impl2,
                allow_multiple_override=True,
            )

            # Verify registry state
            key = (test_op, "CPU")
            self.assertIn(key, self.registry._graphs)
            self.assertEqual(len(self.registry._graphs[key]), 2)

            # Test deregistration
            self.registry.deregister_op_overrides(disable_dsl_names="test_backend1")

            # Verify backend1 is inactive, backend2 is active
            nodes = self.registry._graphs[key]
            backend1_node = next(n for n in nodes if n.dsl_name == "test_backend1")
            backend2_node = next(n for n in nodes if n.dsl_name == "test_backend2")

            self.assertFalse(backend1_node.active)
            self.assertTrue(backend2_node.active)

            # Verify library was actually created
            self.assertIn(key, self.registry._libs)
            self.assertIsInstance(self.registry._libs[key], torch.library.Library)

        except Exception as e:
            # If this fails, it might reveal issues that mocked tests miss
            self.fail(
                f"Integration test failed, suggesting mocking may hide real issues: {e}"
            )
        finally:
            # Clean up - remove our test registrations
            if key in self.registry._libs:
                del self.registry._libs[key]
            if key in self.registry._graphs:
                del self.registry._graphs[key]
            # Clean up mappings
            for mapping in [
                self.registry._dsl_name_to_lib_graph,
                self.registry._op_symbol_to_lib_graph,
                self.registry._dispatch_key_to_lib_graph,
            ]:
                keys_to_remove = []
                for k, v in mapping.items():
                    if key in v:
                        v.remove(key)
                        if not v:  # Remove empty lists
                            keys_to_remove.append(k)
                for k in keys_to_remove:
                    del mapping[k]

    def test_integration_override_chain_middle_removal(self):
        """Integration test: removing override from middle of chain with real torch.library.Library."""
        test_op = "test_middle_removal.Tensor"
        key = (test_op, "CPU")

        try:
            # Create real implementation functions that return identifiable values
            def backend1_impl(dispatch_keys, x):
                return x.clone() + 100  # Distinctive return value

            def backend2_impl(dispatch_keys, x):
                return x.clone() + 200  # Will be removed

            def backend3_impl(dispatch_keys, x):
                return x.clone() + 300  # Should remain

            # Register three overrides in sequence
            self.registry.register_op_override(
                "integration_backend1",
                "aten",
                test_op,
                "CPU",
                backend1_impl,
                allow_multiple_override=True,
            )
            self.registry.register_op_override(
                "integration_backend2",
                "aten",
                test_op,
                "CPU",
                backend2_impl,
                allow_multiple_override=True,
            )
            self.registry.register_op_override(
                "integration_backend3",
                "aten",
                test_op,
                "CPU",
                backend3_impl,
                allow_multiple_override=True,
            )

            # Verify all three are registered
            self.assertEqual(len(self.registry._graphs[key]), 3)
            self.assertTrue(all(node.active for node in self.registry._graphs[key]))

            # Remove middle override (backend2)
            # Note: PyTorch may warn about kernel override (but only shows warning once per session)
            self.registry.deregister_op_overrides(
                disable_dsl_names="integration_backend2"
            )

            # Verify registry state
            nodes = self.registry._graphs[key]
            backend1_node = next(
                n for n in nodes if n.dsl_name == "integration_backend1"
            )
            backend2_node = next(
                n for n in nodes if n.dsl_name == "integration_backend2"
            )
            backend3_node = next(
                n for n in nodes if n.dsl_name == "integration_backend3"
            )

            self.assertTrue(backend1_node.active)
            self.assertFalse(backend2_node.active)  # Should be inactive
            self.assertTrue(backend3_node.active)

            # Verify torch.library.Library was actually recreated
            self.assertIn(key, self.registry._libs)
            self.assertIsInstance(self.registry._libs[key], torch.library.Library)

        finally:
            self._cleanup_test_registration(key)

    def test_integration_complex_multi_criteria_deregistration(self):
        """Integration test: complex multi-criteria deregistration with real torch.library.Library."""
        # Use multiple unique operations to avoid conflicts
        test_ops = ["test_complex_mul.Tensor", "test_complex_add.Tensor"]
        dispatch_keys = ["CPU"]  # Stick to CPU to avoid CUDA requirements
        backends = ["integration_triton", "integration_cutedsl"]

        keys_created = []

        try:
            # Register multiple combinations
            for backend in backends:
                for op in test_ops:
                    for dispatch_key in dispatch_keys:
                        key = (op, dispatch_key)
                        keys_created.append(key)

                        def make_impl(b, o, d):
                            def impl_fn(dispatch_keys, x):
                                return x.clone() + hash(f"{b}_{o}_{d}") % 1000

                            return impl_fn

                        impl_fn = make_impl(backend, op, dispatch_key)
                        self.registry.register_op_override(
                            backend,
                            "aten",
                            op,
                            dispatch_key,
                            impl_fn,
                            allow_multiple_override=True,
                        )

            # Verify all registrations
            for key in keys_created:
                self.assertIn(key, self.registry._graphs)
                self.assertTrue(all(node.active for node in self.registry._graphs[key]))

            # Complex deregistration: disable integration_triton backend + test_complex_mul.Tensor operation
            # Note: PyTorch may warn about kernel override (but only shows warning once per session)
            self.registry.deregister_op_overrides(
                disable_dsl_names="integration_triton",
                disable_op_symbols="test_complex_mul.Tensor",
            )

            # Verify results
            for key in keys_created:
                op, dispatch_key = key
                if key in self.registry._graphs:
                    nodes = self.registry._graphs[key]
                    for node in nodes:
                        # Should be active only if:
                        # - Not integration_triton backend AND
                        # - Not test_complex_mul.Tensor operation
                        should_be_active = (
                            node.dsl_name != "integration_triton"
                            and op != "test_complex_mul.Tensor"
                        )

                        self.assertEqual(
                            node.active,
                            should_be_active,
                            f"Integration test: Node {node.dsl_name}/{op}/{dispatch_key} active state incorrect",
                        )

            # Verify libraries still exist and are real torch.library.Library instances
            for key in keys_created:
                if key in self.registry._libs:
                    self.assertIsInstance(
                        self.registry._libs[key], torch.library.Library
                    )

        finally:
            for key in keys_created:
                self._cleanup_test_registration(key)

    def test_integration_deregister_all_from_chain(self):
        """Integration test: removing all overrides from chain with real torch.library.Library."""
        test_op = "test_remove_all.Tensor"
        key = (test_op, "CPU")

        try:
            backends = ["integration_all1", "integration_all2", "integration_all3"]

            # Register multiple overrides
            for i, backend in enumerate(backends):

                def make_impl(idx):
                    def impl_fn(dispatch_keys, x):
                        return x.clone() + (idx + 1) * 100

                    return impl_fn

                impl_fn = make_impl(i)
                self.registry.register_op_override(
                    backend,
                    "aten",
                    test_op,
                    "CPU",
                    impl_fn,
                    allow_multiple_override=True,
                )

            # Verify all are active
            self.assertEqual(len(self.registry._graphs[key]), 3)
            self.assertTrue(all(node.active for node in self.registry._graphs[key]))

            # Remove all overrides
            # Note: PyTorch may warn about kernel override (but only shows warning once per session)
            self.registry.deregister_op_overrides(disable_dsl_names=backends)

            # Verify all are inactive
            nodes = self.registry._graphs[key]
            self.assertTrue(all(not node.active for node in nodes))

            # Verify library still exists (it gets recreated even with no active overrides)
            self.assertIn(key, self.registry._libs)
            self.assertIsInstance(self.registry._libs[key], torch.library.Library)

        finally:
            self._cleanup_test_registration(key)

    def test_integration_registry_state_consistency_after_operations(self):
        """Integration test: verify registry state remains consistent after complex operations."""
        test_op = "test_consistency.Tensor"
        key = (test_op, "CPU")

        try:
            # Perform a series of registrations and deregistrations
            backends = ["consistency1", "consistency2", "consistency3", "consistency4"]

            # Initial registration
            for backend in backends:

                def make_impl_fn(b):
                    def impl_fn(dispatch_keys, x):
                        return x.clone() + hash(b) % 100

                    return impl_fn

                impl_fn = make_impl_fn(backend)
                self.registry.register_op_override(
                    backend,
                    "aten",
                    test_op,
                    "CPU",
                    impl_fn,
                    allow_multiple_override=True,
                )

            # Partial deregistration
            # Note: PyTorch may warn about kernel override (but only shows warning once per session)
            self.registry.deregister_op_overrides(
                disable_dsl_names=["consistency2", "consistency4"]
            )

            # Verify intermediate state
            nodes = self.registry._graphs[key]
            active_backends = {node.dsl_name for node in nodes if node.active}
            inactive_backends = {node.dsl_name for node in nodes if not node.active}

            self.assertEqual(active_backends, {"consistency1", "consistency3"})
            self.assertEqual(inactive_backends, {"consistency2", "consistency4"})

            # Re-register one that was deregistered
            def new_impl(dispatch_keys, x):
                return x.clone() + 999

            self.registry.register_op_override(
                "consistency2",
                "aten",
                test_op,
                "CPU",
                new_impl,
                allow_multiple_override=True,
            )

            # Verify final state - consistency2 should appear twice now (old inactive + new active)
            nodes = self.registry._graphs[key]
            consistency2_nodes = [n for n in nodes if n.dsl_name == "consistency2"]

            # Should have 2 nodes for consistency2: one inactive (old) and one active (new)
            self.assertEqual(len(consistency2_nodes), 2)
            active_consistency2_nodes = [n for n in consistency2_nodes if n.active]
            inactive_consistency2_nodes = [
                n for n in consistency2_nodes if not n.active
            ]

            self.assertEqual(len(active_consistency2_nodes), 1)
            self.assertEqual(len(inactive_consistency2_nodes), 1)

            # Verify mappings are still consistent
            self.assertIn("consistency2", self.registry._dsl_name_to_lib_graph)
            self.assertIn(test_op, self.registry._op_symbol_to_lib_graph)
            self.assertIn("CPU", self.registry._dispatch_key_to_lib_graph)

            # Verify all mapping entries point to the correct key
            for mapping in [
                self.registry._dsl_name_to_lib_graph,
                self.registry._op_symbol_to_lib_graph,
                self.registry._dispatch_key_to_lib_graph,
            ]:
                for key_list in mapping.values():
                    if key in key_list:
                        # Each mapping should contain valid keys
                        for k in key_list:
                            self.assertIsInstance(k, tuple)
                            self.assertEqual(len(k), 2)

        finally:
            self._cleanup_test_registration(key)

    def test_filter_state_initialization(self):
        """Test _FilterState initialization and default values."""
        filter_state = self.registry._FilterState()

        self.assertIsInstance(filter_state._dsl_names, set)
        self.assertIsInstance(filter_state._op_symbols, set)
        self.assertIsInstance(filter_state._dispatch_keys, set)

        self.assertEqual(len(filter_state._dsl_names), 0)
        self.assertEqual(len(filter_state._op_symbols), 0)
        self.assertEqual(len(filter_state._dispatch_keys), 0)

    def test_filter_state_check_enabled_all_empty(self):
        """Test _FilterState.check_enabled when all filter sets are empty."""
        filter_state = self.registry._FilterState()

        def test_fn(x):
            return x

        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)

        # Empty filters should enable all nodes
        self.assertTrue(filter_state.check_enabled(node))

    def test_filter_state_check_enabled_dsl_filtered(self):
        """Test _FilterState.check_enabled when DSL is filtered."""
        filter_state = self.registry._FilterState()
        filter_state._dsl_names.add("filtered_dsl")

        def test_fn(x):
            return x

        # Node with filtered DSL should be disabled
        filtered_node = self.registry._OverrideNode(
            "filtered_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))

        # Node with different DSL should be enabled
        enabled_node = self.registry._OverrideNode(
            "allowed_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertTrue(filter_state.check_enabled(enabled_node))

    def test_filter_state_check_enabled_op_filtered(self):
        """Test _FilterState.check_enabled when op symbol is filtered."""
        filter_state = self.registry._FilterState()
        filter_state._op_symbols.add("mul.Tensor")

        def test_fn(x):
            return x

        # Node with filtered op should be disabled
        filtered_node = self.registry._OverrideNode(
            "test_dsl", "mul.Tensor", "CPU", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))

        # Node with different op should be enabled
        enabled_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertTrue(filter_state.check_enabled(enabled_node))

    def test_filter_state_check_enabled_dispatch_key_filtered(self):
        """Test _FilterState.check_enabled when dispatch key is filtered."""
        filter_state = self.registry._FilterState()
        filter_state._dispatch_keys.add("CUDA")

        def test_fn(x):
            return x

        # Node with filtered dispatch key should be disabled
        filtered_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CUDA", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))

        # Node with different dispatch key should be enabled
        enabled_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertTrue(filter_state.check_enabled(enabled_node))

    def test_filter_state_update_add_single_values(self):
        """Test _FilterState.update adding single values."""
        filter_state = self.registry._FilterState()

        # Add single values
        filter_state.update("dsl1", "add.Tensor", "CPU")

        self.assertIn("dsl1", filter_state._dsl_names)
        self.assertIn("add.Tensor", filter_state._op_symbols)
        self.assertIn("CPU", filter_state._dispatch_keys)

    def test_filter_state_update_add_multiple_values(self):
        """Test _FilterState.update adding multiple values."""
        filter_state = self.registry._FilterState()

        # Add multiple values
        filter_state.update(
            ["dsl1", "dsl2"], ["add.Tensor", "mul.Tensor"], ["CPU", "CUDA"]
        )

        self.assertEqual(filter_state._dsl_names, {"dsl1", "dsl2"})
        self.assertEqual(filter_state._op_symbols, {"add.Tensor", "mul.Tensor"})
        self.assertEqual(filter_state._dispatch_keys, {"CPU", "CUDA"})

    def test_filter_state_update_remove_values(self):
        """Test _FilterState.update removing values."""
        filter_state = self.registry._FilterState()

        # Add initial values
        filter_state._dsl_names.update(["dsl1", "dsl2", "dsl3"])
        filter_state._op_symbols.update(["add.Tensor", "mul.Tensor"])
        filter_state._dispatch_keys.update(["CPU", "CUDA"])

        # Remove some values
        filter_state.update(["dsl1", "dsl3"], "mul.Tensor", "CUDA", remove_keys=True)

        self.assertEqual(filter_state._dsl_names, {"dsl2"})
        self.assertEqual(filter_state._op_symbols, {"add.Tensor"})
        self.assertEqual(filter_state._dispatch_keys, {"CPU"})

    def test_filter_state_update_none_values(self):
        """Test _FilterState.update with None values (should be no-op)."""
        filter_state = self.registry._FilterState()
        initial_dsl = {"dsl1"}
        filter_state._dsl_names = initial_dsl.copy()

        # Update with None should not change anything
        filter_state.update(None, None, None)

        self.assertEqual(filter_state._dsl_names, initial_dsl)
        self.assertEqual(len(filter_state._op_symbols), 0)
        self.assertEqual(len(filter_state._dispatch_keys), 0)

    def test_filter_state_build_disable_key_set(self):
        """Test _FilterState.build_disable_key_set creates correct key set."""
        filter_state = self.registry._FilterState()

        # Set up test mappings
        key1 = ("add.Tensor", "CPU")
        key2 = ("mul.Tensor", "CUDA")
        key3 = ("div.Tensor", "CPU")

        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1, key2]
        self.registry._op_symbol_to_lib_graph["add.Tensor"] = [key1]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1, key3]

        # Add filters
        filter_state._dsl_names.add("dsl1")
        filter_state._dispatch_keys.add("CPU")

        key_set = filter_state.build_disable_key_set()

        # Should include keys from both dsl1 and CPU filters
        expected_keys = {key1, key2, key3}
        self.assertEqual(key_set, expected_keys)

    def test__build_key_set_function(self):
        """Test _build_key_set standalone function."""
        # Set up test mappings
        key1 = ("add.Tensor", "CPU")
        key2 = ("mul.Tensor", "CUDA")
        key3 = ("div.Tensor", "CPU")

        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1, key2]
        self.registry._op_symbol_to_lib_graph["add.Tensor"] = [key1]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1, key3]

        # Test with single values
        key_set = self.registry._build_key_set("dsl1", None, None)
        self.assertEqual(key_set, {key1, key2})

        # Test with multiple criteria
        key_set = self.registry._build_key_set("dsl1", "add.Tensor", "CPU")
        expected_keys = {key1, key2, key3}  # Union of all matching keys
        self.assertEqual(key_set, expected_keys)

    @patch("torch.library.Library")
    def testreenable_op_overrides_basic(self, mock_library_cls):
        """Test reenable_op_overrides basic functionality."""

        # Set up initial state with some disabled overrides
        def test_fn1(x):
            return x + 1

        def test_fn2(x):
            return x + 2

        key = ("add.Tensor", "CPU")
        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn1, active=False
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "add.Tensor", "CPU", test_fn2, active=True
        )

        self.registry._graphs[key] = [node1, node2]
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key]

        # Add dsl1 to filter (disabled)
        self.registry._filter_state._dsl_names.add("dsl1")

        # Mock library
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib
        self.registry._libs[key] = mock_lib

        # Re-enable dsl1
        self.registry.reenable_op_overrides(enable_dsl_names="dsl1")

        # dsl1 should be removed from filter
        self.assertNotIn("dsl1", self.registry._filter_state._dsl_names)

        # Both nodes should now be active
        self.assertTrue(node1.active)
        self.assertTrue(node2.active)

        # Library impl should be called for the re-enabled node
        self.assertEqual(mock_lib.impl.call_count, 2)  # Once for each active node

    @patch("torch.library.Library")
    def testreenable_op_overrides_multiple_criteria(self, mock_library_cls):
        """Test reenable_op_overrides with multiple enable criteria."""

        # Set up test state
        def test_fn(x):
            return x

        key1 = ("add.Tensor", "CPU")
        key2 = ("mul.Tensor", "CUDA")

        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn, active=False
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "mul.Tensor", "CUDA", test_fn, active=False
        )

        self.registry._graphs[key1] = [node1]
        self.registry._graphs[key2] = [node2]

        # Set up mappings
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1]
        self.registry._op_symbol_to_lib_graph["mul.Tensor"] = [key2]

        # Add to filters
        self.registry._filter_state._dsl_names.add("dsl1")
        self.registry._filter_state._op_symbols.add("mul.Tensor")

        # Mock libraries
        mock_lib1 = MagicMock()
        mock_lib2 = MagicMock()
        mock_library_cls.side_effect = [mock_lib1, mock_lib2]

        # Re-enable both
        self.registry.reenable_op_overrides(
            enable_dsl_names="dsl1", enable_op_symbols="mul.Tensor"
        )

        # Both should be removed from filters
        self.assertNotIn("dsl1", self.registry._filter_state._dsl_names)
        self.assertNotIn("mul.Tensor", self.registry._filter_state._op_symbols)

        # Both nodes should be active
        self.assertTrue(node1.active)
        self.assertTrue(node2.active)

    @patch("torch.library.Library")
    def testreenable_op_overrides_with_remaining_filters(self, mock_library_cls):
        """Test reenable_op_overrides when some filters remain active."""

        def test_fn(x):
            return x

        key = ("add.Tensor", "CPU")
        node1 = self.registry._OverrideNode(
            "filtered_dsl", "add.Tensor", "CPU", test_fn, active=False
        )
        node2 = self.registry._OverrideNode(
            "allowed_dsl", "add.Tensor", "CPU", test_fn, active=False
        )

        self.registry._graphs[key] = [node1, node2]
        self.registry._dsl_name_to_lib_graph["allowed_dsl"] = [key]

        # Add both DSLs to filter initially
        self.registry._filter_state._dsl_names.update(["filtered_dsl", "allowed_dsl"])

        # Mock library
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib
        self.registry._libs[key] = mock_lib

        # Re-enable only allowed_dsl
        self.registry.reenable_op_overrides(enable_dsl_names="allowed_dsl")

        # Only allowed_dsl should be removed from filter
        self.assertIn("filtered_dsl", self.registry._filter_state._dsl_names)
        self.assertNotIn("allowed_dsl", self.registry._filter_state._dsl_names)

        # Only node2 should be active
        self.assertFalse(node1.active)  # Still filtered
        self.assertTrue(node2.active)  # Re-enabled

        # Library impl called only for active node
        mock_lib.impl.assert_called_once()

    def test_filter_state_print_method(self):
        """Test _FilterState.print method (basic functionality check)."""
        filter_state = self.registry._FilterState()
        filter_state._dsl_names.update(["dsl1", "dsl2"])
        filter_state._op_symbols.add("add.Tensor")
        filter_state._dispatch_keys.add("CPU")

        # Should not raise an exception
        try:
            str_repr = str(filter_state)
            self.assertIn("Filter State:", str_repr)
        except Exception as e:
            self.fail(f"FilterState.__str__() raised an exception: {e}")

    def test_global_filter_state_integration(self):
        """Test that global _filter_state is used correctly."""
        # Check that global filter state exists
        self.assertIsInstance(self.registry._filter_state, self.registry._FilterState)

        # Test that it's the same instance across operations
        filter_state_ref1 = self.registry._filter_state
        self.registry._filter_state._dsl_names.add("test_dsl")
        filter_state_ref2 = self.registry._filter_state

        self.assertIs(filter_state_ref1, filter_state_ref2)
        self.assertIn("test_dsl", filter_state_ref2._dsl_names)

    def _cleanup_test_registration(self, key):
        """Helper method to clean up test registrations."""
        # Clean up registry state
        if key in self.registry._libs:
            del self.registry._libs[key]
        if key in self.registry._graphs:
            del self.registry._graphs[key]

        # Clean up mappings
        for mapping in [
            self.registry._dsl_name_to_lib_graph,
            self.registry._op_symbol_to_lib_graph,
            self.registry._dispatch_key_to_lib_graph,
        ]:
            keys_to_remove = []
            for k, v in list(
                mapping.items()
            ):  # Use list() to avoid dict changed during iteration
                if key in v:
                    v.remove(key)
                if not v:  # Remove empty lists
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                if k in mapping:
                    del mapping[k]


if __name__ == "__main__":
    run_tests()
