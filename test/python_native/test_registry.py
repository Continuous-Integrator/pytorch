# Owner(s): ["module: dsl-native-ops"]

"""
Test suite for the torch._native.registry module.

This test suite provides comprehensive coverage for the PyTorch native DSL registration system,
organized into focused test classes for better maintainability.
"""

from unittest.mock import MagicMock, patch

import torch._native.registry as registry_module
import torch.library
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class RegistryTestMixin:
    """Common setup and utilities for registry tests."""

    def setUp(self):
        """Set up clean registry state for each test."""
        self.registry = registry_module
        # Store original state for cleanup
        self._original_graphs = dict(self.registry._graphs)
        self._original_libs = dict(self.registry._libs)
        self._original_filter_state = self.registry._filter_state

        # Clear registries for clean test environment
        self.registry._graphs.clear()
        self.registry._libs.clear()
        self.registry._filter_state = registry_module._FilterState()

    def tearDown(self):
        """Restore original registry state after each test."""
        self.registry._graphs.clear()
        self.registry._graphs.update(self._original_graphs)
        self.registry._libs.clear()
        self.registry._libs.update(self._original_libs)
        self.registry._filter_state = self._original_filter_state

    def _cleanup_test_registration(self, key):
        """Clean up test registration to prevent interference."""
        if key in self.registry._graphs:
            del self.registry._graphs[key]
        if key in self.registry._libs:
            del self.registry._libs[key]


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestRegistryBasics(RegistryTestMixin, TestCase):
    """Test basic registry operations and data structures."""

    def test_override_node_dataclass(self):
        """Test _OverrideNode dataclass creation and attributes."""

        def test_fn(x):
            return x

        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)

        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.override_fn, test_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

        # Test with custom parameters
        def override_fn(x, y):
            return x + y

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
        """Test that _get_or_create_library caches Library instances."""
        key = ("test_caching.Tensor", "CPU")

        lib1 = self.registry._get_or_create_library(*key)
        self.assertIsInstance(lib1, torch.library.Library)

        lib2 = self.registry._get_or_create_library(*key)
        self.assertIs(lib1, lib2, "Should return cached instance")

        # Different dispatch key should create different Library
        key2 = ("test_caching.Tensor", "CUDA")
        lib3 = self.registry._get_or_create_library(*key2)
        self.assertIsNot(lib1, lib3)

        # Cleanup
        self._cleanup_test_registration(key)
        self._cleanup_test_registration(key2)

    def test_resolve_iterable(self):
        """Test _resolve_iterable handles various input types correctly."""
        # Test None input
        result = list(self.registry._resolve_iterable(None))
        self.assertEqual(result, [])

        # Test string input
        result = list(self.registry._resolve_iterable("single"))
        self.assertEqual(result, ["single"])

        # Test iterable input
        input_list = ["a", "b", "c"]
        result = list(self.registry._resolve_iterable(input_list))
        self.assertEqual(result, input_list)

    def test_update_registration_maps(self):
        """Test _update_registration_maps correctly updates mapping dictionaries."""
        key = ("test_op.Tensor", "CPU")

        # Clear any existing mappings
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()

        self.registry._update_registration_maps(
            "test_dsl", "test_op.Tensor", "CPU", key
        )

        # Verify mappings were created
        self.assertIn("test_dsl", self.registry._dsl_name_to_lib_graph)
        self.assertIn("test_op.Tensor", self.registry._op_symbol_to_lib_graph)
        self.assertIn("CPU", self.registry._dispatch_key_to_lib_graph)

        # Verify the key is in each mapping
        self.assertIn(key, self.registry._dsl_name_to_lib_graph["test_dsl"])
        self.assertIn(key, self.registry._op_symbol_to_lib_graph["test_op.Tensor"])
        self.assertIn(key, self.registry._dispatch_key_to_lib_graph["CPU"])

    def test__build_key_set(self):
        """Test _build_key_set correctly builds key sets from mapping dicts."""
        # Setup test mappings
        key1 = ("op1.Tensor", "CPU")
        key2 = ("op2.Tensor", "CPU")

        self.registry._dsl_name_to_lib_graph = {"test_dsl": [key1, key2]}
        self.registry._op_symbol_to_lib_graph = {"op1.Tensor": [key1]}

        # Test building key set from DSL names
        result = self.registry._build_key_set(["test_dsl"], None, None)
        self.assertEqual(result, {key1, key2})

        # Test building key set from op symbols
        result = self.registry._build_key_set(None, ["op1.Tensor"], None)
        self.assertEqual(result, {key1})

        # Test combining multiple criteria
        result = self.registry._build_key_set(["test_dsl"], ["op1.Tensor"], None)
        self.assertEqual(result, {key1, key2})


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestRegistration(RegistryTestMixin, TestCase):
    """Test registration and deregistration functionality."""

    @patch("torch.library.Library")
    def test_register_op_override_variants(self, mock_library_cls):
        """Test register_op_override with various parameter combinations."""

        def test_fn(x):
            return x

        # Test basic registration
        self.registry.register_op_override(
            "test_backend", "aten", "add.Tensor", "CPU", test_fn
        )

        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 1)

        node = self.registry._graphs[key][0]
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.override_fn, test_fn)

        # Test unconditional override
        self.registry.register_op_override(
            "test_backend2",
            "aten",
            "add.Tensor",
            "CPU",
            test_fn,
            unconditional_override=True,
        )

        self.assertEqual(len(self.registry._graphs[key]), 2)
        self.assertTrue(self.registry._graphs[key][1].unconditional_override)

        # Cleanup
        self._cleanup_test_registration(key)

    def test_invalid_lib_symbol_raises_error(self):
        """Test that invalid lib_symbol raises appropriate error."""

        def test_fn(x):
            return x

        with self.assertRaises(ValueError) as cm:
            self.registry.register_op_override(
                "test_backend", "invalid_lib", "add.Tensor", "CPU", test_fn
            )

        self.assertIn('Unsupported lib_symbol (must be "aten"', str(cm.exception))

    @patch("torch.library.Library")
    def testderegister_op_overrides(self, mock_library_cls):
        """Test deregister_op_overrides functionality."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def test_fn(x):
            return x

        key = ("test_deregister.Tensor", "CPU")

        # Register an override
        self.registry.register_op_override(
            "test_backend", "aten", "test_deregister.Tensor", "CPU", test_fn
        )

        # Verify it was registered
        self.assertEqual(len(self.registry._graphs[key]), 1)
        node = self.registry._graphs[key][0]
        self.assertEqual(node.dsl_name, "test_backend")

        # Deregister by DSL name
        self.registry.deregister_op_overrides(disable_dsl_names="test_backend")

        # Verify the graph still exists but the node is marked inactive
        self.assertEqual(len(self.registry._graphs[key]), 1)
        self.assertFalse(self.registry._graphs[key][0].active)

        # Cleanup
        self._cleanup_test_registration(key)

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

        # Cleanup
        self._cleanup_test_registration(key)

    def _setup_override_chain(
        self, backends, op_symbol, dispatch_key, mock_library_cls
    ):
        """Helper method to set up a chain of overrides for testing."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        impl_fns = []
        for i, backend in enumerate(backends):

            def make_impl_fn(b, index):
                def impl_fn(dispatch_keys, x, y):
                    return (f"{b}_{index}", x, y)

                return impl_fn

            impl_fn = make_impl_fn(backend, i)
            impl_fns.append(impl_fn)

            self.registry.register_op_override(
                backend,
                "aten",
                op_symbol,
                dispatch_key,
                impl_fn,
                allow_multiple_override=True,
            )

        return mock_lib, impl_fns

    @patch("torch.library.Library")
    def test_override_chain_operations(self, mock_library_cls):
        """Test override chain creation and various removal scenarios."""
        # Test 1: Basic chain creation
        backends = ["backend1", "backend2", "backend3"]
        op_symbol = "mul.Tensor"
        dispatch_key = "CUDA"
        key = (op_symbol, dispatch_key)

        mock_lib, impl_fns = self._setup_override_chain(
            backends, op_symbol, dispatch_key, mock_library_cls
        )

        self.assertEqual(len(self.registry._graphs[key]), 3)
        nodes = self.registry._graphs[key]
        for i, backend in enumerate(backends):
            self.assertEqual(nodes[i].dsl_name, backend)
        self.assertTrue(all(node.active for node in nodes))

        # Test 2: Remove middle override
        self.registry.deregister_op_overrides(disable_dsl_names="backend2")
        nodes = self.registry._graphs[key]
        backend1_node = next(n for n in nodes if n.dsl_name == "backend1")
        backend2_node = next(n for n in nodes if n.dsl_name == "backend2")
        backend3_node = next(n for n in nodes if n.dsl_name == "backend3")

        self.assertTrue(backend1_node.active)
        self.assertFalse(backend2_node.active)
        self.assertTrue(backend3_node.active)

        # Test 3: Multiple removal (setup fresh chain)
        backends_multi = ["backend1", "backend2", "backend3", "backend4", "backend5"]
        op_symbol_multi = "div.Tensor"
        key_multi = (op_symbol_multi, dispatch_key)

        mock_lib2, impl_fns2 = self._setup_override_chain(
            backends_multi, op_symbol_multi, dispatch_key, mock_library_cls
        )

        self.registry.deregister_op_overrides(
            disable_dsl_names=["backend2", "backend4"]
        )
        nodes_multi = self.registry._graphs[key_multi]

        expected_active = {"backend1", "backend3", "backend5"}
        expected_inactive = {"backend2", "backend4"}

        for node in nodes_multi:
            if node.dsl_name in expected_active:
                self.assertTrue(node.active, f"{node.dsl_name} should be active")
            elif node.dsl_name in expected_inactive:
                self.assertFalse(node.active, f"{node.dsl_name} should be inactive")

        # Test 4: Remove all (using a fresh op to avoid interference)
        op_symbol_all = "sub.Tensor"
        key_all = (op_symbol_all, dispatch_key)
        backends_all = ["backend1", "backend2", "backend3"]

        mock_lib3, impl_fns3 = self._setup_override_chain(
            backends_all, op_symbol_all, dispatch_key, mock_library_cls
        )

        self.registry.deregister_op_overrides(disable_dsl_names=backends_all)
        nodes_all = self.registry._graphs[key_all]
        self.assertTrue(all(not node.active for node in nodes_all))

        # Cleanup
        self._cleanup_test_registration(key)
        self._cleanup_test_registration(key_multi)
        self._cleanup_test_registration(key_all)

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

        # Cleanup
        self._cleanup_test_registration(mul_key)
        self._cleanup_test_registration(add_key)

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

        # Cleanup
        self._cleanup_test_registration(mul_cuda_key)
        self._cleanup_test_registration(add_cuda_key)
        self._cleanup_test_registration(mul_cpu_key)
        self._cleanup_test_registration(add_cpu_key)

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

        # Cleanup all registered combinations
        for op in ops:
            for dispatch_key in dispatch_keys:
                key = (op, dispatch_key)
                self._cleanup_test_registration(key)


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestFilterState(RegistryTestMixin, TestCase):
    """Test filter state management functionality."""

    def test_filter_state_initialization_and_check_enabled(self):
        """Test FilterState initialization and check_enabled method."""

        def test_fn(x):
            return x

        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)

        # Initially, all nodes should be enabled
        filter_state = self.registry._FilterState()
        self.assertTrue(filter_state.check_enabled(node))

        # Add dsl_name to filter - node should be disabled
        filter_state.update(["test_dsl"], None, None)
        self.assertFalse(filter_state.check_enabled(node))

        # Remove from filter - node should be enabled again
        filter_state.update(["test_dsl"], None, None, remove_keys=True)
        self.assertTrue(filter_state.check_enabled(node))

    def test_filter_state_update_operations(self):
        """Test FilterState update method with various inputs."""
        filter_state = self.registry._FilterState()

        # Test adding single values
        filter_state.update("dsl1", "op1", "key1")
        self.assertIn("dsl1", filter_state._dsl_names)
        self.assertIn("op1", filter_state._op_symbols)
        self.assertIn("key1", filter_state._dispatch_keys)

        # Test adding multiple values
        filter_state.update(["dsl2", "dsl3"], ["op2"], None)
        self.assertEqual(len(filter_state._dsl_names), 3)
        self.assertEqual(len(filter_state._op_symbols), 2)

        # Test removing values
        filter_state.update(["dsl1"], None, None, remove_keys=True)
        self.assertNotIn("dsl1", filter_state._dsl_names)
        self.assertEqual(len(filter_state._dsl_names), 2)

    def test_filter_state_miscellaneous(self):
        """Test miscellaneous FilterState functionality."""
        filter_state = self.registry._FilterState()

        # Test string representation
        filter_state.update(["dsl1", "dsl2"], ["op1"], ["CPU"])
        str_repr = str(filter_state)
        self.assertIn("Filter State:", str_repr)
        self.assertIn("dsl1", str_repr)
        self.assertIn("op1", str_repr)
        self.assertIn("CPU", str_repr)

        # Test build_disable_key_set functionality
        # First set up some mappings
        key1 = ("op1.Tensor", "CPU")
        self.registry._dsl_name_to_lib_graph = {"dsl1": [key1]}

        key_set = filter_state.build_disable_key_set()
        self.assertIn(key1, key_set)


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestGraphReordering(RegistryTestMixin, TestCase):
    """Test graph reordering functionality."""

    def test_reorder_graphs_basic_reordering(self):
        """Test basic graph reordering functionality."""
        # Set up test data
        key = ("test_reorder.Tensor", "CPU")

        def impl_fn(x):
            return x

        # Create nodes in specific order
        nodes = [
            self.registry._OverrideNode("dsl_a", "test_reorder.Tensor", "CPU", impl_fn),
            self.registry._OverrideNode("dsl_b", "test_reorder.Tensor", "CPU", impl_fn),
            self.registry._OverrideNode("dsl_c", "test_reorder.Tensor", "CPU", impl_fn),
        ]
        self.registry._graphs[key] = nodes

        # Define reverse ordering function
        def reverse_order(op_symbol, dispatch_key, graph):
            return list(reversed(graph))

        # Apply reordering
        self.registry.reorder_graphs_from_user_function(reverse_order)

        # Verify order was reversed
        reordered_graph = self.registry._graphs[key]
        expected_names = ["dsl_c", "dsl_b", "dsl_a"]
        actual_names = [node.dsl_name for node in reordered_graph]
        self.assertEqual(actual_names, expected_names)

        # Clean up
        self._cleanup_test_registration(key)

    def test_reorder_graphs_no_reregistration_by_default(self):
        """Test that reordering doesn't trigger reregistration by default."""
        # Set up test data
        key = ("test_no_rereg.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_no_rereg.Tensor", "CPU", impl_fn
        )
        self.registry._graphs[key] = [node]

        # Track whether library was created (would indicate reregistration)
        original_libs_count = len(self.registry._libs)

        # Define ordering function that modifies the graph
        def modify_order(op_symbol, dispatch_key, graph):
            modified_node = self.registry._OverrideNode(
                "modified_dsl", op_symbol, dispatch_key, impl_fn
            )
            return [modified_node]

        # Apply reordering without explicit reregistration
        self.registry.reorder_graphs_from_user_function(modify_order)

        # Verify library count didn't change (no reregistration)
        self.assertEqual(len(self.registry._libs), original_libs_count)

        # But verify the graph was still modified
        self.assertEqual(self.registry._graphs[key][0].dsl_name, "modified_dsl")

        # Clean up
        self._cleanup_test_registration(key)

    @patch("torch.library.Library")
    def test_reorder_graphs_with_reregistration(self, mock_library_cls):
        """Test graph reordering with library reregistration."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Set up test data
        key = ("test_reregister.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_reregister.Tensor", "CPU", impl_fn
        )
        self.registry._graphs[key] = [node]

        # Define ordering function that modifies the graph
        def modify_order(op_symbol, dispatch_key, graph):
            modified_graph = graph.copy()
            modified_graph[0] = self.registry._OverrideNode(
                "modified_dsl", op_symbol, dispatch_key, impl_fn
            )
            return modified_graph

        # Apply reordering with reregistration
        self.registry.reorder_graphs_from_user_function(
            modify_order, reregister_overrides=True
        )

        # Verify the graph was modified
        modified_graph = self.registry._graphs[key]
        self.assertEqual(modified_graph[0].dsl_name, "modified_dsl")

        # Verify library operations were called (indicating reregistration)
        mock_library_cls.assert_called()

        # Clean up
        self._cleanup_test_registration(key)

    def test_reorder_graphs_by_dsl_name_alphabetical(self):
        """Test reordering graphs alphabetically by DSL name."""
        # Set up test data
        key = ("test_alphabetical.Tensor", "CPU")

        def impl_fn(x):
            return x

        nodes = [
            self.registry._OverrideNode(
                "zebra_dsl", "test_alphabetical.Tensor", "CPU", impl_fn
            ),
            self.registry._OverrideNode(
                "alpha_dsl", "test_alphabetical.Tensor", "CPU", impl_fn
            ),
            self.registry._OverrideNode(
                "beta_dsl", "test_alphabetical.Tensor", "CPU", impl_fn
            ),
        ]
        self.registry._graphs[key] = nodes

        # Define alphabetical ordering function
        def alphabetical_order(op_symbol, dispatch_key, graph):
            return sorted(graph, key=lambda n: n.dsl_name)

        # Apply reordering
        self.registry.reorder_graphs_from_user_function(alphabetical_order)

        # Verify alphabetical order
        reordered_graph = self.registry._graphs[key]
        expected_names = ["alpha_dsl", "beta_dsl", "zebra_dsl"]
        actual_names = [node.dsl_name for node in reordered_graph]
        self.assertEqual(actual_names, expected_names)

        # Clean up
        self._cleanup_test_registration(key)


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestErrorHandling(RegistryTestMixin, TestCase):
    """Test error handling and edge cases."""

    def test_reorder_graphs_ordering_function_raises_exception(self):
        """Test that exceptions in ordering functions are caught and logged."""
        # Set up test data
        key = ("test_exception.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_exception.Tensor", "CPU", impl_fn
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # Define ordering function that raises an exception
        def failing_order_fn(op_symbol, dispatch_key, graph):
            raise ValueError("Test exception in ordering function")

        # The function should log a warning and preserve the original graph
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function(failing_order_fn)

        # Verify warning was logged with the exception message
        self.assertEqual(len(log.records), 1)
        log_message = log.records[0].getMessage()
        self.assertIn(
            "Graph transformation failed for test_exception.Tensor/CPU", log_message
        )
        # Exception details are logged via exc_info, not in the message itself
        self.assertTrue(log.records[0].exc_info is not None)

        # Verify original graph is preserved
        self.assertEqual(self.registry._graphs[key], original_graph)

        # Clean up
        self._cleanup_test_registration(key)

    def test_reorder_graphs_non_callable_ordering_function(self):
        """Test that non-callable ordering function is caught and logged."""
        # Set up test data - need graphs for the function to be called
        key = ("test_non_callable.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_non_callable.Tensor", "CPU", impl_fn
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # The function should log a warning when it tries to call the non-callable
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function("not_callable")

        # Verify warning was logged about the TypeError
        self.assertEqual(len(log.records), 1)
        log_message = log.records[0].getMessage()
        self.assertIn(
            "Graph transformation failed for test_non_callable.Tensor/CPU", log_message
        )
        # Exception details are logged via exc_info, not in the message itself
        self.assertTrue(log.records[0].exc_info is not None)

        # Verify original graph is preserved
        self.assertEqual(self.registry._graphs[key], original_graph)

        # Clean up
        self._cleanup_test_registration(key)

    def test_reorder_graphs_ordering_function_returns_none(self):
        """Test handling when ordering function returns None."""
        # Set up test data
        key = ("test_none_return.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_none_return.Tensor", "CPU", impl_fn
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # Define ordering function that returns None
        def none_return_fn(op_symbol, dispatch_key, graph):
            return None

        # The function should log a warning and preserve the original graph
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function(none_return_fn)

        # Verify warning was logged about invalid return type
        self.assertEqual(len(log.records), 1)
        log_message = log.records[0].getMessage()
        self.assertIn(
            "Graph transformation returned invalid type NoneType", log_message
        )
        self.assertIn("test_none_return.Tensor/CPU", log_message)
        self.assertIn("Expected list", log_message)

        # Verify original graph is preserved (not None)
        self.assertEqual(self.registry._graphs[key], original_graph)

        # Clean up
        self._cleanup_test_registration(key)

    def test_reorder_graphs_ordering_function_returns_wrong_type(self):
        """Test handling when ordering function returns incorrect type."""
        # Set up test data
        key = ("test_wrong_type.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_wrong_type.Tensor", "CPU", impl_fn
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # Define ordering function that returns wrong type
        def wrong_type_fn(op_symbol, dispatch_key, graph):
            return "not_a_list"

        # The function should log a warning and preserve the original graph
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function(wrong_type_fn)

        # Verify warning was logged about invalid return type
        self.assertEqual(len(log.records), 1)
        log_message = log.records[0].getMessage()
        self.assertIn("Graph transformation returned invalid type str", log_message)
        self.assertIn("test_wrong_type.Tensor/CPU", log_message)
        self.assertIn("Expected list", log_message)

        # Verify original graph is preserved (not the invalid return value)
        self.assertEqual(self.registry._graphs[key], original_graph)

        # Clean up
        self._cleanup_test_registration(key)


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestEnvironmentVariables(RegistryTestMixin, TestCase):
    """Test environment variable integration."""

    def test_get_user_ordering_fn_env_var_not_set(self):
        """Test behavior when environment variable is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear the cache to ensure fresh evaluation
            from torch._native import get_user_ordering_fn

            get_user_ordering_fn.cache_clear()

            result = get_user_ordering_fn()
            self.assertIsNone(result)

    def test_get_user_ordering_fn_invalid_module_path(self):
        """Test handling of invalid module paths."""
        with patch.dict(
            "os.environ",
            {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": "nonexistent.module.function"},
        ):
            from torch._native import get_user_ordering_fn

            # Clear the cache to ensure fresh evaluation
            get_user_ordering_fn.cache_clear()

            with self.assertRaises(ValueError) as cm:
                get_user_ordering_fn()
            self.assertIn("Could not resolve", str(cm.exception))

    def test_get_user_ordering_fn_invalid_function_name(self):
        """Test handling of invalid function name in valid module."""
        with patch.dict(
            "os.environ",
            {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": "os.nonexistent_function"},
        ):
            from torch._native import get_user_ordering_fn

            # Clear the cache to ensure fresh evaluation
            get_user_ordering_fn.cache_clear()

            with self.assertRaises(ValueError) as cm:
                get_user_ordering_fn()
            self.assertIn("Could not resolve", str(cm.exception))

    def test_get_user_ordering_fn_function_not_callable(self):
        """Test handling when resolved object is not callable."""
        with patch.dict(
            "os.environ", {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": "os.name"}
        ):  # os.name is not callable
            from torch._native import get_user_ordering_fn

            # Clear the cache to ensure fresh evaluation
            get_user_ordering_fn.cache_clear()

            with self.assertRaises(ValueError) as cm:
                get_user_ordering_fn()
            self.assertIn("Could not resolve", str(cm.exception))

    def test_get_user_ordering_fn_valid_function_integration(self):
        """Test successful resolution of valid function."""

        # Create a test ordering function
        def test_reverse_ordering(op_symbol, dispatch_key, graph):
            return list(reversed(graph))

        # Use a valid module path (this module itself)
        module_path = f"{__name__}.test_reverse_ordering"

        with patch.dict(
            "os.environ", {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": module_path}
        ):
            from torch._native import get_user_ordering_fn

            # Clear the cache to ensure fresh evaluation
            get_user_ordering_fn.cache_clear()

            # Make the function available in this module's namespace
            globals()["test_reverse_ordering"] = test_reverse_ordering

            try:
                result = get_user_ordering_fn()
                self.assertIsNotNone(result)
                self.assertTrue(callable(result))
                # Test that the function works as expected
                test_graph = [1, 2, 3]
                reversed_graph = result("test_op", "CPU", test_graph)
                self.assertEqual(reversed_graph, [3, 2, 1])
            finally:
                # Clean up
                if "test_reverse_ordering" in globals():
                    del globals()["test_reverse_ordering"]


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestHelperFunctions(RegistryTestMixin, TestCase):
    """Test helper function functionality."""

    def test_apply_graph_filter_basic_filtering(self):
        """Test _apply_graph_filter removes nodes based on filter function."""
        # Set up test data
        key = ("test_filter.Tensor", "CPU")

        def impl_fn(x):
            return x

        nodes = [
            self.registry._OverrideNode(
                "keep_dsl", "test_filter.Tensor", "CPU", impl_fn
            ),
            self.registry._OverrideNode(
                "remove_dsl", "test_filter.Tensor", "CPU", impl_fn
            ),
            self.registry._OverrideNode(
                "keep_another", "test_filter.Tensor", "CPU", impl_fn
            ),
        ]
        self.registry._graphs[key] = nodes

        # Apply filter to remove nodes with "remove" in the name
        self.registry._apply_graph_filter(
            lambda op, dk, node: "remove" not in node.dsl_name,
            reregister_overrides=False,
        )

        # Verify filtering worked
        filtered_graph = self.registry._graphs[key]
        remaining_names = {node.dsl_name for node in filtered_graph}
        expected_names = {"keep_dsl", "keep_another"}
        self.assertEqual(remaining_names, expected_names)

        # Clean up
        self._cleanup_test_registration(key)

    def test_apply_selective_reordering_condition_matching(self):
        """Test _apply_selective_reordering only affects matching conditions."""
        # Set up test data for two different operations
        key1 = ("test_selective1.Tensor", "CPU")
        key2 = ("test_selective2.Tensor", "CUDA")

        def impl_fn(x):
            return x

        nodes1 = [
            self.registry._OverrideNode(
                "dsl_c", "test_selective1.Tensor", "CPU", impl_fn
            ),
            self.registry._OverrideNode(
                "dsl_a", "test_selective1.Tensor", "CPU", impl_fn
            ),
        ]
        nodes2 = [
            self.registry._OverrideNode(
                "dsl_z", "test_selective2.Tensor", "CUDA", impl_fn
            ),
            self.registry._OverrideNode(
                "dsl_b", "test_selective2.Tensor", "CUDA", impl_fn
            ),
        ]
        self.registry._graphs[key1] = nodes1
        self.registry._graphs[key2] = nodes2

        # Apply selective reordering - only reorder CPU operations
        self.registry._apply_selective_reordering(
            condition_fn=lambda op, dk: dk == "CPU",
            ordering_fn=lambda op, dk, g: sorted(g, key=lambda n: n.dsl_name),
            reregister_overrides=False,
        )

        # Verify CPU operation was reordered (alphabetically)
        cpu_graph = self.registry._graphs[key1]
        cpu_names = [node.dsl_name for node in cpu_graph]
        self.assertEqual(cpu_names, ["dsl_a", "dsl_c"])

        # Verify CUDA operation was NOT reordered (preserved original order)
        cuda_graph = self.registry._graphs[key2]
        cuda_names = [node.dsl_name for node in cuda_graph]
        self.assertEqual(cuda_names, ["dsl_z", "dsl_b"])

        # Clean up
        self._cleanup_test_registration(key1)
        self._cleanup_test_registration(key2)

    def test_key_set_building_functions(self):
        """Test key set building functions for filter operations."""
        # Set up test mappings
        key1 = ("add.Tensor", "CPU")
        key2 = ("mul.Tensor", "CUDA")
        key3 = ("div.Tensor", "CPU")

        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1, key2]
        self.registry._op_symbol_to_lib_graph["add.Tensor"] = [key1]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1, key3]

        # Test _build_key_set with single criteria
        key_set = self.registry._build_key_set("dsl1", None, None)
        self.assertEqual(key_set, {key1, key2})

        # Test _build_key_set with multiple criteria (union)
        key_set = self.registry._build_key_set("dsl1", "add.Tensor", "CPU")
        expected_keys = {key1, key2, key3}  # Union of all matching keys
        self.assertEqual(key_set, expected_keys)

        # Test FilterState.build_disable_key_set
        filter_state = self.registry._FilterState()
        filter_state._dsl_names.add("dsl1")
        filter_state._dispatch_keys.add("CPU")

        key_set = filter_state.build_disable_key_set()
        # Should include keys from both dsl1 and CPU filters
        expected_keys = {key1, key2, key3}
        self.assertEqual(key_set, expected_keys)

    @patch("torch.library.Library")
    def test_reenable_op_overrides_scenarios(self, mock_library_cls):
        """Test reenable_op_overrides with different scenarios."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def test_fn(x):
            return x

        # Test 1: Basic single re-enable
        key1 = ("add.Tensor", "CPU")
        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn, active=False
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "add.Tensor", "CPU", test_fn, active=True
        )

        self.registry._graphs[key1] = [node1, node2]
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1]
        self.registry._filter_state._dsl_names.add("dsl1")
        self.registry._libs[key1] = mock_lib

        self.registry.reenable_op_overrides(enable_dsl_names="dsl1")

        self.assertNotIn("dsl1", self.registry._filter_state._dsl_names)
        self.assertTrue(node1.active)
        self.assertTrue(node2.active)
        self.assertEqual(mock_lib.impl.call_count, 2)

        # Test 2: Multiple criteria re-enable
        mock_lib.reset_mock()
        key2 = ("mul.Tensor", "CUDA")
        node3 = self.registry._OverrideNode(
            "dsl3", "mul.Tensor", "CUDA", test_fn, active=False
        )

        self.registry._graphs[key2] = [node3]
        self.registry._op_symbol_to_lib_graph["mul.Tensor"] = [key2]
        self.registry._filter_state._op_symbols.add("mul.Tensor")

        self.registry.reenable_op_overrides(enable_op_symbols="mul.Tensor")

        self.assertNotIn("mul.Tensor", self.registry._filter_state._op_symbols)
        self.assertTrue(node3.active)

        # Test 3: Partial re-enable with remaining filters
        key3 = ("div.Tensor", "CPU")
        node4 = self.registry._OverrideNode(
            "filtered_dsl", "div.Tensor", "CPU", test_fn, active=False
        )
        node5 = self.registry._OverrideNode(
            "allowed_dsl", "div.Tensor", "CPU", test_fn, active=False
        )

        self.registry._graphs[key3] = [node4, node5]
        self.registry._dsl_name_to_lib_graph["allowed_dsl"] = [key3]
        self.registry._filter_state._dsl_names.update(["filtered_dsl", "allowed_dsl"])

        mock_lib_new = MagicMock()
        mock_library_cls.return_value = mock_lib_new
        self.registry._libs[key3] = mock_lib_new

        self.registry.reenable_op_overrides(enable_dsl_names="allowed_dsl")

        self.assertIn("filtered_dsl", self.registry._filter_state._dsl_names)
        self.assertNotIn("allowed_dsl", self.registry._filter_state._dsl_names)
        self.assertFalse(node4.active)  # Still filtered
        self.assertTrue(node5.active)  # Re-enabled

        mock_lib_new.impl.assert_called_once()

        # Cleanup
        self._cleanup_test_registration(key1)
        self._cleanup_test_registration(key2)
        self._cleanup_test_registration(key3)


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestIntegration(RegistryTestMixin, TestCase):
    """Test comprehensive integration scenarios."""

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


if __name__ == "__main__":
    run_tests()
