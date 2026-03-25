# Owner(s): ["module: dsl-native-ops"]

import sys
from contextlib import contextmanager
from unittest.mock import patch

from torch.backends import __allow_nonbracketed_mutation as allow_nonbracketed_mutation
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTorchBackends(TestCase):
    """Tests for torch.backends.cutedsl and torch.backends.triton modules."""

    def setUp(self):
        """Set up test environment."""
        # Store original module states for cleanup
        self._original_modules = {}
        self._backends_to_test = ["cutedsl", "triton"]

    def tearDown(self):
        """Clean up test environment."""
        # Restore any modules that were cleared during testing
        for module_name, module_obj in self._original_modules.items():
            if module_name not in sys.modules:
                sys.modules[module_name] = module_obj

    def _clear_backend_modules(self, backend_name):
        """Helper to clear backend and utility modules for clean imports."""
        modules_to_clear = [
            f"torch.backends.{backend_name}",
            f"torch._native.{backend_name}_utils",
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                if module_name not in self._original_modules:
                    self._original_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

    @contextmanager
    def _mock_native_utils(
        self, backend_names=None, runtime_available=True, runtime_version=None
    ):
        """Context manager to mock the native utility functions."""
        if backend_names is None:
            backend_names = self._backends_to_test

        patches = []
        for backend in backend_names:
            patches.extend(
                [
                    patch(
                        f"torch._native.{backend}_utils.runtime_available",
                        return_value=runtime_available,
                    ),
                    patch(
                        f"torch._native.{backend}_utils.runtime_version",
                        return_value=runtime_version,
                    ),
                ]
            )

        # Apply all patches using contextlib.ExitStack for proper cleanup
        from contextlib import ExitStack

        with ExitStack() as stack:
            for patch_obj in patches:
                stack.enter_context(patch_obj)
            yield

    def test_backend_module_import(self):
        """Test that torch.backends modules can be imported and have required attributes."""
        required_attrs = ["is_available", "version", "enabled", "flags", "set_flags"]

        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                for attr in required_attrs:
                    with self.subTest(backend=backend_name, attribute=attr):
                        self.assertTrue(hasattr(backend, attr))

    def test_backend_is_available(self):
        """Test torch.backends.*.is_available() function with different availability states."""
        availability_scenarios = [
            (True, "runtime available"),
            (False, "runtime not available"),
        ]

        for backend_name in self._backends_to_test:
            for runtime_available, description in availability_scenarios:
                with self.subTest(backend=backend_name, scenario=description):
                    # Clear modules for clean import
                    self._clear_backend_modules(backend_name)

                    with self._mock_native_utils(
                        [backend_name], runtime_available=runtime_available
                    ):
                        backend = __import__(
                            f"torch.backends.{backend_name}", fromlist=[backend_name]
                        )

                        result = backend.is_available()
                        self.assertIsInstance(result, bool)
                        self.assertEqual(result, runtime_available)

    def test_backend_version(self):
        """Test torch.backends.*.version() function with different version scenarios."""
        from packaging.version import Version

        version_scenarios = [
            (Version("1.2.3"), "version available"),
            (None, "version not available"),
        ]

        for backend_name in self._backends_to_test:
            for test_version, description in version_scenarios:
                with self.subTest(backend=backend_name, scenario=description):
                    # Clear modules for clean import
                    self._clear_backend_modules(backend_name)

                    with self._mock_native_utils(
                        [backend_name], runtime_version=test_version
                    ):
                        backend = __import__(
                            f"torch.backends.{backend_name}", fromlist=[backend_name]
                        )

                        result = backend.version()
                        self.assertEqual(result, test_version)

    def test_backend_enabled_default_and_setter(self):
        """Test that torch.backends.* enabled property defaults to True and can be set."""
        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                # Test default value
                self.assertTrue(
                    backend.enabled, f"{backend_name} should be enabled by default"
                )

                # Get initial state for restoration
                initial_state = backend.enabled

                # Test disabling
                with allow_nonbracketed_mutation():
                    backend.enabled = False
                self.assertFalse(
                    backend.enabled, f"Should be able to disable {backend_name}"
                )

                # Test re-enabling
                with allow_nonbracketed_mutation():
                    backend.enabled = True
                self.assertTrue(
                    backend.enabled, f"Should be able to re-enable {backend_name}"
                )

                # Restore initial state
                with allow_nonbracketed_mutation():
                    backend.enabled = initial_state

    @patch("torch._native.registry.reenable_op_overrides")
    @patch("torch._native.registry.deregister_op_overrides")
    def test_backend_set_flags(self, mock_disable, mock_reenable):
        """Test torch.backends.*.set_flags() function."""
        set_flags_scenarios = [
            (False, "disable backend"),
            (True, "enable backend"),
            (None, "no change"),
        ]

        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                for enabled_value, description in set_flags_scenarios:
                    with self.subTest(backend=backend_name, scenario=description):
                        mock_disable.reset_mock()
                        mock_reenable.reset_mock()

                        # Ensure starting state is enabled
                        with allow_nonbracketed_mutation():
                            backend.enabled = True

                        orig_flags = backend.set_flags(_enabled=enabled_value)
                        self.assertEqual(orig_flags, (True,))  # Originally enabled

                        if enabled_value is False:
                            self.assertFalse(backend.enabled)
                            mock_disable.assert_called_with(
                                disable_dsl_names=backend_name
                            )
                            mock_reenable.assert_not_called()
                        elif enabled_value is True:
                            self.assertTrue(backend.enabled)
                            mock_reenable.assert_called_with(
                                enable_dsl_names=backend_name
                            )
                        else:  # None case
                            self.assertTrue(backend.enabled)  # Should remain unchanged
                            mock_disable.assert_not_called()
                            mock_reenable.assert_not_called()

    def test_backend_flags_context_manager(self):
        """Test torch.backends.*.flags context manager with various scenarios."""
        context_scenarios = [
            ({"enabled": False}, "disable in context"),
            ({}, "no arguments"),
        ]

        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                for flags_kwargs, description in context_scenarios:
                    with self.subTest(backend=backend_name, scenario=description):
                        # Ensure starting state is enabled
                        with allow_nonbracketed_mutation():
                            backend.enabled = True
                        initial_state = backend.enabled

                        # Test context manager
                        with backend.flags(**flags_kwargs):
                            if flags_kwargs.get("enabled") is False:
                                self.assertFalse(backend.enabled)
                            else:
                                self.assertEqual(backend.enabled, initial_state)

                        # Should restore original state
                        self.assertEqual(backend.enabled, initial_state)

    def test_backend_flags_context_manager_exception_handling(self):
        """Test torch.backends.*.flags context manager restores state on exception."""
        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                # Ensure starting state is enabled
                with allow_nonbracketed_mutation():
                    backend.enabled = True
                initial_state = backend.enabled

                # Test exception handling
                with self.assertRaises(ValueError):
                    with backend.flags(enabled=False):
                        self.assertFalse(backend.enabled)
                        raise ValueError("Test exception")

                # Should restore original state even after exception
                self.assertEqual(backend.enabled, initial_state)

    def test_backend_module_replacement(self):
        """Test that torch.backends.* modules are properly replaced with PropModule."""
        for backend_name in self._backends_to_test:
            with self.subTest(backend=backend_name):
                backend = __import__(
                    f"torch.backends.{backend_name}", fromlist=[backend_name]
                )

                # Check that the module is the custom PropModule instance
                from torch.backends import PropModule

                module_key = f"torch.backends.{backend_name}"
                self.assertIsInstance(sys.modules[module_key], PropModule)

                # Test that enabled property works through the module replacement
                self.assertTrue(hasattr(backend, "enabled"))
                self.assertIsInstance(backend.enabled, bool)

    def test_both_backends_independent(self):
        """Test that cutedsl and triton backends operate independently."""
        import torch.backends.cutedsl as cutedsl
        import torch.backends.triton as triton

        # Both should start enabled
        self.assertTrue(cutedsl.enabled)
        self.assertTrue(triton.enabled)

        # Disable cutedsl, triton should remain enabled
        with allow_nonbracketed_mutation():
            cutedsl.enabled = False
        self.assertFalse(cutedsl.enabled)
        self.assertTrue(triton.enabled)

        # Disable triton, cutedsl should remain disabled
        with allow_nonbracketed_mutation():
            triton.enabled = False
        self.assertFalse(cutedsl.enabled)
        self.assertFalse(triton.enabled)

        # Re-enable cutedsl, triton should remain disabled
        with allow_nonbracketed_mutation():
            cutedsl.enabled = True
        self.assertTrue(cutedsl.enabled)
        self.assertFalse(triton.enabled)

    @patch("torch._native.registry.reenable_op_overrides")
    @patch("torch._native.registry.deregister_op_overrides")
    def test_nested_context_managers(self, mock_disable, mock_reenable):
        """Test nested context managers for both backends."""
        import torch.backends.cutedsl as cutedsl
        import torch.backends.triton as triton

        # Ensure both start enabled
        with allow_nonbracketed_mutation():
            cutedsl.enabled = True
            triton.enabled = True

        self.assertTrue(cutedsl.enabled)
        self.assertTrue(triton.enabled)

        with cutedsl.flags(enabled=False):
            self.assertFalse(cutedsl.enabled)
            self.assertTrue(triton.enabled)

            with triton.flags(enabled=False):
                self.assertFalse(cutedsl.enabled)
                self.assertFalse(triton.enabled)

            # triton restored, cutedsl still disabled
            self.assertFalse(cutedsl.enabled)
            self.assertTrue(triton.enabled)

        # Both restored to original state
        self.assertTrue(cutedsl.enabled)
        self.assertTrue(triton.enabled)

    def test_integration_with_registry_calls(self):
        """Test that backends correctly integrate with the registry system."""
        # Test that the registry functions are properly imported and accessible

        # These should not raise import errors
        from torch._native.registry import (
            deregister_op_overrides,
            reenable_op_overrides,
        )

        # The backends should be able to call these functions without error
        # (though they may be no-ops if no overrides are registered)
        try:
            deregister_op_overrides(disable_dsl_names="cutedsl")
            reenable_op_overrides(enable_dsl_names="cutedsl")
            deregister_op_overrides(disable_dsl_names="triton")
            reenable_op_overrides(enable_dsl_names="triton")
        except Exception as e:
            self.fail(f"Registry integration failed: {e}")


if __name__ == "__main__":
    run_tests()
