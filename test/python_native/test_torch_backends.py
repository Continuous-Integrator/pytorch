# Owner(s): ["module: dsl-native-ops"]

import sys
from contextlib import contextmanager
from unittest.mock import patch

from torch.testing._internal.common_utils import run_tests, TestCase


class TestTorchBackends(TestCase):
    """Tests for torch.backends.cutedsl and torch.backends.triton modules."""

    def setUp(self):
        """Set up test environment."""
        # Note: Don't clear modules as it interferes with the PropModule replacement

    @contextmanager
    def _mock_native_utils(self, runtime_available=True, runtime_version=None):
        """Context manager to mock the native utility functions."""
        with (
            patch(
                "torch._native.cutedsl_utils.runtime_available",
                return_value=runtime_available,
            ),
            patch(
                "torch._native.cutedsl_utils.runtime_version",
                return_value=runtime_version,
            ),
            patch(
                "torch._native.triton_utils.runtime_available",
                return_value=runtime_available,
            ),
            patch(
                "torch._native.triton_utils.runtime_version",
                return_value=runtime_version,
            ),
        ):
            yield

    def test_cutedsl_module_import(self):
        """Test that torch.backends.cutedsl can be imported."""
        import torch.backends.cutedsl as cutedsl

        self.assertTrue(hasattr(cutedsl, "is_available"))
        self.assertTrue(hasattr(cutedsl, "version"))
        self.assertTrue(hasattr(cutedsl, "enabled"))
        self.assertTrue(hasattr(cutedsl, "flags"))
        self.assertTrue(hasattr(cutedsl, "set_flags"))

    def test_triton_module_import(self):
        """Test that torch.backends.triton can be imported."""
        import torch.backends.triton as triton

        self.assertTrue(hasattr(triton, "is_available"))
        self.assertTrue(hasattr(triton, "version"))
        self.assertTrue(hasattr(triton, "enabled"))
        self.assertTrue(hasattr(triton, "flags"))
        self.assertTrue(hasattr(triton, "set_flags"))

    def test_cutedsl_is_available(self):
        """Test torch.backends.cutedsl.is_available() function."""
        with self._mock_native_utils(runtime_available=True):
            import torch.backends.cutedsl as cutedsl

            result = cutedsl.is_available()
            self.assertIsInstance(result, bool)
            self.assertTrue(result)

        # Test with runtime not available
        with self._mock_native_utils(runtime_available=False):
            if "torch.backends.cutedsl" in sys.modules:
                del sys.modules["torch.backends.cutedsl"]
            import torch.backends.cutedsl as cutedsl

            result = cutedsl.is_available()
            self.assertIsInstance(result, bool)
            self.assertFalse(result)

    def test_triton_is_available(self):
        """Test torch.backends.triton.is_available() function."""
        with self._mock_native_utils(runtime_available=True):
            import torch.backends.triton as triton

            result = triton.is_available()
            self.assertIsInstance(result, bool)
            self.assertTrue(result)

        # Test with runtime not available
        with self._mock_native_utils(runtime_available=False):
            if "torch.backends.triton" in sys.modules:
                del sys.modules["torch.backends.triton"]
            import torch.backends.triton as triton

            result = triton.is_available()
            self.assertIsInstance(result, bool)
            self.assertFalse(result)

    def test_cutedsl_version(self):
        """Test torch.backends.cutedsl.version() function."""
        from packaging.version import Version

        test_version = Version("1.2.3")
        with self._mock_native_utils(runtime_version=test_version):
            import torch.backends.cutedsl as cutedsl

            result = cutedsl.version()
            self.assertEqual(result, test_version)

        with self._mock_native_utils(runtime_version=None):
            if "torch.backends.cutedsl" in sys.modules:
                del sys.modules["torch.backends.cutedsl"]
            import torch.backends.cutedsl as cutedsl

            result = cutedsl.version()
            self.assertIsNone(result)

    def test_triton_version(self):
        """Test torch.backends.triton.version() function."""
        from packaging.version import Version

        test_version = Version("2.1.0")
        with self._mock_native_utils(runtime_version=test_version):
            import torch.backends.triton as triton

            result = triton.version()
            self.assertEqual(result, test_version)

        with self._mock_native_utils(runtime_version=None):
            if "torch.backends.triton" in sys.modules:
                del sys.modules["torch.backends.triton"]
            import torch.backends.triton as triton

            result = triton.version()
            self.assertIsNone(result)

    def test_cutedsl_enabled_default(self):
        """Test that torch.backends.cutedsl.enabled is True by default."""
        import torch.backends.cutedsl as cutedsl

        self.assertTrue(cutedsl.enabled)

    def test_triton_enabled_default(self):
        """Test that torch.backends.triton.enabled is True by default."""
        import torch.backends.triton as triton

        self.assertTrue(triton.enabled)

    def test_cutedsl_enabled_setter(self):
        """Test setting torch.backends.cutedsl.enabled."""
        import torch.backends.cutedsl as cutedsl

        # Get initial state
        initial_state = cutedsl.enabled

        # Test disabling
        cutedsl.enabled = False
        self.assertFalse(cutedsl.enabled)

        # Test re-enabling
        cutedsl.enabled = True
        self.assertTrue(cutedsl.enabled)

        # Restore initial state
        cutedsl.enabled = initial_state

    def test_triton_enabled_setter(self):
        """Test setting torch.backends.triton.enabled."""
        import torch.backends.triton as triton

        # Get initial state
        initial_state = triton.enabled

        # Test disabling
        triton.enabled = False
        self.assertFalse(triton.enabled)

        # Test re-enabling
        triton.enabled = True
        self.assertTrue(triton.enabled)

        # Restore initial state
        triton.enabled = initial_state

    @patch("torch._native.registry._reenable_op_overrides")
    @patch("torch._native.registry._deregister_op_overrides")
    def test_cutedsl_set_flags(self, mock_disable, mock_reenable):
        """Test torch.backends.cutedsl.set_flags() function."""
        import torch.backends.cutedsl as cutedsl

        # Test set_flags with enabled=False
        orig_flags = cutedsl.set_flags(_enabled=False)
        self.assertEqual(orig_flags, (True,))  # Originally enabled
        self.assertFalse(cutedsl.enabled)
        mock_disable.assert_called_with(disable_dsl_names="cutedsl")

        # Test set_flags with enabled=True
        cutedsl.set_flags(_enabled=True)
        self.assertTrue(cutedsl.enabled)
        mock_reenable.assert_called_with(enable_dsl_names="cutedsl")

        # Test set_flags with None (no change)
        orig_flags = cutedsl.set_flags(_enabled=None)
        self.assertEqual(orig_flags, (True,))

    @patch("torch._native.registry._reenable_op_overrides")
    @patch("torch._native.registry._deregister_op_overrides")
    def test_triton_set_flags(self, mock_disable, mock_reenable):
        """Test torch.backends.triton.set_flags() function."""
        import torch.backends.triton as triton

        # Test set_flags with enabled=False
        orig_flags = triton.set_flags(_enabled=False)
        self.assertEqual(orig_flags, (True,))  # Originally enabled
        self.assertFalse(triton.enabled)
        mock_disable.assert_called_with(disable_dsl_names="triton")

        # Test set_flags with enabled=True
        triton.set_flags(_enabled=True)
        self.assertTrue(triton.enabled)
        mock_reenable.assert_called_with(enable_dsl_names="triton")

        # Test set_flags with None (no change)
        orig_flags = triton.set_flags(_enabled=None)
        self.assertEqual(orig_flags, (True,))

    def test_cutedsl_flags_context_manager(self):
        """Test torch.backends.cutedsl.flags context manager."""
        import torch.backends.cutedsl as cutedsl

        # Ensure starting state is enabled
        cutedsl.enabled = True
        initial_state = cutedsl.enabled

        # Test context manager disables and restores
        with cutedsl.flags(enabled=False):
            self.assertFalse(cutedsl.enabled)

        # Should restore original state
        self.assertEqual(cutedsl.enabled, initial_state)

    def test_triton_flags_context_manager(self):
        """Test torch.backends.triton.flags context manager."""
        import torch.backends.triton as triton

        # Ensure starting state is enabled
        triton.enabled = True
        initial_state = triton.enabled

        # Test context manager disables and restores
        with triton.flags(enabled=False):
            self.assertFalse(triton.enabled)

        # Should restore original state
        self.assertEqual(triton.enabled, initial_state)

    def test_cutedsl_flags_context_manager_exception_handling(self):
        """Test torch.backends.cutedsl.flags context manager restores state on exception."""
        import torch.backends.cutedsl as cutedsl

        # Ensure starting state is enabled
        cutedsl.enabled = True
        initial_state = cutedsl.enabled

        # Test exception handling
        with self.assertRaises(ValueError):
            with cutedsl.flags(enabled=False):
                self.assertFalse(cutedsl.enabled)
                raise ValueError("Test exception")

        # Should restore original state even after exception
        self.assertEqual(cutedsl.enabled, initial_state)

    def test_triton_flags_context_manager_exception_handling(self):
        """Test torch.backends.triton.flags context manager restores state on exception."""
        import torch.backends.triton as triton

        # Ensure starting state is enabled
        triton.enabled = True
        initial_state = triton.enabled

        # Test exception handling
        with self.assertRaises(ValueError):
            with triton.flags(enabled=False):
                self.assertFalse(triton.enabled)
                raise ValueError("Test exception")

        # Should restore original state even after exception
        self.assertEqual(triton.enabled, initial_state)

    def test_cutedsl_flags_context_manager_no_args(self):
        """Test torch.backends.cutedsl.flags context manager with no arguments."""
        import torch.backends.cutedsl as cutedsl

        original_state = cutedsl.enabled

        # Context manager with no args should not change state
        with cutedsl.flags():
            self.assertEqual(cutedsl.enabled, original_state)

        # State should remain unchanged after context
        self.assertEqual(cutedsl.enabled, original_state)

    def test_triton_flags_context_manager_no_args(self):
        """Test torch.backends.triton.flags context manager with no arguments."""
        import torch.backends.triton as triton

        original_state = triton.enabled

        # Context manager with no args should not change state
        with triton.flags():
            self.assertEqual(triton.enabled, original_state)

        # State should remain unchanged after context
        self.assertEqual(triton.enabled, original_state)

    def test_cutedsl_module_replacement(self):
        """Test that torch.backends.cutedsl module is properly replaced with CuTeDSLModule."""
        import torch.backends.cutedsl as cutedsl

        # Check that the module is the custom PropModule instance
        from torch.backends import PropModule

        self.assertIsInstance(sys.modules["torch.backends.cutedsl"], PropModule)

        # Test that enabled property works through the module replacement
        self.assertTrue(hasattr(cutedsl, "enabled"))
        self.assertIsInstance(cutedsl.enabled, bool)

    def test_triton_module_replacement(self):
        """Test that torch.backends.triton module is properly replaced with TritonModule."""
        import torch.backends.triton as triton

        # Check that the module is the custom PropModule instance
        from torch.backends import PropModule

        self.assertIsInstance(sys.modules["torch.backends.triton"], PropModule)

        # Test that enabled property works through the module replacement
        self.assertTrue(hasattr(triton, "enabled"))
        self.assertIsInstance(triton.enabled, bool)

    def test_both_backends_independent(self):
        """Test that cutedsl and triton backends operate independently."""
        import torch.backends.cutedsl as cutedsl
        import torch.backends.triton as triton

        # Both should start enabled
        self.assertTrue(cutedsl.enabled)
        self.assertTrue(triton.enabled)

        # Disable cutedsl, triton should remain enabled
        cutedsl.enabled = False
        self.assertFalse(cutedsl.enabled)
        self.assertTrue(triton.enabled)

        # Disable triton, cutedsl should remain disabled
        triton.enabled = False
        self.assertFalse(cutedsl.enabled)
        self.assertFalse(triton.enabled)

        # Re-enable cutedsl, triton should remain disabled
        cutedsl.enabled = True
        self.assertTrue(cutedsl.enabled)
        self.assertFalse(triton.enabled)

    @patch("torch._native.registry._reenable_op_overrides")
    @patch("torch._native.registry._deregister_op_overrides")
    def test_nested_context_managers(self, mock_disable, mock_reenable):
        """Test nested context managers for both backends."""
        import torch.backends.cutedsl as cutedsl
        import torch.backends.triton as triton

        # Both start enabled
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
            _deregister_op_overrides,
            _reenable_op_overrides,
        )

        # The backends should be able to call these functions without error
        # (though they may be no-ops if no overrides are registered)
        try:
            _deregister_op_overrides(disable_dsl_names="cutedsl")
            _reenable_op_overrides(enable_dsl_names="cutedsl")
            _deregister_op_overrides(disable_dsl_names="triton")
            _reenable_op_overrides(enable_dsl_names="triton")
        except Exception as e:
            self.fail(f"Registry integration failed: {e}")


if __name__ == "__main__":
    run_tests()
