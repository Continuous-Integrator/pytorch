# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import subprocess
import sys
import textwrap
import uuid
from unittest.mock import patch

from torch.testing._internal.common_utils import run_tests, TestCase


def _subprocess_lastline(script, env=None):
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return result.rsplit("\n", 1)[-1]


def _import_module_directly(module_name, file_name):
    """Import a module directly without triggering package imports."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_root = os.path.dirname(os.path.dirname(test_dir))
    module_path = os.path.join(pytorch_root, "torch", "_native", file_name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestNativeDSLOps(TestCase):
    """Tests for the torch._native DSL ops framework."""

    def setUp(self):
        """Clear all caches before each test to ensure test isolation."""
        self._cache_functions_to_clear = [
            (
                "torch._native.common_utils",
                ["check_native_jit_disabled", "check_native_version_skip"],
            ),
            (
                "torch._native.triton_utils",
                [
                    "_version_is_sufficient",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.cutedsl_utils",
                [
                    "_version_is_ok",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
        ]
        self._clear_function_caches()

    def _clear_function_caches(self):
        """Helper method to clear function caches with error handling."""
        for module_name, function_names in self._cache_functions_to_clear:
            try:
                module = __import__(module_name, fromlist=function_names)
                for func_name in function_names:
                    if hasattr(module, func_name):
                        getattr(module, func_name).cache_clear()
            except (AttributeError, ImportError):
                # Some functions might not exist or be cached, ignore errors
                pass

    def test_consistent_helper_interface(self):
        """Test triton_utils and cutedsl_utils expose consistent public APIs."""
        modules_info = [
            ("triton_utils.py", "torch._native.triton_utils"),
            ("cutedsl_utils.py", "torch._native.cutedsl_utils"),
        ]

        # Import modules directly to avoid dependency issues
        modules = {}
        for file_name, module_name in modules_info:
            modules[module_name] = _import_module_directly(module_name, file_name)

        required_methods = {
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        }

        # Test each module has required methods and they're callable
        public_apis = {}
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="required_methods"):
                public = {name for name in dir(mod) if not name.startswith("_")}
                public_apis[module_name] = public

                self.assertTrue(
                    required_methods <= public,
                    f"{module_name} missing: {required_methods - public}",
                )

                for method_name in required_methods:
                    with self.subTest(module=module_name, method=method_name):
                        self.assertTrue(callable(getattr(mod, method_name)))

        # Test modules expose identical public APIs
        api_sets = list(public_apis.values())
        self.assertEqual(
            api_sets[0], api_sets[1], "Modules should have identical public APIs"
        )

        # Test runtime functions return expected types
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="runtime_functions"):
                # runtime_available should return bool
                self.assertIsInstance(mod.runtime_available(), bool)

                # runtime_version should return Version or None
                ver = mod.runtime_version()
                if ver is not None:
                    from packaging.version import Version

                    self.assertIsInstance(ver, Version)

    def test_no_dsl_imports_after_import_torch(self):
        """import torch must not transitively import DSL runtimes.

        Note: cuda.bindings may appear because importlib.util.find_spec on
        nested modules (e.g. cuda.bindings.driver) imports parent packages
        as a side-effect.  We check only the primary DSL runtimes here.
        """
        script = textwrap.dedent("""\
            import sys
            import torch
            dsl_modules = ["triton", "cutlass", "tvm_ffi"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "[]", f"DSL modules leaked on import torch: {result}")

    def test_check_native_jit_disabled_environment_variable(self):
        """Test TORCH_DISABLE_NATIVE_JIT environment variable behavior."""
        from torch._native.common_utils import check_native_jit_disabled

        env_scenarios = [
            ({}, False, "unset environment variable"),
            ({"TORCH_DISABLE_NATIVE_JIT": "1"}, True, "set to 1"),
        ]

        for env_patch, expected_result, description in env_scenarios:
            with self.subTest(scenario=description):
                with patch.dict(os.environ, env_patch, clear=False):
                    if not env_patch:  # For empty dict, ensure var is not set
                        os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)

                    # Clear cache so function re-reads environment variable
                    check_native_jit_disabled.cache_clear()
                    self.assertEqual(check_native_jit_disabled(), expected_result)

    def test_unavailable_reason_missing(self):
        """Nonexistent package -> _unavailable_reason returns a string."""
        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )
        reason = common_utils._unavailable_reason(
            [("nonexistent_pkg_xyz", "nonexistent_pkg_xyz")]
        )
        self.assertIsNotNone(reason)
        self.assertIn("nonexistent_pkg_xyz", reason)

    def test_available_version_parsing(self):
        """Test _available_version parses various version formats and handles invalid ones."""
        from packaging.version import Version

        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )

        # Test with real package that has clean version
        ver = common_utils._available_version("typing_extensions")
        self.assertIsInstance(ver, Version)

        # Test various version format scenarios
        version_scenarios = [
            ("0.7.0rc1", "pre-release version"),
            ("3.1.0.post1", "post-release version"),
            ("2.4.0a1", "alpha version"),
            ("1.2.3", "standard version"),
            ("abc", "invalid version string"),
        ]

        for version_str, description in version_scenarios:
            with self.subTest(version=version_str, scenario=description):
                with patch("importlib.metadata.version", return_value=version_str):
                    result = common_utils._available_version("fake_package")

                    if version_str == "abc":
                        # Completely unparsable -> None
                        self.assertIsNone(result)
                    else:
                        # Valid versions should parse correctly
                        self.assertEqual(
                            result,
                            Version(version_str),
                            f"_available_version({version_str!r}) = {result}",
                        )

    def test_registry_mechanics(self):
        """_get_or_create_library caches Library instances per (lib, dispatch_key)."""
        import torch.library

        registry = _import_module_directly("torch._native.registry", "registry.py")

        key = ("_test_native_dsl_registry", "CPU")
        registry._libs.pop(key, None)

        lib1 = registry._get_or_create_library(*key)
        self.assertIsInstance(lib1, torch.library.Library)
        lib2 = registry._get_or_create_library(*key)
        self.assertIs(lib1, lib2, "should return cached instance")

        # Different dispatch key -> different Library
        key2 = ("_test_native_dsl_registry", "CUDA")
        registry._libs.pop(key2, None)
        lib3 = registry._get_or_create_library(*key2)
        self.assertIsNot(lib1, lib3)

        # cleanup
        registry._libs.pop(key, None)
        registry._libs.pop(key2, None)

    def test_deregister_op_overrides_functionality(self):
        """Test deregister_op_overrides methods exist, are callable, and work correctly."""
        modules_to_test = [
            ("triton_utils.py", "torch._native.triton_utils"),
            ("cutedsl_utils.py", "torch._native.cutedsl_utils"),
        ]

        for file_name, module_name in modules_to_test:
            with self.subTest(module=module_name):
                mod = _import_module_directly(module_name, file_name)

                # Test method exists and is callable
                self.assertTrue(hasattr(mod, "deregister_op_overrides"))
                self.assertTrue(callable(mod.deregister_op_overrides))

                # Test method can be called without error (should be no-op when no overrides registered)
                try:
                    mod.deregister_op_overrides()
                except Exception as e:
                    self.fail(
                        f"deregister_op_overrides on {module_name} raised exception: {e}"
                    )

    def test_register_op_skips_when_jit_disabled(self):
        """register_op_override does not call through when TORCH_DISABLE_NATIVE_JIT=1."""
        from torch._native import cutedsl_utils, triton_utils

        # Test the actual environment variable behavior to ensure it works
        # Set TORCH_DISABLE_NATIVE_JIT=1 and clear caches
        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "1"}):
            # Import and clear caches for both modules
            from torch._native.common_utils import check_native_jit_disabled

            check_native_jit_disabled.cache_clear()

            # Import functions from each module and clear their caches too
            triton_utils.check_native_jit_disabled.cache_clear()
            cutedsl_utils.check_native_jit_disabled.cache_clear()

            # Verify the function returns True
            self.assertTrue(check_native_jit_disabled())

            # Mock the registry calls to count how many times they would be called
            with patch("torch._native.registry.register_op_override") as registry_mock:
                # Use a unique operation name
                unique_op = f"test_jit_disabled_{uuid.uuid4().hex[:8]}.Tensor"
                triton_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda: None
                )
                cutedsl_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda: None
                )
                # Should not call the registry function at all since JIT is disabled
                self.assertEqual(registry_mock.call_count, 0)

    def test_version_skip_env_var_overrides(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 allows non-blessed versions."""
        from packaging.version import Version

        fake_version = Version("99.99.99")

        # Set the environment variable and clear caches
        with patch.dict(os.environ, {"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}):
            # Import fresh modules to avoid cached state
            from torch._native import cutedsl_utils, triton_utils
            from torch._native.common_utils import check_native_version_skip

            # Clear all relevant caches to ensure clean state
            check_native_version_skip.cache_clear()

            # Clear module-specific caches with error handling
            for module, cache_names in [
                (
                    triton_utils,
                    [
                        "_version_is_sufficient",
                        "check_native_jit_disabled",
                        "check_native_version_skip",
                    ],
                ),
                (
                    cutedsl_utils,
                    [
                        "_version_is_ok",
                        "check_native_jit_disabled",
                        "check_native_version_skip",
                    ],
                ),
            ]:
                for cache_name in cache_names:
                    if hasattr(module, cache_name):
                        getattr(module, cache_name).cache_clear()

            with (
                patch.object(
                    triton_utils,
                    "_check_runtime_available",
                    return_value=(True, fake_version),
                ),
                patch.object(
                    cutedsl_utils,
                    "_check_runtime_available",
                    return_value=(True, fake_version),
                ),
                patch.object(triton_utils, "_register_op_override_impl") as triton_mock,
                patch.object(cutedsl_utils, "_register_op_override_impl") as cute_mock,
            ):
                # Use unique operation names to avoid conflicts
                op_name = f"test_version_skip_{uuid.uuid4().hex[:8]}.Tensor"

                # Call the register functions
                triton_utils.register_op_override("aten", op_name, "CPU", lambda: None)
                cutedsl_utils.register_op_override("aten", op_name, "CPU", lambda: None)

                # Verify both implementation functions were called
                self.assertEqual(
                    triton_mock.call_count + cute_mock.call_count,
                    2,
                    f"Expected 2 calls but got triton: {triton_mock.call_count}, cutedsl: {cute_mock.call_count}",
                )

    def test_check_native_version_skip_environment_variable(self):
        """Test TORCH_NATIVE_SKIP_VERSION_CHECK environment variable behavior."""
        from torch._native.common_utils import check_native_version_skip

        env_scenarios = [
            ({}, False, "unset environment variable"),
            ({"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}, True, "set to 1"),
        ]

        for env_patch, expected_result, description in env_scenarios:
            with self.subTest(scenario=description):
                with patch.dict(os.environ, env_patch, clear=False):
                    if not env_patch:  # For empty dict, ensure var is not set
                        os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)

                    # Clear cache so function re-reads environment variable
                    check_native_version_skip.cache_clear()
                    self.assertEqual(check_native_version_skip(), expected_result)


if __name__ == "__main__":
    run_tests()
