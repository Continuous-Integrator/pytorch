"""
User-facing API for controlling DSL operation overrides.

The torch.backends.python_native module provides control over DSL (Domain Specific Language)
operation overrides defined in torch._native. This allows users to selectively enable or disable
high-performance implementations from various DSLs like Triton and CuteDSL.

The module supports both coarse-grained control (entire DSLs) and fine-grained control
(individual operations or dispatch keys). All control operations support context managers
for temporary state changes.

Example usage::

    import torch.backends.python_native as pn

    # DSL-level control
    pn.triton.enabled = False  # Disable all triton ops
    pn.cutedsl.enabled = True  # Enable all cutedsl ops

    # Individual operation control
    pn.disable_operations("scaled_mm")  # Disable specific op across all DSLs
    pn.enable_operations("scaled_mm")  # Re-enable specific op

    # Context manager support
    with pn.triton.disabled():
        result = some_computation()  # Triton ops disabled here

    # Query capabilities
    print(pn.available_dsls)  # ['triton', 'cutedsl']
    print(pn.get_dsl_operations("triton"))  # Operations for triton
"""

import functools
import sys
import types
from contextlib import contextmanager


def _get_dsl_registry():
    """Lazy import to avoid circular imports."""
    from torch._native.dsl_registry import dsl_registry

    return dsl_registry


def _get_registry_functions():
    """Lazy import of registry functions."""
    from torch._native.registry import (
        _graphs,
        deregister_op_overrides,
        reenable_op_overrides,
    )

    return deregister_op_overrides, reenable_op_overrides, _graphs


def _get_dsl_module(dsl_name: str):
    """Get the registered DSL module for direct control.

    Uses the DSL registry to dynamically look up DSL modules instead of
    hard-coding the mapping. This makes the function automatically extensible
    for new DSLs without code changes.

    Args:
        dsl_name (str): Name of the DSL to retrieve.

    Returns:
        DSLModuleProtocol: The registered DSL module.

    Raises:
        ValueError: If the DSL is not registered.
    """
    registry = _get_dsl_registry()

    # Access the registered DSL module from the registry
    if dsl_name in registry._dsl_modules:
        return registry._dsl_modules[dsl_name]
    else:
        raise ValueError(f"Unknown DSL: {dsl_name}. Available DSLs: {registry.list_all_dsls()}")


class DSLController:
    """Controller for a specific DSL."""

    def __init__(self, dsl_name: str):
        self._dsl_name = dsl_name
        self._enabled_state = True  # Track our enable/disable state

    @property
    def name(self) -> str:
        return self._dsl_name

    @property
    def available(self) -> bool:
        """Check if DSL runtime is available."""
        registry = _get_dsl_registry()
        return registry.is_dsl_available(self._dsl_name)

    @property
    def version(self):
        """Get DSL version."""
        registry = _get_dsl_registry()
        return registry.get_dsl_version(self._dsl_name)

    @property
    def enabled(self) -> bool:
        """Check if DSL is currently enabled."""
        return self._enabled_state

    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable the DSL."""
        if value:
            self.enable()
        else:
            self.disable()

    def disable(self):
        """Disable all operations for this DSL."""
        dsl_module = _get_dsl_module(self._dsl_name)
        dsl_module.deregister_op_overrides()
        self._enabled_state = False

    def enable(self):
        """Re-enable all operations for this DSL."""
        deregister_op_overrides, reenable_op_overrides, _ = _get_registry_functions()
        reenable_op_overrides(enable_dsl_names=self._dsl_name)
        self._enabled_state = True

    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable DSL."""
        original_state = self._enabled_state
        try:
            self.disable()
            yield
        finally:
            if original_state:
                self.enable()

    def __repr__(self):
        status = "available" if self.available else "unavailable"
        enabled_status = "enabled" if self.enabled else "disabled"
        return f"DSLController({self._dsl_name}, {status}, {enabled_status})"


class PythonNativeModule(types.ModuleType):
    """Main module for python_native DSL control."""

    def __init__(self, original_module):
        super().__init__(original_module.__name__)

        # Copy over existing attributes
        for attr in dir(original_module):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(original_module, attr))

    @property
    def available_dsls(self) -> list[str]:
        """Get list of available DSLs."""
        registry = _get_dsl_registry()
        return registry.list_available_dsls()

    @property
    def all_dsls(self) -> list[str]:
        """Get list of all registered DSLs."""
        registry = _get_dsl_registry()
        return registry.list_all_dsls()

    @functools.lru_cache(maxsize=32)  # noqa: B019
    def get_dsl_operations(self, dsl_name: str) -> list[str]:
        """Get list of operations registered by a specific DSL.

        Args:
            dsl_name (str): Name of the DSL to query (e.g., 'triton', 'cutedsl').

        Returns:
            list[str]: Sorted list of operation names registered by the DSL.

        Example::

            ops = torch.backends.python_native.get_dsl_operations("triton")
            print(ops)  # ['triton_to_mxfp8_dim0', ...]
        """
        _, _, graphs = _get_registry_functions()
        operations = set()

        for (op_symbol, dispatch_key), nodes in graphs.items():
            for node in nodes:
                if node.dsl_name == dsl_name:
                    operations.add(op_symbol)
                    break

        return sorted(operations)

    def disable_operations(self, *op_symbols: str):
        """Disable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to disable.

        Example::

            # Disable scaled matrix multiply across all DSLs
            torch.backends.python_native.disable_operations("scaled_mm")

            # Disable multiple operations
            torch.backends.python_native.disable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        deregister_op_overrides, _, _ = _get_registry_functions()
        deregister_op_overrides(disable_op_symbols=list(op_symbols))

    def enable_operations(self, *op_symbols: str):
        """Re-enable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to re-enable.

        Example::

            # Re-enable previously disabled operations
            torch.backends.python_native.enable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        _, reenable_op_overrides, _ = _get_registry_functions()
        reenable_op_overrides(enable_op_symbols=list(op_symbols))

    def disable_dispatch_keys(self, *dispatch_keys: str):
        """Disable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to disable (e.g., 'CUDA', 'CPU').

        Example::

            # Disable all native operations on CUDA
            torch.backends.python_native.disable_dispatch_keys("CUDA")
        """
        deregister_op_overrides, _, _ = _get_registry_functions()
        deregister_op_overrides(disable_dispatch_keys=list(dispatch_keys))

    def enable_dispatch_keys(self, *dispatch_keys: str):
        """Re-enable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to re-enable (e.g., 'CUDA', 'CPU').

        Example::

            # Re-enable native operations on CUDA
            torch.backends.python_native.enable_dispatch_keys("CUDA")
        """
        _, reenable_op_overrides, _ = _get_registry_functions()
        reenable_op_overrides(enable_dispatch_keys=list(dispatch_keys))

    @contextmanager
    def operations_disabled(self, *op_symbols: str):
        """Context manager to temporarily disable operations.

        Args:
            *op_symbols (str): Names of operations to temporarily disable.

        Example::

            with torch.backends.python_native.operations_disabled("scaled_mm"):
                # scaled_mm is disabled across all DSLs
                result = model(input)
            # scaled_mm is automatically re-enabled here
        """
        # Disable operations
        self.disable_operations(*op_symbols)
        try:
            yield
        finally:
            # Re-enable operations
            self.enable_operations(*op_symbols)

    @functools.lru_cache(maxsize=16)  # noqa: B019
    def _get_dsl_controller(self, name: str) -> "DSLController":
        """Get or create a DSL controller (cached)."""
        return DSLController(name)

    def __getattr__(self, name: str):
        """Dynamic attribute access for DSL controllers."""
        if name in self.all_dsls:
            return self._get_dsl_controller(name)

        raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

    def __dir__(self):
        """Return available attributes including DSL names."""
        attrs = [
            "available_dsls",
            "all_dsls",
            "get_dsl_operations",
            "disable_operations",
            "enable_operations",
            "disable_dispatch_keys",
            "enable_dispatch_keys",
            "operations_disabled",
        ]

        # Add DSL names
        try:
            attrs.extend(self.all_dsls)
        except Exception:
            # If registry not available yet, skip DSL names
            pass

        return sorted(set(attrs))


# Replace the current module with our enhanced version
sys.modules[__name__] = PythonNativeModule(sys.modules[__name__])
