# Owner(s): ["module: dsl-native-ops"]

import logging
from typing import Any, Protocol


log = logging.getLogger(__name__)


class DSLModuleProtocol(Protocol):
    """Expected interface for DSL utility modules"""

    def runtime_available(self) -> bool: ...
    def runtime_version(self) -> str | None: ...


class DSLRegistry:
    """Registry for DSL modules - calls their existing API functions dynamically"""

    def __init__(self):
        self._dsl_modules: dict[str, Any] = {}

    def register_dsl(self, name: str, dsl_module: Any) -> None:
        """Register a DSL module with required interface"""
        # Validate interface at registration time to fail fast
        if not hasattr(dsl_module, "runtime_available") or not callable(
            dsl_module.runtime_available
        ):
            log.warning("DSL %s missing runtime_available() method", name)
            return
        if not hasattr(dsl_module, "runtime_version") or not callable(
            dsl_module.runtime_version
        ):
            log.warning("DSL %s missing runtime_version() method", name)
            return

        self._dsl_modules[name] = dsl_module

    def is_dsl_available(self, dsl_name: str) -> bool:
        """Check if DSL is available by calling its runtime_available()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return False
        try:
            return dsl_module.runtime_available()
        except Exception:
            log.debug("Error checking availability for DSL %s", dsl_name, exc_info=True)
            return False

    def get_dsl_version(self, dsl_name: str) -> str | None:
        """Get DSL version by calling its runtime_version()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return None
        try:
            return dsl_module.runtime_version()
        except Exception:
            log.debug("Error getting version for DSL %s", dsl_name, exc_info=True)
            return None

    def list_available_dsls(self) -> list[str]:
        """Get names of currently available DSLs"""
        available = []
        for name, dsl_module in self._dsl_modules.items():
            try:
                if dsl_module.runtime_available():
                    available.append(name)
            except Exception:
                log.debug("Error checking availability for DSL %s", name, exc_info=True)
        return available

    def list_all_dsls(self) -> list[str]:
        """Get all registered DSL names (available or not)"""
        return list(self._dsl_modules.keys())


# Global registry instance
dsl_registry = DSLRegistry()
