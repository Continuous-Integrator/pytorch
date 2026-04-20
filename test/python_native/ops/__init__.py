"""
Auto-discovery and loading of DSL OpInfo tests.

This module automatically discovers and imports OpInfo definitions from all
subdirectories containing DSL operation tests.
"""

import logging
import os
import sys
from pathlib import Path

from torch.testing._internal.opinfo.core import OpInfo


log = logging.getLogger(__name__)


def discover_dsl_opinfos() -> list[OpInfo]:
    """
    Automatically discover and load all DSL OpInfo definitions from subdirectories.

    Each subdirectory should contain Python files that export OpInfo instances.
    The OpInfo instances should be named with a suffix '_opinfo' (e.g., triton_silu_opinfo).

    Returns:
        List of discovered OpInfo instances.
    """
    opinfos = []

    # Get the directory containing this __init__.py file
    ops_dir = Path(__file__).parent

    # Iterate through all subdirectories
    for item in ops_dir.iterdir():
        if not item.is_dir() or item.name.startswith("_"):
            continue

        op_name = item.name

        # Look for Python files in the subdirectory
        for py_file in item.glob("test_*.py"):
            module_name = py_file.stem  # Remove .py extension

            try:
                # Import the module dynamically
                module_path = f"python_native.ops.{op_name}.{module_name}"
                __import__(module_path, fromlist=[""])
                module = sys.modules[module_path]

                # Look for OpInfo instances (anything ending with '_opinfo')
                for attr_name in dir(module):
                    if attr_name.endswith("_opinfo"):
                        attr_value = getattr(module, attr_name)
                        if isinstance(attr_value, OpInfo):
                            opinfos.append(attr_value)
                            log.info(
                                "Discovered DSL OpInfo: %s from %s",
                                attr_value.name,
                                module_path,
                            )

            except ImportError as e:
                # Skip modules that can't be imported (missing dependencies, etc.)
                log.info("Skipping %s: %s", module_path, e)
                continue

    return opinfos


# Automatically load all DSL OpInfos when this module is imported
dsl_opinfos = discover_dsl_opinfos()

# Export for easy access
__all__ = ["dsl_opinfos", "discover_dsl_opinfos"]
