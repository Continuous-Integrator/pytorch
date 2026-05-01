"""Re-export the vendored quack package under the torch._vendor.quack namespace.

The source lives at ``third_party/quack/quack`` (as a git submodule). This shim
exposes it without requiring quack to be installed separately, and forwards
submodule imports so that ``torch._vendor.quack.X`` and ``quack.X`` always
refer to the same module object.
"""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys

_QUACK_PARENT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "third_party",
        "quack",
    )
)

# quack's own __init__.py does `import quack.cute_dsl_elf_fix` etc., so the
# top-level "quack" name must be resolvable via sys.path.
if _QUACK_PARENT not in sys.path:
    sys.path.insert(0, _QUACK_PARENT)


class _QuackAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Forwards torch._vendor.quack[.X] imports to quack[.X]."""

    _PREFIX = __name__ + "."
    _TARGET_PREFIX = "quack."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        # Let Python create a placeholder module; we swap it for the real one
        # in exec_module.
        return None

    def exec_module(self, module):
        target = self._TARGET_PREFIX + module.__name__[len(self._PREFIX) :]
        sys.modules[module.__name__] = importlib.import_module(target)


if not any(isinstance(f, _QuackAliasFinder) for f in sys.meta_path):
    # Insert before PathFinder so torch._vendor.quack.X is resolved by us,
    # not by Python's path-based import (which would otherwise find quack's
    # source via the __path__ inherited from the quack module and execute
    # submodules a second time under a different name).
    sys.meta_path.insert(0, _QuackAliasFinder())

import quack as _quack  # noqa: E402

# Replace this module with quack so attribute access on torch._vendor.quack
# transparently resolves to quack's attributes.
sys.modules[__name__] = _quack
