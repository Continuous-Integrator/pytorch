# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

from packaging.version import Version

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


# Store enabled state in a container to avoid module replacement issues
_state = {"enabled": True}


def is_available() -> bool:
    r"""Return a bool indicating if the Triton runtime is currently available."""
    from torch._native import triton_utils

    return triton_utils.runtime_available()


def version() -> Version | None:
    r"""Return the installed Triton runtime version, or None if unavailable."""
    from torch._native import triton_utils

    return triton_utils.runtime_version()


def _set_enabled(_enabled: bool) -> None:
    from torch._native.registry import deregister_op_overrides, reenable_op_overrides

    _state["enabled"] = _enabled

    if _enabled:
        reenable_op_overrides(enable_dsl_names="triton")
    else:
        deregister_op_overrides(disable_dsl_names="triton")


def _get_enabled() -> bool:
    return _state["enabled"]


def set_flags(_enabled=None):
    import sys
    current_module = sys.modules[__name__]
    orig_flags = (current_module.enabled,)
    if _enabled is not None:
        _set_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=None):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(_enabled=enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(_enabled=orig_flags[0])


class TritonModule(PropModule):
    enabled = ContextProp(_get_enabled, _set_enabled)


sys.modules[__name__] = TritonModule(sys.modules[__name__], __name__)
