"""Helper module for test_nested_graph_breaks.py.

Functions here are intentionally in a separate module so they have different
f_globals from the compiled function that inlines them.
"""

import torch

_MODULE_CONSTANT = 2


def fn_with_module_global(x: torch.Tensor) -> torch.Tensor:
    x = x + 1
    torch._dynamo.graph_break()
    # _MODULE_CONSTANT is a module-level name in THIS file, not the caller's.
    # If the resume function doesn't use this module's f_globals, this will
    # raise NameError.
    return x + _MODULE_CONSTANT
