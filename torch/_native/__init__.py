import os
from functools import cache

# This handles collecting registration of all native ops
from . import ops, registry


@cache
def get_user_ordering_fn() -> registry.UserOrderingFn | None:
    env_var = os.getenv("TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN")

    if not env_var:
        return None

    try:
        import importlib

        module_name, fn_name = env_var.split(".", 1)

        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)

        if not callable(fn):
            raise TypeError(f"{env_var} does not describe a callable function")

        return fn
    except Exception as e:
        raise ValueError(
            f"Could not resolve {env_var} into an importable & callable function"
        ) from e


user_order_fn = get_user_ordering_fn()
if user_order_fn:
    registry.reorder_graphs_from_user_fn(user_order_fn)


# Actually perform all registrations
registry._register_all_overrides()
