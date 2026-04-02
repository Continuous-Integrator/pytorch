import warnings

from .triton_impl import register_to_dispatch


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", ".*Warning only once for all operators.*")
    register_to_dispatch()
