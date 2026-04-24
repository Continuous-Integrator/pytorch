# Backward-compatibility shim: use torch.distributed.spmd_types instead.
from torch.distributed.spmd_types import *  # noqa: F403
from torch.distributed.spmd_types import _reset, is_available  # noqa: F401
