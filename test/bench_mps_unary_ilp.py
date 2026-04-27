# Owner(s): ["module: mps"]
# Compares the wide (ILP) and one-thread-per-element (scalar) flavors of the
# MPS unary_dense kernel on representative ops. Toggled per-call via the
# PYTORCH_UNARY_FORCE_FLAVOR env var read by exec_unary_kernel.
import os
import timeit

import torch
from torch.utils.benchmark import Compare, Timer


SIZES = [1 << k for k in range(15, 22)]  # 32K .. 2M
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
OPS = {"sin": torch.sin, "silu": torch.nn.functional.silu}


def bench(fn, x, flavor):
    os.environ["PYTORCH_UNARY_FORCE_FLAVOR"] = flavor
    return Timer(
        stmt="f(x); torch.mps.synchronize()",
        globals={"f": fn, "x": x, "torch": torch},
        timer=timeit.default_timer,
        sub_label=f"{op_name} {str(x.dtype).replace('torch.', '')} {x.numel():>9d}",
        description=flavor,
    ).blocked_autorange(min_run_time=0.5)


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    results = []
    for op_name, fn in OPS.items():
        for n in SIZES:
            for dt in DTYPES:
                x = torch.randn(n, device="mps", dtype=dt)
                for flavor in ("scalar", "ilp"):
                    results.append(bench(fn, x, flavor))
    Compare(results).print()
