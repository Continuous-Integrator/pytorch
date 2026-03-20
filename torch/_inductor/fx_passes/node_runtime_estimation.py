"""
Node runtime estimation: CUDA events benchmarking and profile-guided estimation.
"""

from __future__ import annotations

import functools
import itertools
import operator
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import _schedulable_wait_node
from torch._inductor.utils import clear_on_fresh_cache
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.operator_schemas import normalize_function


if TYPE_CHECKING:
    from collections.abc import Callable


def _format_csv(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as CSV. Human-readable and easily parseable."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


def _get_collective_key(coll_node: fx.Node) -> str:
    """Extract a unique key for a collective node including group info and tensor size."""
    from torch._inductor import fx_utils

    opt_args_kwargs = normalize_function(
        coll_node.target,  # type: ignore[arg-type]
        args=coll_node.args,
        kwargs=coll_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    _, kwargs = opt_args_kwargs
    group_name = kwargs.get("group_name", None)
    group_size = kwargs.get("group_size", None)

    tensor_bytes: int | None = None
    success, args, kw = fx_utils.get_fake_args_kwargs(coll_node)
    if success:

        def extract_first_tensor_bytes(t: torch.Tensor) -> torch.Tensor:
            nonlocal tensor_bytes
            if tensor_bytes is None:
                shape = [get_hint(dim) for dim in t.shape]
                if all(s is not None for s in shape):
                    numel = functools.reduce(operator.mul, shape, 1)
                    tensor_bytes = numel * t.dtype.itemsize
            return t

        torch.utils._pytree.tree_map_only(
            torch.Tensor, extract_first_tensor_bytes, (args, kw)
        )

    return f"{coll_node.target} group_size:{group_size} group_name:{group_name} input_bytes:{tensor_bytes}"


def _get_collective_estimations(coll_node: fx.Node) -> tuple[float, float]:
    """Get NCCL and Inductor analytical estimations for a collective node.

    Returns: (nccl_ms, inductor_ms)
    """
    nccl_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=True
        )
    )
    inductor_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=False
        )
    )
    return nccl_ms, inductor_ms


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


# TODO: Consider using a distributed-aware cache or rank-local disk cache
# not using local cache because different ranks might write to it concurrently.
# solvable in future, potentially with workflow to seed cache
@clear_on_fresh_cache
@lru_cache
def _get_collective_cache() -> dict[str, float]:
    """Get process-local cache for collective benchmarks."""
    return {}


def get_cached_runtime(key: str) -> float | None:
    """Get cached runtime from process-local cache."""
    return _get_collective_cache().get(key)


def set_cached_runtime(key: str, value: float) -> None:
    """Set cached runtime in process-local cache."""
    _get_collective_cache()[key] = value


def get_hint(x: int | torch.SymInt) -> int | None:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    return x.node.hint if x.node.has_hint() else None


def can_benchmark_collective() -> bool:
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False

    pg = c10d.distributed_c10d._get_default_group()
    if (
        torch.distributed.distributed_c10d.get_backend(pg)
        == torch.distributed.distributed_c10d.Backend.FAKE
    ):
        return False

    return True


def _median(lst):
    assert len(lst) > 0
    return torch.median(torch.tensor(lst)).item()


def _benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    """
    Core benchmarking logic using CUDA events and barriers.
    Returns runtime in ms or None on failure.
    """
    from torch._dynamo.testing import rand_strided

    # Convert FakeTensors to real tensors before benchmarking
    def to_real(t: torch.Tensor) -> torch.Tensor:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            # This should not happen, as can_benhcmark_collective checks for unbacked
            raise ValueError("Cannot convert tensor with symbolic dimensions")

        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.Tensor,
        to_real,
        (args, kwargs),
    )

    # Warmup: call collective once and wait
    torch.cuda.synchronize()
    result = n.target(*args, **kwargs)  # type: ignore[operator]
    torch.ops._c10d_functional.wait_tensor(result)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    comm_times = []
    for _ in range(nruns):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        result = n.target(*args, **kwargs)  # type: ignore[operator]
        torch.ops._c10d_functional.wait_tensor(result)
        end_evt.record()
        end_evt.synchronize()

        comm_times.append(start_evt.elapsed_time(end_evt))

    return _median(comm_times)


def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    # context manager not allowed with profiler.
    with torch.utils._python_dispatch._disable_current_modes():
        return benchmark_collective_with_cuda_events_impl(n, nruns)


def benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    nruns: int = 3,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    from torch._inductor import fx_utils
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)

    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    group_name = opt_args_kwargs[1]["group_name"]
    group_size = _get_group_size_by_name(group_name)

    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes: int | None = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
        nonlocal actual_bytes
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                assert dim is not None
                total_elems *= dim

            actual_bytes = total_elems * t.dtype.itemsize
        else:
            # out-variants (e.g. all_gather_into_tensor_out) can have multiple tensors
            pass
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Cache key by BYTES (dtype-agnostic)
    key = f"{n.target}: ({group_size} group size, {actual_bytes} bytes)"

    # Check cache
    if (cached := get_cached_runtime(key)) is not None:
        return cached, key

    # Benchmark using CUDA events with actual args/kwargs
    runtime = _benchmark_collective_with_cuda_events_impl(n, args, kwargs, nruns)

    if runtime is None:
        return None, key

    # Cache the result
    set_cached_runtime(key, runtime)
    return runtime, key


def _log_compute_estimations(
    compute_nodes: list[fx.Node],
    benchmarked_estimations: list[float],
    analytical_estimations: list[float],
) -> None:
    """Log compute node runtime estimations comparing benchmarked vs analytical."""
    import torch.utils._pytree as pytree
    from torch._inductor.fx_utils import count_flops_fx
    from torch.utils._dtype_abbrs import dtype_abbrs

    def _node_summary(n: fx.Node) -> str:
        ret = str(n)
        for arg in pytree.arg_tree_leaves(n.args, n.kwargs):
            if not isinstance(arg, torch.fx.Node):
                continue
            if "val" in arg.meta:
                t = arg.meta["val"]
                ret += f" {dtype_abbrs[t.dtype]}{tuple(t.shape)}"
        return ret

    headers = [
        "Node",
        "Benchmarked Est(us)",
        "Analytical Est(us)",
        "Diff(ratio)",
        "Diff(us)",
        "Flops",
    ]

    rows = [
        [
            _node_summary(node),
            f"{est_b * 1e3:.4f}",
            f"{est_a * 1e3:.4f}",
            f"{(est_a / est_b) if est_b > 0 else 0:.4f}",
            f"{(est_a - est_b) * 1e3:.4f}",
            str(count_flops_fx(node)),
        ]
        for node, est_b, est_a in zip(
            compute_nodes, benchmarked_estimations, analytical_estimations
        )
    ]

    log_str = _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_compute_nodes_runtime_estimation",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


def _log_graph_collective_benchmarks(gm: fx.GraphModule, artifact_name: str) -> None:
    collective_nodes = []
    collective_keys = []
    benchmarked = []

    for node in gm.graph.nodes:
        if _schedulable_wait_node(node):
            start = node.args[0]
            collective_nodes.append(start)
            collective_keys.append(_get_collective_key(start))
            benchmarked_ms, _ = benchmark_collective_with_cuda_events(start, nruns=5)
            benchmarked.append(benchmarked_ms if benchmarked_ms else 0.0)

    if collective_nodes:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        _log_collective_benchmarks(
            collective_nodes,
            collective_keys,
            benchmarked,
            world_size,
            artifact_name,
        )


def _log_collective_benchmarks(
    collective_nodes: list[fx.Node],
    collective_keys: list[str] | None = None,
    benchmarked_medians: list[float] | None = None,
    world_size: int | None = None,
    artifact_name: str = "fx_collectives_analytical_estimation",
) -> None:
    """Log collective estimations for tlparse. Includes benchmarks if provided."""
    if world_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    has_benchmarks = benchmarked_medians is not None

    if has_benchmarks:
        headers = [
            "Collective Key",
            "Benchmarked(ms)",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
            "NCCL Diff(ratio)",
            "Inductor Diff(ratio)",
        ]
    else:
        headers = [
            "Collective Key",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
        ]

    rows = []
    for i, coll_node in enumerate(collective_nodes):
        key = collective_keys[i] if collective_keys else _get_collective_key(coll_node)
        nccl_ms, inductor_ms = _get_collective_estimations(coll_node)

        if benchmarked_medians is not None:
            benchmarked_ms = benchmarked_medians[i]
            nccl_diff_pct = (nccl_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            inductor_diff_pct = (
                (inductor_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            )
            rows.append(
                [
                    key,
                    f"{benchmarked_ms:.4f}",
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                    f"{nccl_diff_pct:.2f}",
                    f"{inductor_diff_pct:.2f}",
                ]
            )
        else:
            rows.append(
                [
                    key,
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                ]
            )

    log_str = f"# World size: {world_size}\n"
    log_str += _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


# ---------------------------------------------------------------------------
# Profile-Guided Latency Estimation (PGLE)
# ---------------------------------------------------------------------------


def make_profile_guided_estimator(
    trace_path: str,
) -> Callable[[fx.Node, int | None], float | None]:
    """Create a custom_runtime_estimation function from a Chrome Trace profile.

    Parses the profile and builds lookup tables for collectives, matmuls, and
    attention kernels. Returns a callable matching the custom_runtime_estimation
    interface: (fx.Node, int | None) -> float | None (ms or None for fallback).

    The trace_path supports a ``{rank}`` placeholder that is replaced with the
    current rank so each rank loads its own profile (exact PG rank match).
    Example: ``/traces/iteration_5/rank{rank}_trace.json``

    This is a generic API usable by any pass that needs runtime estimates.
    """
    from torch._inductor.fx_passes.profile_guided_estimation import (
        _get_collective_info,
        _get_mm_shapes,
        _get_node_dtype_str,
        _get_sdpa_key,
        _is_collective_node,
        _is_mm_node,
        _is_sdpa_node,
        _normalize_profile_indices,
        _rank_stride,
        ProfileData,
    )

    # Resolve {rank} placeholder
    resolved_path = trace_path
    if "{rank}" in trace_path:
        import torch.distributed as dist

        if dist.is_initialized():
            resolved_path = trace_path.replace("{rank}", str(dist.get_rank()))
        else:
            log.warning("PGLE: {rank} in path but dist not initialized, using rank 0")
            resolved_path = trace_path.replace("{rank}", "0")

    profile = ProfileData()
    profile.load(resolved_path)
    _normalize_profile_indices(profile)

    estimation_log: list[dict[str, Any]] = []
    miss_log: list[dict[str, Any]] = []

    def estimator(node: fx.Node, override_size: int | None = None) -> float | None:
        # Collectives
        if _is_collective_node(node):
            info = _get_collective_info(node)
            if info is None:
                miss_log.append(
                    {
                        "node": node.name,
                        "target": str(node.target),
                        "reason": "get_collective_info returned None",
                    }
                )
                return None
            coll_name, pg_ranks, nelems, dtype = info
            if override_size is not None:
                # override_size=0 means "startup latency only" — profiles don't
                # separate startup from transfer, so return 0 to let the caller
                # use the full estimated time as the exposed portion.
                if override_size == 0:
                    return 0.0
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    elem_size = val.element_size()
                    if elem_size > 0:
                        nelems = override_size // elem_size
            # Get bytes per element for bandwidth calculation
            dtype_bytes = 0
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                dtype_bytes = val.element_size()
            est = profile.lookup_collective(coll_name, pg_ranks, nelems, dtype)
            if est is not None:
                estimation_log.append(
                    {
                        "node": node.name,
                        "op": coll_name,
                        "nelems": nelems,
                        "dtype": dtype,
                        "dtype_bytes": dtype_bytes,
                        "group_size": len(pg_ranks),
                        "stride": _rank_stride(pg_ranks),
                        "pgle_ms": est,
                        "source": "profile",
                    }
                )
            else:
                miss_log.append(
                    {
                        "node": node.name,
                        "op": coll_name,
                        "nelems": nelems,
                        "dtype": dtype,
                        "group_size": len(pg_ranks),
                        "reason": "no match in profile",
                    }
                )
            return est

        # Matmul
        if _is_mm_node(node):
            shapes = _get_mm_shapes(node)
            if shapes is None:
                miss_log.append(
                    {
                        "node": node.name,
                        "target": str(node.target),
                        "reason": "get_mm_shapes returned None",
                    }
                )
                return None
            dtype = _get_node_dtype_str(node)
            est = profile.lookup_mm(shapes, dtype)
            if est is not None:
                estimation_log.append(
                    {
                        "node": node.name,
                        "op": "mm",
                        "shapes": [list(s) for s in shapes],
                        "dtype": dtype,
                        "pgle_ms": est,
                        "source": "profile",
                    }
                )
            else:
                miss_log.append(
                    {
                        "node": node.name,
                        "op": "mm",
                        "shapes": [list(s) for s in shapes],
                        "dtype": dtype,
                        "reason": "no match in profile",
                    }
                )
            return est

        # SDPA
        if _is_sdpa_node(node):
            sdpa_key = _get_sdpa_key(node)
            if sdpa_key is None:
                miss_log.append(
                    {
                        "node": node.name,
                        "target": str(node.target),
                        "reason": "get_sdpa_key returned None",
                    }
                )
                return None
            batch, heads, seq_len, head_dim, dtype, is_bwd = sdpa_key
            est = profile.lookup_sdpa(batch, heads, seq_len, head_dim, dtype, is_bwd)
            if est is not None:
                estimation_log.append(
                    {
                        "node": node.name,
                        "op": "sdpa_bwd" if is_bwd else "sdpa_fwd",
                        "shape": [batch, heads, seq_len, head_dim],
                        "dtype": dtype,
                        "pgle_ms": est,
                        "source": "profile",
                    }
                )
            else:
                miss_log.append(
                    {
                        "node": node.name,
                        "op": "sdpa",
                        "shape": [batch, heads, seq_len, head_dim],
                        "dtype": dtype,
                        "reason": "no match in profile",
                    }
                )
            return est

        return None

    estimator.estimation_log = estimation_log  # type: ignore[attr-defined]
    estimator.miss_log = miss_log  # type: ignore[attr-defined]
    estimator.profile = profile  # type: ignore[attr-defined]

    return estimator


def log_pgle_estimations(
    estimator: Callable[..., Any],
    analytical_estimates: dict[str, float] | None = None,
) -> None:
    """Dump PGLE estimation results via trace_structured for tlparse."""
    estimation_log = getattr(estimator, "estimation_log", None) or []
    miss_log = getattr(estimator, "miss_log", None) or []

    rows = []
    for entry in estimation_log:
        row = dict(entry)
        node_name = entry.get("node", "")
        if analytical_estimates and node_name in analytical_estimates:
            analytical_ms = analytical_estimates[node_name]
            row["analytical_ms"] = analytical_ms
            pgle_ms = entry.get("pgle_ms", 0)
            if analytical_ms > 0:
                row["pgle_vs_analytical_pct"] = round(
                    (pgle_ms - analytical_ms) / analytical_ms * 100, 1
                )
        rows.append(row)

    # Single combined table: estimations + misses at the bottom
    table = _format_pgle_table(rows, miss_log)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pgle_estimations_table",
            "encoding": "string",
        },
        payload_fn=lambda: table,
    )

    log.info(
        "PGLE: %d estimations, %d misses logged to trace_structured",
        len(rows),
        len(miss_log),
    )


def _format_bytes(nbytes: int) -> str:
    """Format byte count as human-readable K/M/G string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f}G"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f}M"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f}K"
    return f"{nbytes}B"


def _format_pgle_table(
    rows: list[dict[str, Any]],
    miss_log: list[dict[str, Any]] | None = None,
) -> str:
    """Format PGLE estimations + misses as a single aligned text table."""
    misses: list[dict[str, Any]] = list(miss_log) if miss_log is not None else []
    # GPU info header
    lines: list[str] = []
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        lines.append(f"GPU: {gpu_name} x{gpu_count}")
    except Exception:
        lines.append("GPU: unknown")
    lines.append("")

    has_analytical = any("analytical_ms" in r for r in rows)

    # Header
    if has_analytical:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3}"
            f" {'pgle_ms':>10} {'analytical_ms':>15} {'diff%':>8} {'':>3}"
            f" {'pgle_GB/s':>10} {'analytical_GB/s':>15}"
        )
    else:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3} {'pgle_ms':>10}"
        )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        node = row.get("node", "")[:44]
        op = row.get("op", "")[:29]
        gs = row.get("group_size", "")
        stride = row.get("stride", "")
        stride_str = str(stride) if stride is not None else "-"
        pgle_ms = row.get("pgle_ms", 0)
        dtype_bytes = row.get("dtype_bytes", 0)
        n = row.get("nelems", 0) if isinstance(row.get("nelems"), int) else 0
        size_str = _format_bytes(n * dtype_bytes) if dtype_bytes > 0 and n > 0 else "-"

        if has_analytical:
            anal_ms = row.get("analytical_ms", None)
            diff_pct = row.get("pgle_vs_analytical_pct", None)
            anal_str = f"{anal_ms:.4f}" if anal_ms is not None else "-"
            diff_str = f"{diff_pct:+.1f}%" if diff_pct is not None else "-"
            flag = ""
            if diff_pct is not None:
                adp = abs(diff_pct)
                if adp > 50:
                    flag = "***"
                elif adp > 15:
                    flag = "**"
            # Bandwidth in GB/s for comm ops
            pgle_bw_str = ""
            anal_bw_str = ""
            if dtype_bytes > 0 and n > 0:
                data_bytes = n * dtype_bytes
                if pgle_ms > 0:
                    pgle_bw = data_bytes / (pgle_ms * 1e-3) / 1e9
                    pgle_bw_str = f"{pgle_bw:.1f}"
                if anal_ms is not None and anal_ms > 0:
                    anal_bw = data_bytes / (anal_ms * 1e-3) / 1e9
                    anal_bw_str = f"{anal_bw:.1f}"
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pgle_ms:>10.4f} {anal_str:>15} {diff_str:>8} {flag:>3}"
                f" {pgle_bw_str:>10} {anal_bw_str:>15}"
            )
        else:
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pgle_ms:>10.4f}"
            )
        lines.append(line.rstrip())

    # Summary
    lines.append("")
    lines.append(f"Total: {len(rows)} estimations")
    if has_analytical:
        f1 = sum(
            1
            for r in rows
            if r.get("pgle_vs_analytical_pct") is not None
            and abs(r["pgle_vs_analytical_pct"]) > 50
        )
        f2 = sum(
            1
            for r in rows
            if r.get("pgle_vs_analytical_pct") is not None
            and 15 < abs(r["pgle_vs_analytical_pct"]) <= 50
        )
        lines.append(f"Flagged: {f2} ** (>15%), {f1} *** (>50%)")

    # Misses section
    if misses:
        lines.append("")
        lines.append(f"=== MISSES ({len(misses)}) ===")
        miss_header = f"{'node':<45} {'op':<30} {'reason'}"
        lines.append(miss_header)
        lines.append("-" * len(miss_header))
        for m in misses:  # pyrefly: ignore[unsupported-operation]
            node = m.get("node", "")[:44]
            op = (m.get("op") or m.get("target") or "")[:29]
            reason = m.get("reason", "")
            # Show shape info if available
            shapes = m.get("shapes")
            shape_info = m.get("shape")
            extra = ""
            if shapes:
                extra = f" shapes={shapes}"
            elif shape_info:
                extra = f" shape={shape_info}"
            lines.append(f"{node:<45} {op:<30} {reason}{extra}".rstrip())

    return "\n".join(lines)
