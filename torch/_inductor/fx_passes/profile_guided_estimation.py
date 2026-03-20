"""
Profile-Guided Latency Estimation (PGLE) for overlap scheduling.

Parses a Chrome Trace JSON (from torch.profiler) and builds lookup tables
for collective, matmul, and attention kernel runtimes. These are used as
a custom_runtime_estimation hook in the overlap scheduler.

When the same profile is loaded on all ranks, estimates are deterministic
and no cross-rank synchronization is needed.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh dimension identification via rank stride patterns
# ---------------------------------------------------------------------------


def _rank_stride(ranks: tuple[int, ...]) -> int | None:
    """Compute the stride of a sorted rank tuple, or None if non-uniform.

    Examples:
        (0, 2, 4, 6) → stride 2
        (0, 1)       → stride 1
        (1, 3, 5, 7) → stride 2
        (0, 1, 4, 5) → None (non-uniform)
    """
    if len(ranks) <= 1:
        return None
    stride = ranks[1] - ranks[0]
    if stride <= 0:
        return None
    for i in range(2, len(ranks)):
        if ranks[i] - ranks[i - 1] != stride:
            return None
    return stride


# ---------------------------------------------------------------------------
# Data classes for profile records
# ---------------------------------------------------------------------------


@dataclass
class CollectiveRecord:
    """A single collective kernel observation from the profile."""

    collective_name: str  # "all_gather_into_tensor", "reduce_scatter_tensor", etc.
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    nelems: int
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float


@dataclass
class MatmulRecord:
    """A single aten::mm observation from the profile."""

    # input shapes: ((M, K), (K, N))
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    dtype: str
    duration_us: float  # sum of all GPU kernels for this CPU op


@dataclass
class SdpaRecord:
    """A single SDPA (scaled dot product attention) observation."""

    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: str
    is_backward: bool
    duration_us: float  # sum of all GPU kernels for this CPU op


# ---------------------------------------------------------------------------
# Profile data loader and lookup
# ---------------------------------------------------------------------------


@dataclass
class ProfileData:
    """Parse Chrome Trace JSON and build lookup tables for kernel runtimes."""

    collectives: list[CollectiveRecord] = field(default_factory=list)
    matmuls: list[MatmulRecord] = field(default_factory=list)
    sdpa_records: list[SdpaRecord] = field(default_factory=list)
    pg_configs: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Lookup indices built after loading
    _collective_index: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Fallback index by (name, stride, group_size, dtype) — matches PGs
    # belonging to the same mesh dimension regardless of specific ranks.
    # E.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4.
    _collective_index_by_stride: dict[
        tuple[str, int, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Count of distinct strides per (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share stride+size).
    _pg_count_by_stride: dict[tuple[int, int], int] = field(default_factory=dict)
    _matmul_index: dict[tuple[tuple[int, ...], tuple[int, ...], str], float] = field(
        default_factory=dict
    )
    _sdpa_index: dict[tuple[int, int, int, int, str, bool], float] = field(
        default_factory=dict
    )

    def load(self, trace_path: str) -> None:
        """Load and parse a Chrome Trace JSON file."""
        with open(trace_path) as f:
            data = json.load(f)

        self._parse_pg_configs(data)
        self._parse_events(data.get("traceEvents", []))
        self._build_indices()

        log.info(
            "PGLE loaded: %d collectives, %d matmuls, %d sdpa records, %d PGs",
            len(self.collectives),
            len(self.matmuls),
            len(self.sdpa_records),
            len(self.pg_configs),
        )

    def _parse_pg_configs(self, data: dict[str, Any]) -> None:
        dist_info = data.get("distributedInfo", {})
        pg_config = dist_info.get("pg_config", {})
        # pg_config can be a list of dicts or a dict of dicts
        if isinstance(pg_config, list):
            for pg_info in pg_config:
                pg_name = str(pg_info.get("pg_name", ""))
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))
        elif isinstance(pg_config, dict):
            for pg_name, pg_info in pg_config.items():
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))

    def _parse_events(self, events: list[dict[str, Any]]) -> None:
        # Build External id -> CPU op mapping
        cpu_ops: dict[int, dict[str, Any]] = {}
        for ev in events:
            cat = ev.get("cat", "")
            if cat == "cpu_op":
                eid = ev.get("args", {}).get("External id")
                if eid is not None:
                    cpu_ops[eid] = ev

        # Build External id -> list of GPU kernel durations
        gpu_kernels: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for ev in events:
            if ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            eid = args.get("External id")
            dur = ev.get("dur", 0.0)
            name = ev.get("name", "")
            if eid is not None and dur > 0:
                gpu_kernels[eid].append((name, dur))

        # Parse collectives from GPU kernel events directly
        # (NCCL kernels carry collective metadata in args)
        for ev in events:
            if ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            coll_name = args.get("Collective name")
            if coll_name is None:
                continue
            pg_name = args.get("Process Group Name", "")
            pg_ranks_str = args.get("Process Group Ranks", "")
            group_size = args.get("Group size", 0)
            nelems = args.get("In msg nelems", 0)
            dtype = args.get("dtype", "")
            dur = ev.get("dur", 0.0)
            if dur <= 0:
                continue

            # Parse ranks from string like "[0, 1, 2, 3]"
            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    nelems=nelems,
                    dtype=dtype,
                    duration_us=dur,
                )
            )

        # Parse matmuls and SDPA from CPU ops correlated to GPU kernels
        for eid, cpu_ev in cpu_ops.items():
            name = cpu_ev.get("name", "")
            cpu_args = cpu_ev.get("args", {})
            kernels = gpu_kernels.get(eid, [])
            if not kernels:
                continue
            total_dur = sum(dur for _, dur in kernels)

            if name == "aten::mm":
                self._parse_mm(cpu_args, total_dur)
            elif "attention" in name.lower() or "sdpa" in name.lower():
                self._parse_sdpa(name, cpu_args, total_dur)

    def _parse_ranks(self, ranks_str: str, pg_name: str) -> tuple[int, ...]:
        """Parse rank list from profile string or fall back to pg_configs."""
        if isinstance(ranks_str, str) and ranks_str.startswith("["):
            try:
                ranks = json.loads(ranks_str)
                return tuple(sorted(ranks))
            except (json.JSONDecodeError, TypeError):
                pass
        # Fall back to pg_configs
        if pg_name in self.pg_configs:
            return self.pg_configs[pg_name]
        return ()

    def _parse_mm(self, args: dict[str, Any], total_dur: float) -> None:
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        if len(input_dims) < 2:
            return
        dtype = input_types[0] if input_types else ""
        shapes = (tuple(input_dims[0]), tuple(input_dims[1]))
        self.matmuls.append(
            MatmulRecord(input_shapes=shapes, dtype=dtype, duration_us=total_dur)
        )

    def _parse_sdpa(self, op_name: str, args: dict[str, Any], total_dur: float) -> None:
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        if not input_dims or not input_dims[0]:
            return
        # Q tensor shape: [batch, num_heads, seq_len, head_dim]
        q_shape = input_dims[0]
        if len(q_shape) != 4:
            return
        dtype = input_types[0] if input_types else ""
        is_backward = "backward" in op_name.lower()
        self.sdpa_records.append(
            SdpaRecord(
                batch=q_shape[0],
                num_heads=q_shape[1],
                seq_len=q_shape[2],
                head_dim=q_shape[3],
                dtype=dtype,
                is_backward=is_backward,
                duration_us=total_dur,
            )
        )

    def _build_indices(self) -> None:
        """Build lookup indices from parsed records."""
        coll_idx: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        coll_idx_by_stride: dict[tuple[str, int, int, str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        # Track distinct PG rank sets per (stride, group_size) for ambiguity check
        pg_sets_by_stride: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.nelems, rec.duration_us)
            )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                coll_idx_by_stride[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.nelems, rec.duration_us)
                )
                pg_sets_by_stride[(stride, gs)].add(rec.pg_ranks)
        # Sort by nelems for interpolation
        self._collective_index = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx.items()
        }
        self._collective_index_by_stride = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx_by_stride.items()
        }
        self._pg_count_by_stride = {k: len(pgs) for k, pgs in pg_sets_by_stride.items()}

        # Matmul index: (shape_a, shape_b, dtype) -> avg_dur_us
        mm_groups: dict[tuple[tuple[int, ...], tuple[int, ...], str], list[float]] = (
            defaultdict(list)
        )
        for rec in self.matmuls:
            key = (rec.input_shapes[0], rec.input_shapes[1], rec.dtype)
            mm_groups[key].append(rec.duration_us)
        self._matmul_index = {k: sum(v) / len(v) for k, v in mm_groups.items()}

        # SDPA index: (batch, heads, seq_len, head_dim, dtype, is_bwd) -> avg_dur_us
        sdpa_groups: dict[tuple[int, int, int, int, str, bool], list[float]] = (
            defaultdict(list)
        )
        for rec in self.sdpa_records:
            key = (
                rec.batch,
                rec.num_heads,
                rec.seq_len,
                rec.head_dim,
                rec.dtype,
                rec.is_backward,
            )
            sdpa_groups[key].append(rec.duration_us)
        self._sdpa_index = {k: sum(v) / len(v) for k, v in sdpa_groups.items()}

    @staticmethod
    def _normalize_collective_name(name: str) -> str:
        """Normalize collective name between profile and FX conventions.

        Profile uses: _allgather_base, allreduce, reduce_scatter_tensor_coalesced
        FX uses: all_gather_into_tensor, all_reduce, reduce_scatter_tensor
        """
        n = name.lower()
        if "allgather" in n or "all_gather" in n:
            return "all_gather"
        if "reduce_scatter" in n:
            return "reduce_scatter"
        if "allreduce" in n or "all_reduce" in n:
            return "all_reduce"
        if "all_to_all" in n or "alltoall" in n:
            return "all_to_all"
        return name

    # -----------------------------------------------------------------------
    # Lookup methods (return ms or None)
    # -----------------------------------------------------------------------

    def lookup_collective(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Look up collective duration in ms. Interpolates by nelems in log-log space.

        Tries exact rank match first, then falls back to stride-based match
        (same mesh dimension across ranks: e.g. (0,2,4,6) and (1,3,5,7) both
        have stride=2, size=4 → same dimension).
        """
        norm_name = self._normalize_collective_name(collective_name)
        # Try exact rank match first
        key = (norm_name, pg_ranks, dtype)
        entries = self._collective_index.get(key)
        if not entries:
            # Fall back to stride-based match: same mesh dimension
            gs = len(pg_ranks)
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_stride.get((stride, gs), 0) == 1
            ):
                stride_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_stride.get(stride_key)
            if not entries:
                return None

        # Exact match
        for n, dur in entries:
            if n == nelems:
                return dur / 1e3  # us -> ms

        # Interpolation in log-log space
        return self._interpolate_log_log(entries, nelems)

    def _interpolate_log_log(
        self, entries: list[tuple[int, float]], target_nelems: int
    ) -> float | None:
        """Interpolate duration in log-log space (log(nelems) vs log(dur))."""
        if not entries or target_nelems <= 0:
            return None

        log_target = math.log(target_nelems)

        # Find bracketing entries
        lower: tuple[int, float] | None = None
        upper: tuple[int, float] | None = None
        for n, dur in entries:
            if n <= 0 or dur <= 0:
                continue
            if n <= target_nelems:
                lower = (n, dur)
            if n >= target_nelems and upper is None:
                upper = (n, dur)

        if lower is not None and upper is not None:
            log_n0, log_d0 = math.log(lower[0]), math.log(lower[1])
            log_n1, log_d1 = math.log(upper[0]), math.log(upper[1])
            if log_n1 == log_n0:
                return lower[1] / 1e3
            t = (log_target - log_n0) / (log_n1 - log_n0)
            log_dur = log_d0 + t * (log_d1 - log_d0)
            return math.exp(log_dur) / 1e3  # us -> ms
        elif lower is not None:
            # Extrapolate from nearest lower (scale proportionally)
            return (lower[1] * target_nelems / lower[0]) / 1e3
        elif upper is not None:
            # Extrapolate from nearest upper
            return (upper[1] * target_nelems / upper[0]) / 1e3

        return None

    def lookup_mm(
        self,
        input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        dtype: str,
    ) -> float | None:
        """Look up matmul duration in ms.

        Tries exact shape match first, then interpolates by FLOP ratio from
        the nearest matmul with the same dtype.
        """
        key = (input_shapes[0], input_shapes[1], dtype)
        dur_us = self._matmul_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        # Interpolate: scale by FLOP ratio from nearest same-dtype matmul
        target_flops = self._mm_flops(input_shapes[0], input_shapes[1])
        if target_flops <= 0:
            return None
        best_ratio = float("inf")
        best_dur: float | None = None
        for (sa, sb, dt), d in self._matmul_index.items():
            if dt != dtype:
                continue
            ref_flops = self._mm_flops(sa, sb)
            if ref_flops <= 0:
                continue
            ratio = max(target_flops / ref_flops, ref_flops / target_flops)
            if ratio < best_ratio:
                best_ratio = ratio
                best_dur = d * (target_flops / ref_flops)
        if best_dur is not None:
            return best_dur / 1e3
        return None

    @staticmethod
    def _mm_flops(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """Compute 2*M*N*K for a matmul (A=[M,K] @ B=[K,N])."""
        if len(a) >= 2 and len(b) >= 2:
            return 2 * a[-2] * b[-1] * a[-1]
        return 0

    def lookup_sdpa(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: str,
        is_backward: bool,
    ) -> float | None:
        """Look up SDPA duration in ms.

        Tries exact shape match first, then interpolates by FLOP ratio from
        the nearest SDPA with the same dtype and direction (fwd/bwd).
        """
        key = (batch, num_heads, seq_len, head_dim, dtype, is_backward)
        dur_us = self._sdpa_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        # Interpolate: SDPA FLOPs ~ batch * heads * seq_len^2 * head_dim
        target_flops = batch * num_heads * seq_len * seq_len * head_dim
        if target_flops <= 0:
            return None
        best_ratio = float("inf")
        best_dur: float | None = None
        for (b2, h2, s2, d2, dt2, bwd2), dur in self._sdpa_index.items():
            if dt2 != dtype or bwd2 != is_backward:
                continue
            ref_flops = b2 * h2 * s2 * s2 * d2
            if ref_flops <= 0:
                continue
            ratio = max(target_flops / ref_flops, ref_flops / target_flops)
            if ratio < best_ratio:
                best_ratio = ratio
                best_dur = dur * (target_flops / ref_flops)
        if best_dur is not None:
            return best_dur / 1e3
        return None


# ---------------------------------------------------------------------------
# FX node inspection helpers
# ---------------------------------------------------------------------------

# Mapping from torch dtype to profile dtype strings
_DTYPE_TO_PROFILE_STR: dict[torch.dtype, str] = {
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.int8: "Char",
    torch.uint8: "Byte",
}

# Also map C10 type strings to normalized form
_C10_DTYPE_TO_PROFILE_STR: dict[str, str] = {
    "c10::Float": "Float",
    "c10::Half": "Half",
    "c10::BFloat16": "BFloat16",
    "c10::Double": "Double",
    "c10::Int": "Int",
    "c10::Long": "Long",
    "c10::Char": "Char",
    "c10::Byte": "Byte",
}


def _dtype_to_profile_str(dtype: torch.dtype) -> str:
    return _DTYPE_TO_PROFILE_STR.get(dtype, str(dtype))


def _normalize_profile_dtype(dtype_str: str) -> str:
    """Normalize profile dtype string (may be 'c10::BFloat16' or 'BFloat16')."""
    return _C10_DTYPE_TO_PROFILE_STR.get(dtype_str, dtype_str)


def _get_node_dtype_str(node: fx.Node) -> str:
    """Extract dtype string from FX node metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return _dtype_to_profile_str(val.dtype)
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return _dtype_to_profile_str(first.dtype)
    return ""


def _is_mm_node(node: fx.Node) -> bool:
    """Check if node is a matrix multiplication."""
    return node.target in (
        torch.ops.aten.mm.default,
        torch.ops.aten.mm.out,
    )


def _is_sdpa_node(node: fx.Node) -> bool:
    """Check if node is a scaled dot-product attention op."""
    target = node.target
    if not hasattr(target, "__name__"):
        return False
    name = target.__name__ if hasattr(target, "__name__") else str(target)
    return any(
        kw in name.lower()
        for kw in ("scaled_dot_product", "cudnn_attention", "flash_attention", "sdpa")
    )


def _is_sdpa_backward(node: fx.Node) -> bool:
    """Check if SDPA node is a backward op."""
    target = node.target
    name = target.__name__ if hasattr(target, "__name__") else str(target)
    return "backward" in name.lower()


def _get_mm_shapes(
    node: fx.Node,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Extract (A_shape, B_shape) from mm node metadata."""
    if len(node.args) < 2:
        return None
    a_node, b_node = node.args[0], node.args[1]
    if not isinstance(a_node, fx.Node) or not isinstance(b_node, fx.Node):
        return None
    a_val = a_node.meta.get("val")
    b_val = b_node.meta.get("val")
    if not isinstance(a_val, torch.Tensor) or not isinstance(b_val, torch.Tensor):
        return None

    def _resolve_shape(t: torch.Tensor) -> tuple[int, ...] | None:
        from torch._inductor.fx_passes.node_runtime_estimation import get_hint

        shape = [get_hint(s) for s in t.shape]
        if any(s is None for s in shape):
            log.debug("PGLE: unresolved symbolic dims in shape %s", t.shape)
            return None
        return tuple(shape)  # type: ignore[arg-type]

    a_shape = _resolve_shape(a_val)
    b_shape = _resolve_shape(b_val)
    if a_shape is None or b_shape is None:
        return None
    return (a_shape, b_shape)


def _get_sdpa_key(
    node: fx.Node,
) -> tuple[int, int, int, int, str, bool] | None:
    """Extract (batch, heads, seq_len, head_dim, dtype, is_bwd) from SDPA node."""
    # Q is first input
    if not node.args:
        return None
    q_node = node.args[0]
    if not isinstance(q_node, fx.Node):
        return None
    q_val = q_node.meta.get("val")
    if not isinstance(q_val, torch.Tensor):
        return None
    shape = q_val.shape
    if len(shape) != 4:
        return None
    try:
        batch, heads, seq_len, head_dim = (int(s) for s in shape)
    except (TypeError, ValueError):
        return None
    dtype = _dtype_to_profile_str(q_val.dtype)
    is_bwd = _is_sdpa_backward(node)
    return (batch, heads, seq_len, head_dim, dtype, is_bwd)


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    return node.target in (
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.all_to_all_single.default,
    )


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], int, str] | None:
    """Extract (collective_name, pg_ranks, nelems, dtype) from collective node."""
    from torch.fx.operator_schemas import normalize_function

    try:
        target = node.target
        assert isinstance(target, torch._ops.OpOverload)
        collective_name = target.name().split("::")[-1].split(".")[0]

        opt = normalize_function(
            target,
            args=node.args,
            kwargs=node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt is None:
            return None
        _, kwargs = opt
        group_name = kwargs.get("group_name", "")

        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))

        # Get nelems from input tensor
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            nelems = 1
            for s in val.shape:
                nelems *= int(s)
            dtype = _dtype_to_profile_str(val.dtype)
        else:
            # Try first arg
            if node.args and isinstance(node.args[0], fx.Node):
                inp_val = node.args[0].meta.get("val")
                if isinstance(inp_val, torch.Tensor):
                    nelems = 1
                    for s in inp_val.shape:
                        nelems *= int(s)
                    dtype = _dtype_to_profile_str(inp_val.dtype)
                else:
                    return None
            else:
                return None

        return (collective_name, pg_ranks, nelems, dtype)
    except Exception:
        log.debug(
            "PGLE: failed to extract collective info for %s", node.name, exc_info=True
        )
        return None


def _normalize_profile_indices(profile: ProfileData) -> None:
    """Normalize dtype strings in profile indices to match profile format."""
    # Rebuild collective indices with normalized dtypes
    new_coll: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = {}
    for (coll_name, pg_ranks, dtype), entries in profile._collective_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        key = (coll_name, pg_ranks, norm_dtype)
        if key in new_coll:
            new_coll[key].extend(entries)
        else:
            new_coll[key] = list(entries)
    profile._collective_index = {
        k: sorted(v, key=lambda x: x[0]) for k, v in new_coll.items()
    }

    new_coll_by_stride: dict[tuple[str, int, int, str], list[tuple[int, float]]] = {}
    for (
        coll_name,
        stride,
        gs,
        dtype,
    ), entries in profile._collective_index_by_stride.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        key = (coll_name, stride, gs, norm_dtype)
        if key in new_coll_by_stride:
            new_coll_by_stride[key].extend(entries)
        else:
            new_coll_by_stride[key] = list(entries)
    profile._collective_index_by_stride = {
        k: sorted(v, key=lambda x: x[0]) for k, v in new_coll_by_stride.items()
    }

    # Rebuild matmul index with normalized dtypes (average on collision)
    mm_groups: dict[tuple[tuple[int, ...], tuple[int, ...], str], list[float]] = (
        defaultdict(list)
    )
    for (sa, sb, dtype), dur in profile._matmul_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        mm_groups[(sa, sb, norm_dtype)].append(dur)
    profile._matmul_index = {k: sum(v) / len(v) for k, v in mm_groups.items()}

    # Rebuild sdpa index with normalized dtypes (average on collision)
    sdpa_groups: dict[tuple[int, int, int, int, str, bool], list[float]] = defaultdict(
        list
    )
    for (b, h, s, d, dtype, bwd), dur in profile._sdpa_index.items():
        norm_dtype = _normalize_profile_dtype(dtype)
        sdpa_groups[(b, h, s, d, norm_dtype, bwd)].append(dur)
    profile._sdpa_index = {k: sum(v) / len(v) for k, v in sdpa_groups.items()}
