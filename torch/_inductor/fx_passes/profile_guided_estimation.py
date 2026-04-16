"""
Profile-Guided Estimation (PGE) for overlap scheduling.

Parses a Chrome Trace JSON (from torch.profiler) and builds lookup tables
for kernel runtimes (collectives, matmuls, attention, custom ops, etc.).
Used as a custom_runtime_estimation hook in the overlap scheduler.

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
from torch._inductor.analysis.profile_analysis import (
    _create_extern_mapping,
    _get_size_from_string,
)
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


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


@dataclass
class CollectiveRecord:
    """A single collective kernel observation from the profile."""

    collective_name: str  # "all_gather_into_tensor", "reduce_scatter_tensor", etc.
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    in_nelems: int  # "In msg nelems" from profile
    out_nelems: int  # "Out msg nelems" from profile
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float


@dataclass
class OpRecord:
    """A single op observation from the profile (any CPU op with GPU kernels)."""

    op_name: str  # normalized name, e.g. "aten::mm", "mylib::my_custom_op"
    input_shapes: tuple[tuple[int, ...], ...]
    dtype: str
    duration_us: float  # sum of all GPU kernels for this CPU op


def _to_nested_tuple(x: Any) -> Any:
    """Recursively convert nested lists to tuples for hashability."""
    if isinstance(x, (list, tuple)):
        return tuple(_to_nested_tuple(i) for i in x)
    return x


@dataclass
class ProfileData:
    """Parse Chrome Trace JSON and build lookup tables for kernel runtimes."""

    collectives: list[CollectiveRecord] = field(default_factory=list)
    ops: list[OpRecord] = field(default_factory=list)
    pg_configs: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Lookup indices built after loading
    _collective_index: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Fallback index by mesh dimension (name, stride, group_size, dtype).
    # Matches PGs belonging to the same mesh dimension regardless of specific ranks.
    # E.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim.
    _collective_index_by_mesh_dim: dict[
        tuple[str, int, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Count of distinct PGs per mesh dimension (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share the same mesh dim).
    _pg_count_by_mesh_dim: dict[tuple[int, int], int] = field(default_factory=dict)
    # Generic op index: (op_name, input_shapes, dtype) -> avg_dur_us
    _op_index: dict[tuple[str, tuple[tuple[int, ...], ...], str], float] = field(
        default_factory=dict
    )
    # Peak observed bandwidth per PG (GB/s), computed from largest messages
    _pg_peak_bw: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Mesh-dimension fallback: (stride, group_size) -> peak BW (GB/s)
    _mesh_dim_peak_bw: dict[tuple[int, int], float] = field(default_factory=dict)

    def load(self, trace_path: str) -> None:
        """Load and parse a Chrome Trace JSON file."""
        import os

        if not os.path.isfile(trace_path):
            raise FileNotFoundError(
                f"PGE trace file not found: {trace_path}. "
                f"Check config.aten_distributed_optimizations.profile_guided_estimations_profile_path"
            )
        with open(trace_path) as f:
            data = json.load(f)

        self._parse_pg_configs(data)
        self._parse_events(data)
        self._build_indices()

        log.info(
            "PGE loaded: %d collectives, %d op records (%d distinct shapes), %d PGs",
            len(self.collectives),
            len(self.ops),
            len(self._op_index),
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

    def _parse_events(self, data: dict[str, Any]) -> None:
        events = data.get("traceEvents", [])
        # Reuse profile_analysis's External id -> CPU op mapping
        try:
            extern_mapping = _create_extern_mapping(data)
        except (KeyError, IndexError, RuntimeError):
            # _create_extern_mapping raises on malformed traces; build manually
            extern_mapping = defaultdict(list)
            for ev in events:
                if (
                    isinstance(ev, dict)
                    and ev.get("cat") == "cpu_op"
                    and "args" in ev
                    and "External id" in ev["args"]
                ):
                    extern_mapping[ev["args"]["External id"]].append(ev)

        # Build External id -> total GPU kernel duration
        gpu_dur: dict[int, float] = defaultdict(float)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            eid = args.get("External id")
            dur = ev.get("dur", 0.0)
            if eid is not None and dur > 0:
                gpu_dur[eid] += dur

        # Parse collectives from GPU kernel events directly
        # (NCCL kernels carry collective metadata in args)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            coll_name = args.get("Collective name")
            if coll_name is None:
                continue
            pg_name = args.get("Process Group Name", "")
            pg_ranks_str = args.get("Process Group Ranks", "")
            group_size = args.get("Group size", 0)
            in_nelems = args.get("In msg nelems", 0)
            out_nelems = args.get("Out msg nelems", 0)
            dtype = args.get("dtype", "")
            dur = ev.get("dur", 0.0)
            if dur <= 0:
                continue

            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    in_nelems=in_nelems,
                    out_nelems=out_nelems,
                    dtype=dtype,
                    duration_us=dur,
                )
            )

        # Parse all CPU ops that have associated GPU kernels
        for eid, cpu_evs in extern_mapping.items():
            if not cpu_evs:
                continue
            total_dur = gpu_dur.get(eid, 0.0)
            if total_dur <= 0:
                continue
            cpu_ev = cpu_evs[0]
            self._parse_op(cpu_ev.get("name", ""), cpu_ev.get("args", {}), total_dur)

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

    def _parse_op(self, name: str, args: dict[str, Any], total_dur: float) -> None:
        """Parse any CPU op into a generic OpRecord."""
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        if not input_dims:
            return
        dtype = input_types[0] if input_types else ""
        # Skip empty entries (non-tensor args like scalars/None) so the shape
        # tuple matches what _get_node_input_shapes extracts from FX nodes.
        shapes = tuple(
            _to_nested_tuple(d)
            for d in input_dims
            if isinstance(d, (list, tuple)) and d
        )
        if not shapes:
            return
        self.ops.append(
            OpRecord(
                op_name=name,
                input_shapes=shapes,
                dtype=dtype,
                duration_us=total_dur,
            )
        )

    def _build_indices(self) -> None:
        """Build lookup indices from parsed records."""
        coll_idx: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        coll_idx_by_mesh_dim: dict[
            tuple[str, int, int, str], list[tuple[int, float]]
        ] = defaultdict(list)
        # Track distinct PG rank sets per mesh dimension for ambiguity check
        pg_sets_by_mesh_dim: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.out_nelems, rec.duration_us)
            )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                coll_idx_by_mesh_dim[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
                pg_sets_by_mesh_dim[(stride, gs)].add(rec.pg_ranks)
        # Sort by nelems for interpolation
        self._collective_index = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx.items()
        }
        self._collective_index_by_mesh_dim = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx_by_mesh_dim.items()
        }
        self._pg_count_by_mesh_dim = {
            k: len(pgs) for k, pgs in pg_sets_by_mesh_dim.items()
        }

        # Generic op index: (op_name, input_shapes, dtype) -> avg_dur_us
        op_groups: dict[tuple[str, tuple[tuple[int, ...], ...], str], list[float]] = (
            defaultdict(list)
        )
        for rec in self.ops:
            key = (rec.op_name, rec.input_shapes, rec.dtype)
            op_groups[key].append(rec.duration_us)
        self._op_index = {k: sum(v) / len(v) for k, v in op_groups.items()}

        # Per-PG peak bandwidth: compute bytes/us for each collective observation,
        # then take the max from the top-N largest messages per PG (where bandwidth
        # is most representative of hardware speed, not dominated by startup latency).
        # Uses output-convention bytes (matching _estimate_with_pg_bandwidth).
        _TOP_N = 5  # consider top N largest messages for peak BW
        pg_bw_samples: dict[tuple[int, ...], list[tuple[int, float]]] = defaultdict(
            list
        )
        mesh_dim_bw_samples: dict[tuple[int, int], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        for rec in self.collectives:
            if rec.out_nelems <= 0 or rec.duration_us <= 0:
                continue
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            elem_bytes = self._dtype_elem_bytes(rec.dtype)
            total_bytes = rec.out_nelems * elem_bytes
            bw_gbps = total_bytes / (rec.duration_us * 1e-6) / 1e9  # GB/s
            pg_bw_samples[rec.pg_ranks].append((total_bytes, bw_gbps))
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                mesh_dim_bw_samples[(stride, gs)].append((total_bytes, bw_gbps))

        def _peak_bw_from_samples(
            samples: list[tuple[int, float]],
        ) -> float:
            """Get peak BW from the top-N largest messages."""
            # Sort by message size descending, take top N, return max BW
            sorted_samples = sorted(samples, key=lambda x: x[0], reverse=True)
            top = sorted_samples[:_TOP_N]
            return max(bw for _, bw in top) if top else 0.0

        self._pg_peak_bw = {
            pg: _peak_bw_from_samples(samples)
            for pg, samples in pg_bw_samples.items()
            if samples
        }
        self._mesh_dim_peak_bw = {
            key: _peak_bw_from_samples(samples)
            for key, samples in mesh_dim_bw_samples.items()
            if samples
        }

    def get_collective_keys(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return the collective index keys: (name, pg_ranks, dtype)."""
        return list(self._collective_index.keys())

    @property
    def op_count(self) -> int:
        """Number of distinct op shapes in the index."""
        return len(self._op_index)

    def get_op_names(self) -> list[str]:
        """Return distinct op names in the op index."""
        return list(OrderedSet(name for name, _, _ in self._op_index))

    @staticmethod
    def _dtype_elem_bytes(dtype: str) -> int:
        """Return bytes per element for a dtype string (NCCL CamelCase or TypeMeta)."""
        return _get_size_from_string(dtype.lower())

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

    # Maximum ratio of target_nelems / max_observed before switching from
    # log-log extrapolation to bandwidth-based estimation.
    EXTRAPOLATION_CAP = 2.0

    def _estimate_with_pg_bandwidth(
        self,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Estimate collective duration using peak observed bandwidth for this PG.

        Used when the target size exceeds the extrapolation cap. Returns ms or None.
        """
        bw_gbps = self._pg_peak_bw.get(pg_ranks)
        if bw_gbps is None or bw_gbps <= 0:
            # Try mesh-dimension fallback
            stride = _rank_stride(pg_ranks)
            gs = len(pg_ranks)
            if stride is not None:
                bw_gbps = self._mesh_dim_peak_bw.get((stride, gs))
        if bw_gbps is None or bw_gbps <= 0:
            return None  # fall through to analytical
        elem_bytes = self._dtype_elem_bytes(dtype)
        total_bytes = nelems * elem_bytes
        dur_ms = total_bytes / (bw_gbps * 1e6)  # GB/s → bytes/ms = 1e6
        return dur_ms

    def lookup_collective(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> tuple[float, str] | None:
        """Look up collective duration in ms. Returns (duration_ms, source) or None.

        ``source`` is ``"profile"`` for exact/interpolated matches, or
        ``"pg_bandwidth"`` when bandwidth-based extrapolation was used.

        Tries exact rank match first, then falls back to mesh-dimension match
        (e.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim).

        When the target size exceeds EXTRAPOLATION_CAP * max_observed, uses
        bandwidth-based estimation from peak observed bandwidth instead of
        linear extrapolation (which overestimates for large messages).
        """
        norm_name = self._normalize_collective_name(collective_name)
        # Try exact rank match first
        key = (norm_name, pg_ranks, dtype)
        entries = self._collective_index.get(key)
        if not entries:
            # Fall back to mesh-dimension match
            gs = len(pg_ranks)
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
            ):
                mesh_dim_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_mesh_dim.get(mesh_dim_key)
            if not entries:
                return None

        # Exact match
        for n, dur in entries:
            if n == nelems:
                return (dur / 1e3, "profile")  # us -> ms

        # Check extrapolation distance: if target is far beyond observed range,
        # use bandwidth-based model instead of log-log extrapolation
        max_observed = max((n for n, _ in entries if n > 0), default=0)
        if max_observed > 0 and nelems > max_observed * self.EXTRAPOLATION_CAP:
            est = self._estimate_with_pg_bandwidth(pg_ranks, nelems, dtype)
            if est is not None:
                return (est, "pg_bandwidth")
            # Fall through to log-log if no BW data available

        # Interpolation in log-log space
        result = self._interpolate_log_log(entries, nelems)
        if result is not None:
            return (result, "profile")
        return None

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
            # Linear extrapolation (not log-log) from nearest lower;
            # EXTRAPOLATION_CAP in lookup_collective limits how far this reaches.
            return (lower[1] * target_nelems / lower[0]) / 1e3
        elif upper is not None:
            # Linear extrapolation from nearest upper
            return (upper[1] * target_nelems / upper[0]) / 1e3

        return None

    def lookup_op(
        self,
        op_name: str,
        input_shapes: tuple[tuple[int, ...], ...],
        dtype: str,
    ) -> float | None:
        """Look up op duration in ms by exact shape match. Returns None on miss."""
        key = (op_name, input_shapes, dtype)
        dur_us = self._op_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        return None


# Mapping from torch.dtype to C++ caffe2::TypeMeta::name() string, as emitted
# by the Kineto profiler in the "Input type" field of Chrome Trace JSON events.
# Related: profile_analysis._dtype_map provides the inverse for a subset of types.
_DTYPE_TO_TYPEMETA_STR: dict[torch.dtype, str] = {
    torch.float32: "float",
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.float64: "double",
    torch.int32: "int",
    torch.int64: "long int",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
}

# Mapping from torch.dtype to the NCCL dtype string used in collective kernel
# events (e.g. "Float", "BFloat16"). Different from TypeMeta names above.
_DTYPE_TO_NCCL_STR: dict[torch.dtype, str] = {
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.int8: "Char",
    torch.uint8: "Byte",
}


def _dtype_to_profile_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to NCCL dtype string (for collective matching)."""
    return _DTYPE_TO_NCCL_STR.get(dtype, str(dtype))


def _dtype_to_typemeta_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to TypeMeta name string (for op matching)."""
    return _DTYPE_TO_TYPEMETA_STR.get(dtype, str(dtype))


def _get_node_dtype_str(node: fx.Node) -> str:
    """Extract dtype as TypeMeta name string from FX node metadata (for op matching)."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return _dtype_to_typemeta_str(val.dtype)
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return _dtype_to_typemeta_str(first.dtype)
    return ""


def _fx_target_to_profile_name(node: fx.Node) -> str | None:
    """Convert FX node target to the profile op name format.

    FX: torch.ops.aten.mm.default → "aten::mm"
    FX: torch.ops.deepep.dispatch.default → "deepep::dispatch"
    """
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        # e.g. "aten::mm" from torch.ops.aten.mm.default
        ns = target.namespace
        op_name = target._schema.name.split("::")[-1]
        return f"{ns}::{op_name}"
    if hasattr(target, "__name__"):
        return target.__name__
    return None


def _get_node_input_shapes(
    node: fx.Node,
) -> tuple[tuple[int, ...], ...] | None:
    """Extract input shapes from FX node args, matching profile's Input Dims format."""
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint

    shapes: list[tuple[int, ...]] = []
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            continue
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            resolved = []
            for s in val.shape:
                h = get_hint(s)
                if h is None:
                    log.debug(
                        "PGE: unresolved symbolic dim in %s shape %s",
                        node.name,
                        val.shape,
                    )
                    return None
                resolved.append(h)
            shapes.append(tuple(resolved))
    if not shapes:
        return None
    return tuple(shapes)


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    return (
        is_all_gather_into_tensor(node)
        or is_reduce_scatter_tensor(node)
        or is_all_reduce_tensor(node)
        or is_all_to_all_tensor(node)
    )


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], int, str] | None:
    """Extract (collective_name, pg_ranks, nelems, dtype) from collective node."""
    import torch.distributed as c10d
    from torch.fx.operator_schemas import normalize_function

    if not c10d.is_initialized():
        return None

    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return None
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

    try:
        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))
    except (RuntimeError, KeyError, ValueError):
        log.debug(
            "PGE: failed to resolve process group for %s", node.name, exc_info=True
        )
        return None

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


class ProfileGuidedEstimator:
    """Profile-guided runtime estimator for FX nodes.

    Implements the ``custom_runtime_estimation`` interface:
    ``(fx.Node, int | None) -> float | None`` (returns ms or None for fallback).

    Handles collectives via interpolation (latency + bandwidth model) and all
    other ops (matmul, SDPA, custom ops, etc.) via exact shape match from the
    profile trace.
    """

    def __init__(self, trace_path: str) -> None:
        self.profile = ProfileData()
        self.estimation_log: list[dict[str, Any]] = []
        self.miss_log: list[dict[str, Any]] = []
        self.profile.load(trace_path)
        self._log_profile_stats(trace_path)

    def _log_profile_stats(self, trace_path: str) -> None:
        """Log profile stats to trace_structured for tlparse."""
        profile = self.profile
        coll_keys = profile.get_collective_keys()
        op_count = profile.op_count
        op_names = profile.get_op_names()
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pge_profile_loaded",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {
                    "path": trace_path,
                    "collective_keys": [
                        {
                            "name": k[0],
                            "ranks": list(k[1]),
                            "group_size": len(k[1]),
                            "stride": _rank_stride(k[1]),
                            "dtype": k[2],
                        }
                        for k in coll_keys
                    ],
                    "op_count": op_count,
                    "op_names": op_names,
                }
            ),
        )
        log.info("PGE: using profile-guided estimation from %s", trace_path)

    def __call__(self, node: fx.Node, override_size: int | None = None) -> float | None:
        if _is_collective_node(node):
            return self._estimate_collective(node, override_size)
        return self._estimate_op(node)

    def _estimate_collective(
        self, node: fx.Node, override_size: int | None
    ) -> float | None:
        info = _get_collective_info(node)
        if info is None:
            self.miss_log.append(
                {
                    "node": node.name,
                    "target": str(node.target),
                    "reason": "get_collective_info returned None",
                }
            )
            return None
        coll_name, pg_ranks, nelems, dtype = info
        val = node.meta.get("val")
        if override_size is not None:
            if override_size == 0:
                return None  # no profile data for zero-size; fall back to analytical
            if isinstance(val, torch.Tensor):
                elem_size = val.element_size()
                if elem_size > 0:
                    nelems = override_size // elem_size
            else:
                log.debug(
                    "PGE: override_size=%d but val is not a tensor for %s",
                    override_size,
                    node.name,
                )
        dtype_bytes = val.element_size() if isinstance(val, torch.Tensor) else 0
        result = self.profile.lookup_collective(coll_name, pg_ranks, nelems, dtype)
        if result is not None:
            est, source = result
            self.estimation_log.append(
                {
                    "node": node.name,
                    "op": coll_name,
                    "nelems": nelems,
                    "dtype": dtype,
                    "dtype_bytes": dtype_bytes,
                    "group_size": len(pg_ranks),
                    "stride": _rank_stride(pg_ranks),
                    "pge_ms": est,
                    "source": source,
                }
            )
            return est
        self.miss_log.append(
            {
                "node": node.name,
                "op": coll_name,
                "nelems": nelems,
                "dtype": dtype,
                "group_size": len(pg_ranks),
                "reason": "no match in profile",
            }
        )
        return None

    def _estimate_op(self, node: fx.Node) -> float | None:
        """Estimate any non-collective op via exact shape match in profile."""
        profile_name = _fx_target_to_profile_name(node)
        if profile_name is None:
            return None
        input_shapes = _get_node_input_shapes(node)
        if input_shapes is None:
            return None
        dtype = _get_node_dtype_str(node)
        est = self.profile.lookup_op(profile_name, input_shapes, dtype)
        if est is not None:
            self.estimation_log.append(
                {
                    "node": node.name,
                    "op": profile_name,
                    "shapes": [list(s) for s in input_shapes],
                    "dtype": dtype,
                    "pge_ms": est,
                    "source": "profile",
                }
            )
        else:
            self.miss_log.append(
                {
                    "node": node.name,
                    "op": profile_name,
                    "shapes": [list(s) for s in input_shapes],
                    "dtype": dtype,
                    "reason": "no match in profile",
                }
            )
        return est

    def log_diagnostics(
        self,
        gm: fx.GraphModule,
    ) -> None:
        """Log PGE vs analytical comparison to trace_structured for tlparse."""
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _collect_analytical_estimates,
        )

        analytical = _collect_analytical_estimates(gm, self)
        log_pge_estimations(self, analytical)


def log_pge_estimations(
    estimator: ProfileGuidedEstimator,
    analytical_estimates: dict[str, float] | None = None,
) -> None:
    """Dump PGE estimation results via trace_structured for tlparse."""
    rows = []
    for entry in estimator.estimation_log:
        row = dict(entry)
        node_name = entry.get("node", "")
        if analytical_estimates and node_name in analytical_estimates:
            analytical_ms = analytical_estimates[node_name]
            row["analytical_ms"] = analytical_ms
            pge_ms = entry.get("pge_ms", 0)
            if analytical_ms > 0:
                row["pge_vs_analytical_pct"] = round(
                    (pge_ms - analytical_ms) / analytical_ms * 100, 1
                )
        rows.append(row)

    table = _format_pge_table(rows, estimator.miss_log)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pge_estimations_table",
            "encoding": "string",
        },
        payload_fn=lambda: table,
    )

    log.info(
        "PGE: %d estimations, %d misses logged to trace_structured",
        len(rows),
        len(estimator.miss_log),
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


def _format_pge_table(
    rows: list[dict[str, Any]],
    miss_log: list[dict[str, Any]] | None = None,
) -> str:
    """Format PGE estimations + misses as a single aligned text table."""
    misses: list[dict[str, Any]] = list(miss_log) if miss_log is not None else []
    lines: list[str] = []
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        lines.append(f"GPU: {gpu_name} x{gpu_count}")
    except (RuntimeError, AssertionError):
        lines.append("GPU: unknown")
    lines.append("")

    has_analytical = any("analytical_ms" in r for r in rows)

    if has_analytical:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3}"
            f" {'pge_ms':>10} {'analytical_ms':>15} {'diff%':>8} {'':>3}"
            f" {'pge_GB/s':>10} {'analytical_GB/s':>15}"
        )
    else:
        header = (
            f"{'node':<45} {'op':<30} {'size':>8} {'gs':>4} {'st':>3} {'pge_ms':>10}"
        )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        node = row.get("node", "")[:44]
        op = row.get("op", "")[:29]
        gs = row.get("group_size", "")
        stride = row.get("stride", "")
        stride_str = str(stride) if stride is not None else "-"
        pge_ms = row.get("pge_ms", 0)
        dtype_bytes = row.get("dtype_bytes", 0)
        n = row.get("nelems", 0) if isinstance(row.get("nelems"), int) else 0
        size_str = _format_bytes(n * dtype_bytes) if dtype_bytes > 0 and n > 0 else "-"

        if has_analytical:
            anal_ms = row.get("analytical_ms", None)
            diff_pct = row.get("pge_vs_analytical_pct", None)
            anal_str = f"{anal_ms:.4f}" if anal_ms is not None else "-"
            diff_str = f"{diff_pct:+.1f}%" if diff_pct is not None else "-"
            flag = ""
            if diff_pct is not None:
                adp = abs(diff_pct)
                if adp > 50:
                    flag = "***"
                elif adp > 15:
                    flag = "**"
            pge_bw_str = ""
            anal_bw_str = ""
            if dtype_bytes > 0 and n > 0:
                data_bytes = n * dtype_bytes
                if pge_ms > 0:
                    pge_bw = data_bytes / (pge_ms * 1e-3) / 1e9
                    pge_bw_str = f"{pge_bw:.1f}"
                if anal_ms is not None and anal_ms > 0:
                    anal_bw = data_bytes / (anal_ms * 1e-3) / 1e9
                    anal_bw_str = f"{anal_bw:.1f}"
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pge_ms:>10.4f} {anal_str:>15} {diff_str:>8} {flag:>3}"
                f" {pge_bw_str:>10} {anal_bw_str:>15}"
            )
        else:
            line = (
                f"{node:<45} {op:<30} {size_str:>8} {gs:>4} {stride_str:>3}"
                f" {pge_ms:>10.4f}"
            )
        lines.append(line.rstrip())

    lines.append("")
    lines.append(f"Total: {len(rows)} estimations")
    if has_analytical:
        f1 = sum(
            1
            for r in rows
            if r.get("pge_vs_analytical_pct") is not None
            and abs(r["pge_vs_analytical_pct"]) > 50
        )
        f2 = sum(
            1
            for r in rows
            if r.get("pge_vs_analytical_pct") is not None
            and 15 < abs(r["pge_vs_analytical_pct"]) <= 50
        )
        lines.append(f"Flagged: {f2} ** (>15%), {f1} *** (>50%)")

    if misses:
        lines.append("")
        lines.append(f"=== MISSES ({len(misses)}) ===")
        miss_header = f"{'node':<45} {'op':<30} {'reason'}"
        lines.append(miss_header)
        lines.append("-" * len(miss_header))
        for m in misses:
            node = m.get("node", "")[:44]
            op = (m.get("op") or m.get("target") or "")[:29]
            reason = m.get("reason", "")
            shapes = m.get("shapes")
            shape_info = m.get("shape")
            extra = ""
            if shapes:
                extra = f" shapes={shapes}"
            elif shape_info:
                extra = f" shape={shape_info}"
            lines.append(f"{node:<45} {op:<30} {reason}{extra}".rstrip())

    return "\n".join(lines)
