from __future__ import annotations

import logging
import warnings

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _get_collective_info(node):
    """Return (is_ag, group_name) if node is an AG/RS collective, else None."""
    from torch._inductor.fx_passes.bucketing import (
        is_all_gather_into_tensor,
        is_reduce_scatter_tensor,
    )
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name

    if is_all_gather_into_tensor(node):
        return True, get_group_name(node)
    if is_reduce_scatter_tensor(node):
        return False, get_group_name(node)
    return None


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
) -> None:
    """Replace FSDP collectives with copy-engine symm_mem variants."""
    symm_mem = torch.ops.symm_mem

    collectives = []
    groups: OrderedSet[str] = OrderedSet()
    for node in list(graph.nodes):
        info = _get_collective_info(node)
        if info is None:
            continue
        is_ag, group_name = info
        collectives.append((node, is_ag, group_name))
        groups.add(group_name)

    if not collectives:
        return

    # Enable symmetric memory for all groups found in collectives.
    # Some group names (e.g. DTensor mesh-derived "mesh_get_process_group")
    # can't be resolved at compile time — skip those groups.
    valid_groups: OrderedSet[str] = OrderedSet()
    for group_name in groups:
        if _enable_symm_mem(group_name):
            valid_groups.add(group_name)

    # Filter to collectives whose groups we can actually resolve
    collectives = [
        (node, is_ag, gn) for node, is_ag, gn in collectives if gn in valid_groups
    ]
    if not collectives:
        return

    from torch._inductor import config

    min_bytes = config.aten_distributed_optimizations.low_contention_min_bytes_per_rank

    replacements = 0
    skipped_small = 0
    skipped_no_overlap = 0
    skipped_nvlink_contention = 0
    for node, is_ag, group_name in collectives:
        coll_type = "AG" if is_ag else "RS"

        # Size filter: LC barrier overhead dominates for small messages
        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                log.debug(
                    "LC skip %s %s: size %d < min_bytes %d",
                    coll_type,
                    node.name,
                    per_rank_bytes,
                    min_bytes,
                )
                continue

        # Only replace collectives that overlap compute-bound ops
        # (matmul, conv, attention). LC adds barrier overhead, so replacing
        # a collective on the critical path with no compute to hide behind
        # would add latency for no benefit.
        if not _has_compute_bound_overlap(node, graph):
            skipped_no_overlap += 1
            log.debug("LC skip %s %s: no compute-bound overlap", coll_type, node.name)
            continue

        # Skip if other groups' NCCL collectives (e.g. TP) overlap,
        # to avoid NVLink contention between LC and NCCL traffic
        if _has_other_group_collectives(node, group_name, graph):
            skipped_nvlink_contention += 1
            log.debug(
                "LC skip %s %s: overlaps other-group collectives (NVLink contention)",
                coll_type,
                node.name,
            )
            continue

        _replace_collective(node, graph, symm_mem, is_ag, group_name)
        replacements += 1

    log.info(
        "Replaced %d/%d FSDP collectives "
        "(skipped_small=%d, skipped_no_overlap=%d, "
        "skipped_nvlink_contention=%d, min_bytes=%d)",
        replacements,
        len(collectives),
        skipped_small,
        skipped_no_overlap,
        skipped_nvlink_contention,
        min_bytes,
    )


def _enable_symm_mem(group_name):
    """Try to enable symmetric memory for a group. Returns True on success."""
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if is_symm_mem_enabled_for_group(group_name):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
        return True
    except (TypeError, RuntimeError) as e:
        log.debug("LC: cannot enable symm_mem for group %s: %s", group_name, e)  # noqa: G200
        return False


def _replace_collective(node, graph, symm_mem, is_ag, group_name):
    input_node = node.args[0]
    if is_ag:
        target = symm_mem._low_contention_all_gather.default
        args = (input_node, group_name)
    else:
        reduce_op = node.args[1]
        target = symm_mem._low_contention_reduce_scatter.default
        args = (input_node, reduce_op, group_name)

    with graph.inserting_before(node):
        new_node = graph.call_function(target, args=args)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _get_per_rank_bytes(node):
    """Return per-rank message bytes for a collective, or None if unknown."""
    input_val = node.args[0].meta.get("val") if node.args else None
    if not isinstance(input_val, torch.Tensor):
        return None
    return input_val.nelement() * input_val.element_size()


def _has_compute_bound_overlap(start_node, graph):
    """Check if compute-bound ops (matmul, conv, attention) exist between
    the collective start and its wait in topological order."""
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    node_positions = {node: i for i, node in enumerate(graph.nodes)}
    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]

    for node in graph.nodes:
        pos = node_positions[node]
        if pos <= start_pos or pos >= wait_pos:
            continue
        if is_compute_node(node):
            return True
    return False


def _has_other_group_collectives(start_node, group_name, graph):
    """Check if collectives from other groups exist between start and wait.

    If a different group's NCCL collective (e.g. TP all-reduce) runs
    concurrently with our LC collective, they compete for NVLink bandwidth.
    """
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    node_positions = {node: i for i, node in enumerate(graph.nodes)}
    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]

    for node in graph.nodes:
        pos = node_positions[node]
        if pos <= start_pos or pos >= wait_pos:
            continue
        info = _get_collective_info(node)
        if info is not None:
            _, other_group = info
            if other_group != group_name:
                log.debug(
                    "LC contention %s: found %s (group %s) between start/wait",
                    start_node.name,
                    node.name,
                    other_group,
                )
                return True
    return False


def _is_wait_tensor(node):
    """Check if node is a wait_tensor op (direct or wrapped in ControlDeps)."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops._c10d_functional.wait_tensor.default:
        return True
    # Handles public namespace (c10d_functional.wait_tensor) and
    # ControlDeps-wrapped wait_tensor (from TBB manual scheduling)
    return "wait_tensor" in node.name


def _find_wait_for_collective(start_node):
    """Find the wait_tensor node for a collective.

    Handles multiple graph patterns:
    1. Direct: start -> wait_tensor(start)
    2. _out variant: start(out=buf) -> wait_tensor(buf)
    3. ControlDeps-wrapped: start -> control_deps(wait_tensor_subgraph, start)
    """
    for user in start_node.users:
        if _is_wait_tensor(user):
            return user

    # For _out variants, check users of the out-buffer argument.
    c10d = torch.ops._c10d_functional
    out_arg_idx = None
    if start_node.target is c10d.all_gather_into_tensor_out.default:
        out_arg_idx = 3
    elif start_node.target is c10d.reduce_scatter_tensor_out.default:
        out_arg_idx = 4

    if out_arg_idx is not None and len(start_node.args) > out_arg_idx:
        out_buf = start_node.args[out_arg_idx]
        if isinstance(out_buf, torch.fx.Node):
            for user in out_buf.users:
                if _is_wait_tensor(user):
                    return user

    return None
