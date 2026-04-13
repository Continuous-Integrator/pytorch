from __future__ import annotations

import logging
import warnings

import torch

log = logging.getLogger(__name__)


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
    mode: bool | None = None,
) -> None:
    """Replace FSDP collectives with copy-engine symm_mem variants.

    mode:
        True  — replace all FSDP collectives (force).
        False — don't replace any.
        None  — per-collective: replace only those overlapping compute-bound ops
                (matmul, conv, attention) and above the minimum size threshold.
    """
    if mode is False:
        return

    c10d = torch.ops._c10d_functional
    symm_mem = torch.ops.symm_mem

    AG_TARGETS = {
        c10d.all_gather_into_tensor.default,
        c10d.all_gather_into_tensor_out.default,
    }
    RS_TARGETS = {
        c10d.reduce_scatter_tensor.default,
        c10d.reduce_scatter_tensor_out.default,
    }

    _enabled_groups: set[str] = set()
    collectives = []

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        is_ag = node.target in AG_TARGETS
        is_rs = node.target in RS_TARGETS
        if not is_ag and not is_rs:
            continue
        collectives.append((node, is_ag))

    if not collectives:
        return

    from torch._inductor import config

    min_bytes = config.aten_distributed_optimizations.low_contention_min_bytes_per_rank

    replacements = 0
    skipped_no_overlap = 0
    skipped_small = 0
    for node, is_ag in collectives:
        # Size filter: LC barrier overhead dominates for small messages
        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                continue

        # In auto mode, only replace collectives overlapping compute-bound ops
        if mode is None:
            has_overlap = node.meta.get("has_compute_bound_overlap")
            if has_overlap is None:
                has_overlap = _has_compute_bound_overlap(node, graph)
            if not has_overlap:
                skipped_no_overlap += 1
                continue

        if is_ag:
            _replace_all_gather(node, graph, symm_mem, _enabled_groups)
        else:
            _replace_reduce_scatter(node, graph, symm_mem, _enabled_groups)
        replacements += 1

    log.info(
        "Replaced %d/%d FSDP collectives "
        "(skipped_no_overlap=%d, skipped_small=%d, min_bytes=%d)",
        replacements,
        len(collectives),
        skipped_no_overlap,
        skipped_small,
        min_bytes,
    )


def _replace_all_gather(node, graph, symm_mem, enabled_groups):
    input_node = node.args[0]
    group_name = node.args[2]
    _ensure_symm_mem_for_group(group_name, enabled_groups)
    with graph.inserting_before(node):
        new_node = graph.call_function(
            symm_mem._low_contention_all_gather.default,
            args=(input_node, group_name),
        )
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _replace_reduce_scatter(node, graph, symm_mem, enabled_groups):
    input_node = node.args[0]
    reduce_op = node.args[1]
    group_name = node.args[3]
    _ensure_symm_mem_for_group(group_name, enabled_groups)
    with graph.inserting_before(node):
        new_node = graph.call_function(
            symm_mem._low_contention_reduce_scatter.default,
            args=(input_node, reduce_op, group_name),
        )
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

    wait_node = None
    for user in start_node.users:
        if _is_wait_tensor(user):
            wait_node = user
            break
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


def _is_wait_tensor(node):
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.wait_tensor.default
    )


def _ensure_symm_mem_for_group(group_name, enabled_groups):
    if group_name in enabled_groups:
        return
    enabled_groups.add(group_name)
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
