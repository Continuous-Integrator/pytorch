from __future__ import annotations

import logging
import warnings

import torch

log = logging.getLogger(__name__)


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
    mode: bool | None = None,
    ag_impl: str = "low_contention",
) -> None:
    """Replace FSDP collectives with symm_mem variants.

    ag_impl: "low_contention" (copy engine P2P) or "multimem" (multicast store).
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

    # In auto mode, decide per-graph: replace all or none.
    # Mixing NCCL and LC collectives in the same graph causes scheduling
    # conflicts that are worse than either pure approach.
    from torch._inductor import config

    if mode is None:
        threshold = config.aten_distributed_optimizations.low_contention_auto_threshold
        # Use weighted overlap: each collective contributes its overlap ratio
        # (0.0 = fully exposed, 1.0 = fully hidden). Falls back to binary
        # (0 or 1) when overlap_scheduling didn't annotate ratios.
        weighted_hidden = 0.0
        for node, _ in collectives:
            ratio = node.meta.get("compute_overlap_ratio")
            if ratio is not None:
                weighted_hidden += ratio
            else:
                has_overlap = node.meta.get("has_compute_overlap")
                if has_overlap is None:
                    has_overlap = _has_compute_overlap(node, graph)
                weighted_hidden += 1.0 if has_overlap else 0.0
        hidden_ratio = weighted_hidden / len(collectives)
        # Replace all if weighted hidden ratio exceeds threshold.
        # Otherwise keep all as NCCL to avoid mixing penalty.
        if hidden_ratio <= threshold:
            log.info(
                "LC auto: skipping graph (%.1f/%d hidden, ratio=%.1f%% <= %.0f%%)",
                weighted_hidden,
                len(collectives),
                hidden_ratio * 100,
                threshold * 100,
            )
            return
        log.info(
            "LC auto: replacing all %d collectives (%.1f/%d hidden, ratio=%.1f%% > %.0f%%)",
            len(collectives),
            weighted_hidden,
            len(collectives),
            hidden_ratio * 100,
            threshold * 100,
        )

    min_bytes = config.aten_distributed_optimizations.low_contention_min_bytes_per_rank

    replacements = 0
    skipped_small = 0
    for node, is_ag in collectives:
        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                continue
        if is_ag:
            _replace_all_gather(node, graph, symm_mem, _enabled_groups, ag_impl)
        else:
            _replace_reduce_scatter(node, graph, symm_mem, _enabled_groups)
        replacements += 1

    log.info(
        "Replaced %d/%d FSDP collectives (ag_impl=%s, skipped_small=%d, min_bytes=%d)",
        replacements,
        len(collectives),
        ag_impl,
        skipped_small,
        min_bytes,
    )


def _replace_all_gather(node, graph, symm_mem, enabled_groups, ag_impl="low_contention"):
    input_node = node.args[0]
    group_name = node.args[2]
    _ensure_symm_mem_for_group(group_name, enabled_groups)

    if ag_impl == "multimem":
        _replace_all_gather_multimem(node, graph, symm_mem, input_node, group_name)
    elif ag_impl == "multimem_inplace":
        _replace_all_gather_multimem(
            node, graph, symm_mem, input_node, group_name, inplace=True
        )
    else:
        with graph.inserting_before(node):
            new_node = graph.call_function(
                symm_mem._low_contention_all_gather.default,
                args=(input_node, group_name),
            )
        new_node.meta.update(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def _replace_all_gather_multimem(
    node, graph, symm_mem, input_node, group_name, inplace=False
):
    """Replace all_gather with multimem variant."""
    target = (
        symm_mem._multimem_all_gather_inplace.default
        if inplace
        else symm_mem._multimem_all_gather.default
    )
    with graph.inserting_before(node):
        new_node = graph.call_function(
            target,
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


def _has_compute_overlap(start_node, graph):
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
