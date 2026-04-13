from __future__ import annotations

import logging
import warnings

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def is_compute_bound_node(n: torch.fx.Node) -> bool:
    """True if n is a compute-bound op (matmul, conv, attention/SDPA).

    Checks whether the op has a registered FLOP formula — only
    compute-intensive ops (mm, bmm, addmm, conv, sdpa, etc.) do.
    """
    from torch.utils.flop_counter import flop_registry

    return getattr(n.target, "overloadpacket", None) in flop_registry


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

    AG_TARGETS = OrderedSet(
        [
            c10d.all_gather_into_tensor.default,
            c10d.all_gather_into_tensor_out.default,
        ]
    )
    RS_TARGETS = OrderedSet(
        [
            c10d.reduce_scatter_tensor.default,
            c10d.reduce_scatter_tensor_out.default,
        ]
    )

    _enabled_groups: OrderedSet[str] = OrderedSet()
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

        # In auto mode, only replace collectives overlapping compute-bound ops
        if mode is None:
            meta_overlap = node.meta.get("has_compute_bound_overlap")
            has_overlap = meta_overlap
            if has_overlap is None:
                has_overlap = _has_compute_bound_overlap(node, graph)
            log.debug(
                "LC overlap %s %s: meta=%s fallback=%s",
                coll_type,
                node.name,
                meta_overlap,
                has_overlap,
            )
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
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        log.debug(
            "LC overlap %s: no wait_tensor found. "
            "target=%s, args=%s, users=%s",
            start_node.name,
            start_node.target,
            [(a.name if isinstance(a, torch.fx.Node) else a) for a in start_node.args],
            [
                (u.name, u.target)
                for u in start_node.users
            ],
        )
        return False

    node_positions = {node: i for i, node in enumerate(graph.nodes)}
    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]
    nodes_between = wait_pos - start_pos - 1

    for node in graph.nodes:
        pos = node_positions[node]
        if pos <= start_pos or pos >= wait_pos:
            continue
        if is_compute_bound_node(node):
            log.debug(
                "LC overlap %s: found compute %s (%d nodes between start/wait)",
                start_node.name,
                node.target,
                nodes_between,
            )
            return True
    log.debug(
        "LC overlap %s: no compute-bound ops (%d nodes between start/wait)",
        start_node.name,
        nodes_between,
    )
    return False


_WAIT_TARGETS: OrderedSet | None = None


def _get_wait_targets():
    global _WAIT_TARGETS
    if _WAIT_TARGETS is None:
        _WAIT_TARGETS = OrderedSet([torch.ops._c10d_functional.wait_tensor.default])
        try:
            _WAIT_TARGETS.add(torch.ops.c10d_functional.wait_tensor.default)
        except AttributeError:
            pass
    return _WAIT_TARGETS


def _is_wait_tensor(node):
    """Check if node is a wait_tensor op (direct or wrapped in ControlDeps)."""
    if node.op != "call_function":
        return False
    # Direct wait_tensor
    if node.target in _get_wait_targets():
        return True
    # ControlDeps-wrapped wait_tensor (from TBB manual scheduling):
    # control_deps(deps, subgraph, *args) where subgraph wraps wait_tensor
    if _is_control_deps_wrapping_wait(node):
        return True
    return False


def _is_control_deps_wrapping_wait(node):
    """Check if a ControlDeps node wraps a wait_tensor op."""
    from torch._inductor.fx_passes.control_dependencies import ControlDeps

    if not isinstance(node.target, ControlDeps):
        return False
    # Check subgraph (args[1] is a GraphModule containing the wrapped op)
    if len(node.args) >= 2:
        subgraph = node.args[1]
        if isinstance(subgraph, torch.fx.GraphModule):
            for n in subgraph.graph.nodes:
                if n.op == "call_function" and n.target in _get_wait_targets():
                    return True
    # Fallback: check if the node name indicates wait_tensor
    if "wait_tensor" in node.name:
        return True
    return False


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
