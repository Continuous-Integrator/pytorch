"""Layout canonicalization passes."""

import operator

import torch
from torch._torchlite.passes.common import (
    _set_phase,
    AddLayerNormKernel,
    AddRmsNormKernel,
    FusedKernel,
    MatmulEpilogueKernel,
    PassResult,
)
from torch.fx import GraphModule, Node

_VIEWLIKE_TARGETS = frozenset(
    {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.Tensor.reshape,
        torch.Tensor.view,
    }
)

_CONTIGUOUS_PRODUCERS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.full.default,
        torch.repeat_interleave,
    }
)


def _is_contiguous_producer(node: Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in _CONTIGUOUS_PRODUCERS:
        return True
    return isinstance(
        node.target,
        (FusedKernel, MatmulEpilogueKernel, AddRmsNormKernel, AddLayerNormKernel),
    )


def _is_viewlike_target(target) -> bool:
    return target in _VIEWLIKE_TARGETS


def _is_simple_slice_view(node: Node) -> bool:
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    if len(node.args) != 2:
        return False
    base, index = node.args
    if not isinstance(base, Node) or not _is_contiguous_producer(base):
        return False
    if not isinstance(index, tuple):
        return False
    return all(isinstance(dim, (int, slice)) for dim in index)


def canonicalize_layouts(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Remove redundant layout ops and collapse stacked view chains."""

    del example_inputs

    graph = gm.graph
    changed = False
    local_changed = True

    while local_changed:
        local_changed = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            src = node.args[0] if node.args else None

            if node.target == torch.ops.aten.contiguous.default:
                if not isinstance(src, Node):
                    continue
                if not _is_contiguous_producer(src):
                    continue

                src_shape = src.meta.get("shape")
                if src_shape is not None:
                    src.meta["shape"] = list(src_shape)
                src.meta["dtype"] = src.meta.get(
                    "dtype",
                    node.meta.get("dtype", torch.float32),
                )
                _set_phase(
                    src,
                    src.meta.get("phase", node.meta.get("phase", "forward")),
                )

                node.replace_all_uses_with(src)
                graph.erase_node(node)
                local_changed = True
                changed = True
                continue

            if not _is_viewlike_target(node.target) or not isinstance(src, Node):
                continue

            src_shape = src.meta.get("shape")
            node_shape = node.meta.get("shape")
            if (
                src_shape is not None
                and node_shape is not None
                and list(src_shape) == list(node_shape)
            ):
                src.meta["dtype"] = src.meta.get(
                    "dtype",
                    node.meta.get("dtype", torch.float32),
                )
                _set_phase(
                    src,
                    src.meta.get("phase", node.meta.get("phase", "forward")),
                )
                node.replace_all_uses_with(src)
                graph.erase_node(node)
                local_changed = True
                changed = True
                continue

            if (
                src.op == "call_function"
                and _is_viewlike_target(src.target)
                and len(src.users) == 1
                and src.args
                and isinstance(src.args[0], Node)
                and (
                    _is_contiguous_producer(src.args[0])
                    or _is_simple_slice_view(src.args[0])
                )
            ):
                node.replace_input_with(src, src.args[0])
                if not src.users:
                    graph.erase_node(src)
                local_changed = True
                changed = True
                continue

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)
