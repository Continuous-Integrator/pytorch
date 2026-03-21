"""Training-specific graph cleanup passes."""

from collections.abc import Hashable

import torch
from torch.fx import GraphModule, Node

from torch._torchlite.ops import (
    _load_rng_state,
    _save_for_backward,
    _save_rng_state,
    adamw_step,
    param_update,
    sgd_step,
)
from torch._torchlite.passes.common import PassResult
from torch._torchlite.passes.decompose import (
    _manual_decomp_sigmoid_backward,
    _manual_decomp_tanh_backward,
    _manual_decomp_threshold_backward,
)

_BACKWARD_DECOMP_TARGETS = {
    torch.ops.aten.threshold_backward.default: _manual_decomp_threshold_backward,
    torch.ops.aten.sigmoid_backward.default: _manual_decomp_sigmoid_backward,
    torch.ops.aten.tanh_backward.default: _manual_decomp_tanh_backward,
}

_NON_CSE_TARGETS = {
    _save_for_backward,
    _save_rng_state,
    _load_rng_state,
    sgd_step,
    param_update,
    adamw_step,
    torch.Tensor.copy_,
}

_NON_CSE_TARGET_NAMES = frozenset(
    {
        "dropout",
        "native_dropout",
        "bernoulli",
        "rand",
        "rand_like",
        "randn",
        "randn_like",
    }
)


def decompose_training_backward(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Lower training-only backward pointwise ops to fuse-friendly ATen.

    This pass runs after autograd_per_op.  Its job is intentionally narrow:
    decompose backward ops such as threshold_backward into simple pointwise
    expressions so matmul_epilogue/fuse can absorb them into surrounding GEMMs.
    """

    del example_inputs

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.meta.get("phase", "forward") != "backward":
            continue
        manual_fn = _BACKWARD_DECOMP_TARGETS.get(node.target)
        if manual_fn is None:
            continue
        manual_fn(graph, node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)


def _phase_key(node: Node) -> str:
    return node.meta.get("phase", "forward")


def _freeze_arg(arg, replacements):
    if isinstance(arg, Node):
        return ("node", replacements.get(arg, arg))
    if isinstance(arg, slice):
        return ("slice", arg.start, arg.stop, arg.step)
    if isinstance(arg, (tuple, list)):
        return tuple(_freeze_arg(a, replacements) for a in arg)
    if isinstance(arg, dict):
        return tuple(
            sorted((k, _freeze_arg(v, replacements)) for k, v in arg.items())
        )
    if isinstance(arg, Hashable):
        return arg
    return repr(arg)


def _is_cse_eligible(node: Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in _NON_CSE_TARGETS:
        return False
    target_name = getattr(node.target, "__name__", str(node.target))
    return not any(name in target_name for name in _NON_CSE_TARGET_NAMES)


def common_subexpression_elimination(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Deduplicate repeated pure call_function nodes.

    autograd_per_op often materializes duplicate forward subgraphs solely to
    save activations for backward.  Collapsing them before fusion exposes the
    original single-use chains again so matmul_epilogue and pointwise fusion
    can absorb more work.
    """

    del example_inputs

    graph = gm.graph
    changed = False
    seen = {}
    replacements = {}

    for node in list(graph.nodes):
        if not _is_cse_eligible(node):
            continue

        key = (
            _phase_key(node),
            node.target,
            _freeze_arg(node.args, replacements),
            _freeze_arg(node.kwargs, replacements),
            tuple(node.meta.get("shape", ())),
            node.meta.get("dtype", None),
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = replacements.get(node, node)
            continue

        node.replace_all_uses_with(existing)
        replacements[node] = existing
        graph.erase_node(node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)
