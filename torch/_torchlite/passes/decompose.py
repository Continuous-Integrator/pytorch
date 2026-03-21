"""Decomposition pass: lower compound ops into core ATen ops."""
import logging
import operator
from typing import Dict, List

import torch
from torch._decomp import core_aten_decompositions
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _aten_op_name,
    _create_name,
    _is_torch_op,
    _iter_node_args,
    _PROVENANCE_KEYS,
    _set_phase,
    FusedKernel,
    PassResult,
)

log = logging.getLogger(__name__)
_CORE_ATEN_DECOMP_TABLE = core_aten_decompositions()


class _DecompRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Intercepts aten ops during decomposition, applies decompositions from
    the table for non-core ops, and records leaf (core) ops directly into an
    existing FX graph. This replaces make_fx with inline graph surgery.
    """

    def __init__(self, graph, id_to_node, decomp_table, new_nodes, prov):
        self.graph = graph
        self.id_to_node = id_to_node
        self.decomp_table = decomp_table
        self.new_nodes = new_nodes
        self.prov = prov

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        decomp = self.decomp_table.get(func)
        if decomp is not None:
            return decomp(*args, **kwargs)

        if func is torch.ops.aten.view.default:
            func = torch.ops.aten.reshape.default

        result = func(*args, **kwargs)

        def _to_fx(x):
            if isinstance(x, torch.Tensor) and id(x) in self.id_to_node:
                return self.id_to_node[id(x)]
            if isinstance(x, (list, tuple)):
                return type(x)(_to_fx(i) for i in x)
            return x

        fx_args = tuple(
            _to_fx(a) if not isinstance(a, (list, tuple))
            else type(a)(_to_fx(x) for x in a)
            for a in args
        )
        fx_kwargs = {k: _to_fx(v) for k, v in kwargs.items()}
        if func is torch.ops.aten._to_copy.default:
            dev = fx_kwargs.get("device")
            if isinstance(dev, torch.device) and dev.type == "meta":
                del fx_kwargs["device"]

        new_node = self.graph.call_function(func, fx_args, fx_kwargs)
        pkt = getattr(func, "overloadpacket", None)
        name_hint = getattr(pkt, "__name__", None) if pkt else None
        if name_hint is None:
            name_hint = getattr(func, "__name__", "decomp")
        new_node.name = _create_name(self.graph, name_hint)

        if isinstance(result, torch.Tensor):
            new_node.meta["shape"] = list(result.shape)
            new_node.meta["dtype"] = result.dtype
            self.id_to_node[id(result)] = new_node
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    get_node = self.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    get_node.name = _create_name(self.graph, f"{name_hint}_item")
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    self.id_to_node[id(r)] = get_node
                    self.new_nodes.append(get_node)

        for k, v in self.prov.items():
            new_node.meta[k] = v

        self.new_nodes.append(new_node)
        return result


import torch.nn.functional as _F

_DECOMP_BLOCKLIST = frozenset({
    torch.ops.aten.reshape.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    _F.rms_norm,
    _F.scaled_dot_product_attention,
})

_INFERENCE_LAYOUT_TARGETS = frozenset({
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.permute.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.select.int,
    torch.Tensor.reshape,
    torch.Tensor.view,
})

_INFERENCE_LINEAR_BLOCKERS = frozenset({
    "add", "sub", "rsub", "mul", "div", "where",
    "gt", "ge", "lt", "le", "eq", "ne",
    "pow", "clamp", "clamp_min", "clamp_max",
    "sin", "cos", "neg", "abs",
    "relu", "sigmoid", "tanh",
    "rsqrt", "sqrt", "exp", "log", "reciprocal",
    "silu", "gelu",
})

_INFERENCE_NORM_TARGETS = {
    _F.rms_norm,
    _F.layer_norm,
}
_DEFAULT_INFERENCE_REGIONS = frozenset({"attention", "ffn"})
_INFERENCE_REGION_POINTWISE = frozenset({"relu", "silu", "gelu", "mul"})


def _manual_decomp_threshold_backward(graph, node):
    grad, input_node, threshold = node.args
    shape = node.meta.get("shape")
    dtype = node.meta.get("dtype", torch.float32)
    phase = node.meta.get("phase", "backward")
    prov = {k: node.meta[k] for k in _PROVENANCE_KEYS if k in node.meta}

    graph.inserting_before(node)
    gt_node = graph.call_function(
        torch.ops.aten.gt.Scalar, (input_node, threshold)
    )
    gt_node.name = _create_name(graph, "gt")
    gt_node.meta["dtype"] = torch.bool
    if shape is not None:
        gt_node.meta["shape"] = list(shape)
    for k, v in prov.items():
        gt_node.meta[k] = v

    mul_node = graph.call_function(
        torch.ops.aten.mul.Tensor, (grad, gt_node)
    )
    mul_node.name = _create_name(graph, "threshold_backward_mul")
    if shape is not None:
        mul_node.meta["shape"] = list(shape)
    mul_node.meta["dtype"] = dtype
    for k, v in prov.items():
        mul_node.meta[k] = v

    node.replace_all_uses_with(mul_node)
    graph.erase_node(node)
    return True


def _manual_decomp_sigmoid_backward(graph, node):
    grad, output = node.args
    shape = node.meta.get("shape")
    dtype = node.meta.get("dtype", torch.float32)
    phase = node.meta.get("phase", "backward")
    prov = {k: node.meta[k] for k in _PROVENANCE_KEYS if k in node.meta}

    graph.inserting_before(node)

    one_minus = graph.call_function(
        torch.ops.aten.rsub.Scalar, (output, 1)
    )
    one_minus.name = _create_name(graph, "one_minus_sigmoid")
    if shape is not None:
        one_minus.meta["shape"] = list(shape)
    one_minus.meta["dtype"] = dtype
    for k, v in prov.items():
        one_minus.meta[k] = v

    sig_deriv = graph.call_function(
        torch.ops.aten.mul.Tensor, (output, one_minus)
    )
    sig_deriv.name = _create_name(graph, "sigmoid_deriv")
    if shape is not None:
        sig_deriv.meta["shape"] = list(shape)
    sig_deriv.meta["dtype"] = dtype
    for k, v in prov.items():
        sig_deriv.meta[k] = v

    result = graph.call_function(
        torch.ops.aten.mul.Tensor, (grad, sig_deriv)
    )
    result.name = _create_name(graph, "sigmoid_backward_result")
    if shape is not None:
        result.meta["shape"] = list(shape)
    result.meta["dtype"] = dtype
    for k, v in prov.items():
        result.meta[k] = v

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True


def _manual_decomp_tanh_backward(graph, node):
    grad, output = node.args
    shape = node.meta.get("shape")
    dtype = node.meta.get("dtype", torch.float32)
    phase = node.meta.get("phase", "backward")
    prov = {k: node.meta[k] for k in _PROVENANCE_KEYS if k in node.meta}

    graph.inserting_before(node)

    sq = graph.call_function(
        torch.ops.aten.mul.Tensor, (output, output)
    )
    sq.name = _create_name(graph, "tanh_sq")
    if shape is not None:
        sq.meta["shape"] = list(shape)
    sq.meta["dtype"] = dtype
    for k, v in prov.items():
        sq.meta[k] = v

    one_minus_sq = graph.call_function(
        torch.ops.aten.rsub.Scalar, (sq, 1)
    )
    one_minus_sq.name = _create_name(graph, "one_minus_tanh_sq")
    if shape is not None:
        one_minus_sq.meta["shape"] = list(shape)
    one_minus_sq.meta["dtype"] = dtype
    for k, v in prov.items():
        one_minus_sq.meta[k] = v

    result = graph.call_function(
        torch.ops.aten.mul.Tensor, (grad, one_minus_sq)
    )
    result.name = _create_name(graph, "tanh_backward_result")
    if shape is not None:
        result.meta["shape"] = list(shape)
    result.meta["dtype"] = dtype
    for k, v in prov.items():
        result.meta[k] = v

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True


_MANUAL_DECOMPS = {
    torch.ops.aten.threshold_backward.default: _manual_decomp_threshold_backward,
    torch.ops.aten.sigmoid_backward.default: _manual_decomp_sigmoid_backward,
    torch.ops.aten.tanh_backward.default: _manual_decomp_tanh_backward,
}


def _inference_terminal_users(node):
    seen = set()
    worklist = list(node.users.keys())
    terminals = []

    while worklist:
        user = worklist.pop()
        if user in seen:
            continue
        seen.add(user)

        if user.op == "output":
            terminals.append(user)
            continue

        if user.op != "call_function":
            terminals.append(user)
            continue

        if user.target is operator.getitem or user.target in _INFERENCE_LAYOUT_TARGETS:
            worklist.extend(user.users.keys())
            continue

        terminals.append(user)

    return terminals


def _should_preserve_inference_node(
    node,
    *,
    preserve_regions=_DEFAULT_INFERENCE_REGIONS,
) -> bool:
    region_kind = node.meta.get("region_kind")
    if region_kind in preserve_regions:
        op_name = _aten_op_name(node.target)
        if region_kind == "attention":
            if node.target is torch._C._nn.linear:
                return True
            if node.target is _F.scaled_dot_product_attention:
                return True
            if op_name == "repeat_interleave":
                return True
        if region_kind == "ffn":
            if node.target in _INFERENCE_NORM_TARGETS:
                return True
            if op_name in _INFERENCE_REGION_POINTWISE:
                return True

    if _aten_op_name(node.target) == "repeat_interleave":
        return True

    if node.target is not torch._C._nn.linear:
        return False

    if region_kind == "ffn":
        return False

    terminals = _inference_terminal_users(node)
    if not terminals:
        return True

    for user in terminals:
        if user.op == "output":
            continue
        if user.op != "call_function":
            return False
        if user.target in _INFERENCE_NORM_TARGETS:
            return False
        if _aten_op_name(user.target) in _INFERENCE_LINEAR_BLOCKERS:
            return False

    return True


def _decompose_impl(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    preserve_predicate=None,
) -> PassResult:
    graph = gm.graph

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        manual_fn = _MANUAL_DECOMPS.get(node.target)
        if manual_fn is not None:
            manual_fn(graph, node)

    decomp_table = _CORE_ATEN_DECOMP_TABLE

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target in _DECOMP_BLOCKLIST:
            continue
        if preserve_predicate is not None and preserve_predicate(node):
            continue
        if not _is_torch_op(node.target):
            continue

        tensor_map = {}
        id_to_node = {}
        can_decompose = True

        for a in _iter_node_args(node):
            if not isinstance(a, torch.fx.Node):
                continue
            if a in tensor_map:
                continue
            shape = a.meta.get("shape")
            if shape is None:
                can_decompose = False
                break
            dtype = a.meta.get("dtype", torch.float32)
            t = torch.empty(shape if shape else [], dtype=dtype, device="meta")
            tensor_map[a] = t
            id_to_node[id(t)] = a

        if not can_decompose or not tensor_map:
            continue

        prov = {k: node.meta[k] for k in _PROVENANCE_KEYS if k in node.meta}
        new_nodes: list = []

        def _resolve(x):
            if isinstance(x, torch.fx.Node):
                return tensor_map[x]
            if isinstance(x, (list, tuple)):
                return type(x)(_resolve(i) for i in x)
            return x

        real_args = tuple(_resolve(a) for a in node.args)
        real_kwargs = {k: _resolve(v) for k, v in node.kwargs.items()}

        graph.inserting_before(node)
        try:
            with _DecompRecorder(graph, id_to_node, decomp_table, new_nodes, prov):
                decomp_result = node.target(*real_args, **real_kwargs)
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as e:
            op_name = getattr(node.target, "__name__", str(node.target))
            log.debug("decompose: skipping %s (%s)", op_name, e)
            for n in reversed(new_nodes):
                if not n.users:
                    graph.erase_node(n)
            continue

        if not new_nodes:
            continue

        if len(new_nodes) == 1 and new_nodes[0].target is node.target:
            graph.erase_node(new_nodes[0])
            continue

        # For multi-output ops like native_layer_norm, the decomposition
        # returns a tuple of tensors.  The original graph has getitem nodes
        # that extract each element.  We need to replace each getitem with
        # the corresponding decomposed tensor's FX node, not wholesale-replace
        # the multi-output op with a single node (which would make the
        # getitem nodes index into the wrong value at runtime).
        if isinstance(decomp_result, (tuple, list)):
            result_map = {}
            for i, r in enumerate(decomp_result):
                if isinstance(r, torch.Tensor) and id(r) in id_to_node:
                    result_map[i] = id_to_node[id(r)]
            for user in list(node.users.keys()):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and len(user.args) >= 2
                ):
                    idx = user.args[1]
                    replacement = result_map.get(idx)
                    if replacement is not None:
                        user.replace_all_uses_with(replacement)
                        graph.erase_node(user)
            if not node.users:
                graph.erase_node(node)
        elif isinstance(decomp_result, torch.Tensor):
            result_node = id_to_node.get(id(decomp_result), new_nodes[-1])
            node.replace_all_uses_with(result_node)
            graph.erase_node(node)
        else:
            result_node = new_nodes[-1]
            node.replace_all_uses_with(result_node)
            graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def decompose(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    return _decompose_impl(gm, example_inputs)


def decompose_inference(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    preserve_regions=_DEFAULT_INFERENCE_REGIONS,
) -> PassResult:
    """Inference-oriented decomposition that preserves some high-level ops.

    Preserve region-marked attention projections/SDPA plus FFN norms and
    pointwise ops that downstream inference passes already optimize well,
    while still decomposing FFN projections back to matmul form. Standalone
    linears that do not feed a pointwise epilogue chain and repeat_interleave
    nodes are also preserved. This keeps the pipeline from exploding
    attention structure too early without blocking later FFN/attention
    matmul canonicalization.
    """

    return _decompose_impl(
        gm,
        example_inputs,
        preserve_predicate=lambda node: _should_preserve_inference_node(
            node,
            preserve_regions=preserve_regions,
        ),
    )
