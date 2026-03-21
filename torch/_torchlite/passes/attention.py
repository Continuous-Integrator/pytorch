"""Attention-specific graph canonicalization passes."""

import operator
from typing import Optional

import torch
import torch.nn.functional as F
from torch._torchlite.passes.common import (
    _aten_op_name,
    _copy_region_meta,
    _create_name,
    _deep_getattr,
    _set_phase,
    PassResult,
)
from torch.fx import GraphModule, Node


_RESHAPE_TARGETS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.Tensor.reshape,
    torch.Tensor.view,
}
_LAYOUT_TARGET_NAMES = frozenset({
    "reshape",
    "view",
    "_unsafe_view",
    "transpose",
    "permute",
    "expand",
    "unsqueeze",
    "slice",
    "select",
    "contiguous",
    "clone",
    "repeat_interleave",
})
_PROJECTION_TARGET_NAMES = frozenset({"linear", "matmul", "mm", "addmm"})
_ATTN_SCORE_TARGET_NAMES = frozenset({"matmul", "bmm"})
_SOFTMAX_TARGET_NAMES = frozenset({"softmax", "_softmax", "_safe_softmax"})
_SCALE_TARGET_NAMES = frozenset({"mul", "div"})
_FFN_ACTIVATION_NAMES = frozenset({"relu", "gelu", "silu"})
_FFN_NORM_NAMES = frozenset({"rms_norm", "layer_norm"})
_ATTN_TERMINAL_PASSTHROUGH_NAMES = _LAYOUT_TARGET_NAMES | frozenset({"unbind"})


def _shape(node: Node) -> Optional[list[int]]:
    if not isinstance(node, Node):
        return None
    return node.meta.get("shape")


def _normalized_dim(dim: int, rank: int) -> Optional[int]:
    if not isinstance(dim, int):
        return None
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    return dim


def _single_user(node: Node) -> bool:
    return len(list(node.users.keys())) == 1


def _sole_user(node: Node) -> Optional[Node]:
    users = list(node.users.keys())
    if len(users) == 1:
        return users[0]
    return None


def _is_layout_node(node: Node) -> bool:
    return node.op == "call_function" and _aten_op_name(node.target) in _LAYOUT_TARGET_NAMES


def _is_projection_node(node: Node) -> bool:
    return (
        node.op == "call_function"
        and _aten_op_name(node.target) in _PROJECTION_TARGET_NAMES
    )


def _mark_region(
    node: Node,
    *,
    region_id: int,
    region_kind: str,
    region_role: str,
) -> None:
    node.meta["region_id"] = region_id
    node.meta["region_kind"] = region_kind
    node.meta["region_role"] = region_role


def _mark_region_nodes(
    nodes: list[Node],
    *,
    region_id: int,
    region_kind: str,
    region_role: str,
) -> None:
    for node in nodes:
        _mark_region(
            node,
            region_id=region_id,
            region_kind=region_kind,
            region_role=region_role,
        )


def _collect_backward_chain(node: Node) -> list[Node]:
    chain = []
    seen = set()
    cur = node

    while isinstance(cur, Node) and cur not in seen and cur.op == "call_function":
        seen.add(cur)
        chain.append(cur)
        if _is_projection_node(cur):
            break
        if not cur.args or not isinstance(cur.args[0], Node):
            break
        nxt = cur.args[0]
        if not isinstance(nxt, Node) or nxt.op != "call_function":
            break
        if not (_is_layout_node(cur) or _aten_op_name(cur.target) == "_to_copy"):
            break
        cur = nxt

    return chain


def _projection_input_node(node: Node) -> Optional[Node]:
    if not _is_projection_node(node) or not node.args:
        return None
    arg0 = node.args[0]
    if isinstance(arg0, Node):
        return arg0
    return None


def _maybe_mark_ffn_norm(
    source_node: Optional[Node],
    *,
    region_id: int,
) -> None:
    if (
        isinstance(source_node, Node)
        and source_node.op == "call_function"
        and _aten_op_name(source_node.target) in _FFN_NORM_NAMES
    ):
        _mark_region(
            source_node,
            region_id=region_id,
            region_kind="ffn",
            region_role="norm",
        )


def _mark_attention_output_tail(
    start: Node,
    *,
    region_id: int,
) -> None:
    cur = start

    while True:
        user = _sole_user(cur)
        if user is None or user.op != "call_function":
            return

        op_name = _aten_op_name(user.target)
        if _is_layout_node(user):
            _mark_region(
                user,
                region_id=region_id,
                region_kind="attention",
                region_role="out_layout",
            )
            cur = user
            continue

        if _is_projection_node(user):
            _mark_region(
                user,
                region_id=region_id,
                region_kind="attention",
                region_role="out_proj",
            )
            cur = user
            continue

        if op_name == "add":
            _mark_region(
                user,
                region_id=region_id,
                region_kind="attention",
                region_role="residual",
            )
        return


def _match_scaled_scores(node: Node) -> tuple[Optional[Node], Optional[Node]]:
    if node.op != "call_function":
        return None, None

    op_name = _aten_op_name(node.target)
    if op_name not in _SCALE_TARGET_NAMES:
        return node, None

    if len(node.args) != 2:
        return None, None

    arg0, arg1 = node.args
    if isinstance(arg0, Node) and not isinstance(arg1, Node):
        return arg0, node
    if isinstance(arg1, Node) and not isinstance(arg0, Node):
        return arg1, node
    return None, None


def _attention_terminal_users(node: Node) -> list[Node]:
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

        op_name = _aten_op_name(user.target)
        if user.target is operator.getitem or op_name in _ATTN_TERMINAL_PASSTHROUGH_NAMES:
            worklist.extend(user.users.keys())
            continue

        terminals.append(user)

    return terminals


def _should_decompose_attention_projection(node: Node) -> bool:
    if node.op != "call_function" or node.target is not torch._C._nn.linear:
        return False

    terminals = _attention_terminal_users(node)
    if node.meta.get("region_kind") == "attention":
        return True

    return any(
        terminal.op == "call_function"
        and _aten_op_name(terminal.target) == "scaled_dot_product_attention"
        for terminal in terminals
    )


def _mark_attention_region(
    *,
    region_id: int,
    q_node: Node,
    k_node: Node,
    v_node: Node,
    core_node: Node,
    tail_node: Node,
    score_node: Optional[Node] = None,
    scale_node: Optional[Node] = None,
    softmax_node: Optional[Node] = None,
) -> None:
    _mark_region_nodes(
        _collect_backward_chain(q_node),
        region_id=region_id,
        region_kind="attention",
        region_role="q",
    )
    _mark_region_nodes(
        _collect_backward_chain(k_node),
        region_id=region_id,
        region_kind="attention",
        region_role="k",
    )
    _mark_region_nodes(
        _collect_backward_chain(v_node),
        region_id=region_id,
        region_kind="attention",
        region_role="v",
    )
    if score_node is not None:
        _mark_region(
            score_node,
            region_id=region_id,
            region_kind="attention",
            region_role="scores",
        )
    if scale_node is not None:
        _mark_region(
            scale_node,
            region_id=region_id,
            region_kind="attention",
            region_role="scale",
        )
    if softmax_node is not None:
        _mark_region(
            softmax_node,
            region_id=region_id,
            region_kind="attention",
            region_role="softmax",
        )
    _mark_region(
        core_node,
        region_id=region_id,
        region_kind="attention",
        region_role="core",
    )
    _mark_attention_output_tail(tail_node, region_id=region_id)


def _match_repeat_interleave_decomp(node: Node) -> Optional[tuple[Node, int, int]]:
    if node.op != "call_function" or node.target not in _RESHAPE_TARGETS:
        return None

    clone_node = node.args[0]
    if not isinstance(clone_node, Node):
        return None
    if (
        clone_node.op != "call_function"
        or clone_node.target != torch.ops.aten.clone.default
    ):
        return None

    expand_node = clone_node.args[0]
    if not isinstance(expand_node, Node):
        return None
    if (
        expand_node.op != "call_function"
        or expand_node.target != torch.ops.aten.expand.default
    ):
        return None

    unsqueeze_node = expand_node.args[0]
    if not isinstance(unsqueeze_node, Node):
        return None
    if (
        unsqueeze_node.op != "call_function"
        or unsqueeze_node.target != torch.ops.aten.unsqueeze.default
    ):
        return None

    source = unsqueeze_node.args[0]
    dim = unsqueeze_node.args[1]
    if not isinstance(source, Node) or not isinstance(dim, int) or dim <= 0:
        return None

    source_shape = _shape(source)
    expand_shape = _shape(expand_node)
    out_shape = _shape(node)
    if source_shape is None or expand_shape is None or out_shape is None:
        return None
    if len(source_shape) + 1 != len(expand_shape):
        return None

    expected_expand = list(source_shape)
    expected_expand.insert(dim, 1)
    repeats = expand_shape[dim]
    if not isinstance(repeats, int) or repeats <= 1:
        return None

    for idx, (want, actual) in enumerate(zip(expected_expand, expand_shape)):
        if idx == dim:
            continue
        if want != actual:
            return None

    expected_out = list(source_shape)
    expected_out[dim - 1] *= repeats
    if expected_out != list(out_shape):
        return None

    if not (
        _single_user(unsqueeze_node)
        and _single_user(expand_node)
        and _single_user(clone_node)
    ):
        return None

    return source, repeats, dim - 1


def _match_repeat_interleave_node(node: Node) -> Optional[tuple[Node, int, int]]:
    if node.op != "call_function" or _aten_op_name(node.target) != "repeat_interleave":
        return None

    if not node.args or not isinstance(node.args[0], Node):
        return None

    repeats = None
    if len(node.args) > 1 and isinstance(node.args[1], int):
        repeats = node.args[1]
    elif isinstance(node.kwargs.get("repeats"), int):
        repeats = node.kwargs["repeats"]
    if repeats is None or repeats <= 1:
        return None

    dim = node.kwargs.get("dim")
    if dim is None and len(node.args) > 2:
        dim = node.args[2]
    source = node.args[0]
    source_shape = _shape(source)
    out_shape = _shape(node)
    if source_shape is None or out_shape is None or len(source_shape) != len(out_shape):
        return None

    dim = _normalized_dim(dim, len(source_shape))
    if dim is None:
        return None

    expected_shape = list(source_shape)
    expected_shape[dim] *= repeats
    if expected_shape != list(out_shape):
        return None

    return source, repeats, dim


def extract_attention_regions(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Annotate attention subgraphs with stable region metadata.

    This marks both explicit SDPA graphs and raw attention score paths
    (q @ k^T -> scale -> softmax -> attn @ v) so later passes can preserve
    or transform whole regions without rediscovering the pattern.
    """

    del example_inputs

    region_id = 0

    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is F.scaled_dot_product_attention
            and len(node.args) >= 3
        ):
            q_node, k_node, v_node = node.args[:3]
            if not all(isinstance(arg, Node) for arg in (q_node, k_node, v_node)):
                continue
            _mark_attention_region(
                region_id=region_id,
                q_node=q_node,
                k_node=k_node,
                v_node=v_node,
                core_node=node,
                tail_node=node,
            )
            node.meta["attention_kind"] = node.meta.get("attention_kind", "sdpa")
            region_id += 1
            continue

        if (
            node.op != "call_function"
            or _aten_op_name(node.target) not in _SOFTMAX_TARGET_NAMES
            or not node.args
            or not isinstance(node.args[0], Node)
        ):
            continue

        score_node, scale_node = _match_scaled_scores(node.args[0])
        if (
            score_node is None
            or score_node.op != "call_function"
            or _aten_op_name(score_node.target) not in _ATTN_SCORE_TARGET_NAMES
            or len(score_node.args) != 2
        ):
            continue

        q_node, k_node = score_node.args
        if not isinstance(q_node, Node) or not isinstance(k_node, Node):
            continue

        ctx_node = _sole_user(node)
        if (
            ctx_node is None
            or ctx_node.op != "call_function"
            or _aten_op_name(ctx_node.target) not in _ATTN_SCORE_TARGET_NAMES
            or len(ctx_node.args) != 2
            or ctx_node.args[0] is not node
            or not isinstance(ctx_node.args[1], Node)
        ):
            continue

        v_node = ctx_node.args[1]
        _mark_attention_region(
            region_id=region_id,
            q_node=q_node,
            k_node=k_node,
            v_node=v_node,
            core_node=ctx_node,
            tail_node=ctx_node,
            score_node=score_node,
            scale_node=scale_node,
            softmax_node=node,
        )
        node.meta["attention_kind"] = node.meta.get("attention_kind", "manual")
        region_id += 1

    return PassResult(gm=gm, changed=region_id > 0)


def extract_ffn_regions(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Annotate dense FFN islands with stable region metadata."""

    del example_inputs

    region_id = 0

    for node in gm.graph.nodes:
        if (
            "region_kind" in node.meta
            or node.op != "call_function"
            or _aten_op_name(node.target) != "mul"
            or len(node.args) != 2
        ):
            continue

        arg0, arg1 = node.args
        gate_node = None
        up_node = None
        for lhs, rhs in ((arg0, arg1), (arg1, arg0)):
            if (
                isinstance(lhs, Node)
                and lhs.op == "call_function"
                and _aten_op_name(lhs.target) in _FFN_ACTIVATION_NAMES
                and isinstance(rhs, Node)
            ):
                gate_node = lhs
                up_node = rhs
                break
        if gate_node is None or up_node is None:
            continue

        if not gate_node.args or not isinstance(gate_node.args[0], Node):
            continue
        gate_proj_node = gate_node.args[0]
        down_node = _sole_user(node)
        if down_node is None or not _is_projection_node(down_node):
            continue

        gate_chain = _collect_backward_chain(gate_proj_node)
        up_chain = _collect_backward_chain(up_node)
        if not gate_chain or not up_chain:
            continue

        _mark_region_nodes(
            gate_chain,
            region_id=region_id,
            region_kind="ffn",
            region_role="gate",
        )
        _mark_region(
            gate_node,
            region_id=region_id,
            region_kind="ffn",
            region_role="activation",
        )
        _mark_region_nodes(
            up_chain,
            region_id=region_id,
            region_kind="ffn",
            region_role="up",
        )
        _mark_region(
            node,
            region_id=region_id,
            region_kind="ffn",
            region_role="combine",
        )
        _mark_region(
            down_node,
            region_id=region_id,
            region_kind="ffn",
            region_role="down",
        )
        _maybe_mark_ffn_norm(
            _projection_input_node(gate_chain[-1]),
            region_id=region_id,
        )
        region_id += 1

    for node in gm.graph.nodes:
        if (
            "region_kind" in node.meta
            or node.op != "call_function"
            or _aten_op_name(node.target) not in _FFN_ACTIVATION_NAMES
            or not node.args
            or not isinstance(node.args[0], Node)
        ):
            continue

        up_chain = _collect_backward_chain(node.args[0])
        down_node = _sole_user(node)
        if not up_chain or down_node is None or not _is_projection_node(down_node):
            continue

        _mark_region_nodes(
            up_chain,
            region_id=region_id,
            region_kind="ffn",
            region_role="up",
        )
        _mark_region(
            node,
            region_id=region_id,
            region_kind="ffn",
            region_role="activation",
        )
        _mark_region(
            down_node,
            region_id=region_id,
            region_kind="ffn",
            region_role="down",
        )
        _maybe_mark_ffn_norm(
            _projection_input_node(up_chain[-1]),
            region_id=region_id,
        )
        region_id += 1

    return PassResult(gm=gm, changed=region_id > 0)


def _updated_view_args(
    view_node: Node,
    new_source: Node,
    dim: int,
    repeats: int,
) -> Optional[tuple]:
    if len(view_node.args) < 2:
        return None

    if len(view_node.args) == 2 and isinstance(view_node.args[1], (list, tuple)):
        shape_args = list(view_node.args[1])
        if dim >= len(shape_args) or not isinstance(shape_args[dim], int):
            return None
        shape_args[dim] *= repeats
        return (new_source, type(view_node.args[1])(shape_args))

    shape_args = list(view_node.args[1:])
    if dim >= len(shape_args) or not isinstance(shape_args[dim], int):
        return None
    shape_args[dim] *= repeats
    return (new_source, *shape_args)


def _match_gqa_projection_repeat(
    gm: GraphModule,
    node: Node,
) -> Optional[tuple]:
    repeat_match = _match_repeat_interleave_node(node)
    if repeat_match is None:
        return None

    transpose_node, repeats, repeat_dim = repeat_match
    if (
        transpose_node.op != "call_function"
        or _aten_op_name(transpose_node.target) != "transpose"
        or len(transpose_node.args) < 3
    ):
        return None

    view_node = transpose_node.args[0]
    if (
        not isinstance(view_node, Node)
        or view_node.op != "call_function"
        or view_node.target not in _RESHAPE_TARGETS
    ):
        return None

    view_shape = _shape(view_node)
    transpose_shape = _shape(transpose_node)
    repeat_shape = _shape(node)
    if view_shape is None or transpose_shape is None or repeat_shape is None:
        return None
    if len(view_shape) < 3:
        return None

    dim0 = _normalized_dim(transpose_node.args[1], len(view_shape))
    dim1 = _normalized_dim(transpose_node.args[2], len(view_shape))
    if dim0 is None or dim1 is None:
        return None

    if repeat_dim == dim0:
        source_head_dim = dim1
    elif repeat_dim == dim1:
        source_head_dim = dim0
    else:
        return None
    if source_head_dim != len(view_shape) - 2:
        return None

    linear_node = view_node.args[0]
    if (
        not isinstance(linear_node, Node)
        or linear_node.op != "call_function"
        or linear_node.target is not torch._C._nn.linear
    ):
        return None
    if not (
        _single_user(linear_node)
        and _single_user(view_node)
        and _single_user(transpose_node)
    ):
        return None

    kv_heads = view_shape[source_head_dim]
    head_dim = view_shape[-1]
    if not isinstance(kv_heads, int) or not isinstance(head_dim, int):
        return None
    if transpose_shape[repeat_dim] != kv_heads:
        return None
    if repeat_shape[repeat_dim] != kv_heads * repeats:
        return None

    weight_node = linear_node.args[1] if len(linear_node.args) > 1 else None
    if not isinstance(weight_node, Node) or weight_node.op != "get_attr":
        return None
    weight = _deep_getattr(gm, weight_node.target)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    if weight.shape[0] != kv_heads * head_dim:
        return None

    bias = None
    if len(linear_node.args) > 2 and linear_node.args[2] is not None:
        bias_node = linear_node.args[2]
        if not isinstance(bias_node, Node) or bias_node.op != "get_attr":
            return None
        bias = _deep_getattr(gm, bias_node.target)
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            return None
        if bias.shape[0] != kv_heads * head_dim:
            return None

    return (
        linear_node,
        view_node,
        transpose_node,
        repeats,
        source_head_dim,
        kv_heads,
        head_dim,
        weight,
        bias,
    )


def expand_gqa_projections(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Expand inference-only GQA K/V projection weights once at compile time.

    Rewrites ``linear -> view -> transpose -> repeat_interleave`` into a
    larger ``linear -> view -> transpose`` by repeating the K/V head blocks in
    the constant weight and bias tensors. This removes the runtime repeat and
    exposes equal-width Q/K/V projections for later packing.
    """

    del example_inputs

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        match = _match_gqa_projection_repeat(gm, node)
        if match is None:
            continue

        (
            linear_node,
            view_node,
            transpose_node,
            repeats,
            head_dim_idx,
            kv_heads,
            head_dim,
            weight,
            bias,
        ) = match
        input_node = linear_node.args[0]
        if not isinstance(input_node, Node):
            continue

        expanded_weight = (
            weight.view(kv_heads, head_dim, weight.shape[1])
            .repeat_interleave(repeats, dim=0)
            .reshape(kv_heads * repeats * head_dim, weight.shape[1])
            .contiguous()
        )
        expanded_bias = None
        if bias is not None:
            expanded_bias = (
                bias.view(kv_heads, head_dim)
                .repeat_interleave(repeats, dim=0)
                .reshape(kv_heads * repeats * head_dim)
                .contiguous()
            )

        graph.inserting_before(node)

        weight_name = _create_name(graph, f"{linear_node.name}_gqa_weight")
        gm.register_buffer(weight_name, expanded_weight, persistent=False)
        expanded_weight_node = graph.get_attr(weight_name)
        expanded_weight_node.meta["shape"] = list(expanded_weight.shape)
        expanded_weight_node.meta["dtype"] = expanded_weight.dtype

        expanded_bias_node = None
        if expanded_bias is not None:
            bias_name = _create_name(graph, f"{linear_node.name}_gqa_bias")
            gm.register_buffer(bias_name, expanded_bias, persistent=False)
            expanded_bias_node = graph.get_attr(bias_name)
            expanded_bias_node.meta["shape"] = list(expanded_bias.shape)
            expanded_bias_node.meta["dtype"] = expanded_bias.dtype

        expanded_linear = graph.call_function(
            torch._C._nn.linear,
            (input_node, expanded_weight_node, expanded_bias_node),
        )
        expanded_linear.name = _create_name(graph, f"{linear_node.name}_gqa")
        expanded_linear.meta["shape"] = list(linear_node.meta.get("shape", []))
        if expanded_linear.meta["shape"]:
            expanded_linear.meta["shape"][-1] *= repeats
        expanded_linear.meta["dtype"] = linear_node.meta.get("dtype", torch.float32)
        _set_phase(expanded_linear, linear_node.meta.get("phase", "forward"))
        _copy_region_meta(expanded_linear, linear_node)

        view_args = _updated_view_args(view_node, expanded_linear, head_dim_idx, repeats)
        if view_args is None:
            graph.erase_node(expanded_linear)
            continue

        expanded_view = graph.call_function(
            view_node.target,
            view_args,
            dict(view_node.kwargs),
        )
        expanded_view.name = _create_name(graph, f"{view_node.name}_gqa")
        expanded_view.meta["shape"] = list(view_node.meta.get("shape", []))
        if expanded_view.meta["shape"]:
            expanded_view.meta["shape"][head_dim_idx] *= repeats
        expanded_view.meta["dtype"] = view_node.meta.get("dtype", torch.float32)
        _set_phase(expanded_view, view_node.meta.get("phase", "forward"))
        _copy_region_meta(expanded_view, view_node)

        expanded_transpose = graph.call_function(
            transpose_node.target,
            (expanded_view, *transpose_node.args[1:]),
            dict(transpose_node.kwargs),
        )
        expanded_transpose.name = _create_name(graph, f"{transpose_node.name}_gqa")
        expanded_transpose.meta["shape"] = list(node.meta.get("shape", []))
        expanded_transpose.meta["dtype"] = node.meta.get("dtype", torch.float32)
        _set_phase(expanded_transpose, node.meta.get("phase", "forward"))
        _copy_region_meta(expanded_transpose, node)

        node.replace_all_uses_with(expanded_transpose)
        graph.erase_node(node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)


def attention_canonicalize(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    allow_gqa: bool = True,
    allow_masks: bool = True,
) -> PassResult:
    """Canonicalize attention-adjacent layout patterns.

    Today this focuses on the GQA repeat-KV decomposition:
    ``unsqueeze -> expand -> clone -> reshape`` becomes a single
    ``torch.repeat_interleave`` call, which is both easier to reason
    about and safer for inference codegen.
    """

    del allow_masks
    del example_inputs

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        match = _match_repeat_interleave_decomp(node) if allow_gqa else None
        if match is None:
            if (
                node.op == "call_function"
                and node.target is F.scaled_dot_product_attention
                and len(node.args) >= 3
            ):
                _, k, v = node.args[:3]
                if _match_repeat_interleave_node(k) and _match_repeat_interleave_node(v):
                    node.meta["attention_kind"] = "gqa"
            continue

        source, repeats, dim = match
        graph.inserting_before(node)
        repeat_node = graph.call_function(
            torch.repeat_interleave,
            (source, repeats),
            {"dim": dim},
        )
        repeat_node.name = _create_name(graph, "repeat_kv")
        repeat_node.meta["shape"] = list(node.meta.get("shape", []))
        repeat_node.meta["dtype"] = node.meta.get("dtype", torch.float32)
        _set_phase(repeat_node, node.meta.get("phase", "forward"))
        _copy_region_meta(repeat_node, node)

        node.replace_all_uses_with(repeat_node)
        graph.erase_node(node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)


def decompose_attention_projections(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Rewrite preserved attention projections to explicit addmm/mm graphs.

    The inference pipeline intentionally preserves some high-level ``linear``
    nodes to avoid over-decomposing the graph too early. For attention-heavy
    models this leaves packed QKV and out-proj paths on the slower generic
    linear lowering path. Rewriting those projections back to explicit
    ``addmm/mm`` nodes lets later codegen take the fast matmul path without
    changing the surrounding attention structure.
    """

    del example_inputs

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if not _should_decompose_attention_projection(node):
            continue

        if len(node.args) < 2:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = node.args[2] if len(node.args) > 2 else None

        if not isinstance(input_node, Node) or not isinstance(weight_node, Node):
            continue

        input_shape = _shape(input_node)
        output_shape = _shape(node)
        weight_shape = _shape(weight_node)
        if input_shape is None or output_shape is None or weight_shape is None:
            continue
        if len(input_shape) < 2 or len(output_shape) < 2 or len(weight_shape) != 2:
            continue

        M = 1
        for dim in output_shape[:-1]:
            M *= dim
        K = input_shape[-1]
        N = output_shape[-1]
        if weight_shape != [N, K]:
            continue

        graph.inserting_before(node)

        input_2d = input_node
        if len(input_shape) != 2:
            input_2d = graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, [M, K]),
            )
            input_2d.name = _create_name(graph, f"{node.name}_input_2d")
            input_2d.meta["shape"] = [M, K]
            input_2d.meta["dtype"] = input_node.meta.get("dtype", torch.float32)
            _set_phase(input_2d, node.meta.get("phase", "forward"))
            _copy_region_meta(input_2d, node)

        weight_t = graph.call_function(torch.ops.aten.t.default, (weight_node,))
        weight_t.name = _create_name(graph, f"{node.name}_weight_t")
        weight_t.meta["shape"] = [K, N]
        weight_t.meta["dtype"] = weight_node.meta.get("dtype", torch.float32)
        _set_phase(weight_t, node.meta.get("phase", "forward"))
        _copy_region_meta(weight_t, node)

        if bias_node is None:
            matmul = graph.call_function(torch.ops.aten.mm.default, (input_2d, weight_t))
            matmul.name = _create_name(graph, "mm")
        else:
            matmul = graph.call_function(
                torch.ops.aten.addmm.default,
                (bias_node, input_2d, weight_t),
            )
            matmul.name = _create_name(graph, "addmm")
        matmul.meta["shape"] = [M, N]
        matmul.meta["dtype"] = node.meta.get("dtype", torch.float32)
        matmul.meta["disable_memory_pool"] = True
        _set_phase(matmul, node.meta.get("phase", "forward"))
        _copy_region_meta(matmul, node)

        replacement = matmul
        if len(output_shape) != 2:
            replacement = graph.call_function(
                torch.ops.aten.reshape.default,
                (matmul, list(output_shape)),
            )
            replacement.name = _create_name(graph, "reshape")
            replacement.meta["shape"] = list(output_shape)
            replacement.meta["dtype"] = node.meta.get("dtype", torch.float32)
            _set_phase(replacement, node.meta.get("phase", "forward"))
            _copy_region_meta(replacement, node)

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)
