"""Packing passes for sibling linear projections."""

import operator
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch._torchlite.ops import packed_silu_mul
from torch._torchlite.passes.common import (
    _aten_op_name,
    _copy_region_meta,
    _create_name,
    _deep_getattr,
    _set_phase,
    PassResult,
)
from torch.fx import GraphModule, Node


_LINEAR_TARGET = torch._C._nn.linear
_CAT_TARGET = torch.ops.aten.cat.default
_MM_TARGETS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
}
_PACKABLE_INPUT_VIEW_TARGETS = frozenset({
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.Tensor.reshape,
    torch.Tensor.view,
})
_LAYOUT_USER_TARGETS = frozenset({
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
_PACKING_BLOCKERS = frozenset({
    "add", "sub", "rsub", "mul", "div", "where",
    "gt", "ge", "lt", "le", "eq", "ne",
    "pow", "clamp", "clamp_min", "clamp_max",
    "sin", "cos", "neg", "abs",
    "relu", "sigmoid", "tanh",
    "rsqrt", "sqrt", "exp", "log", "reciprocal",
    "silu", "gelu",
})
_PACKING_NORM_TARGETS = {
    F.rms_norm,
    F.layer_norm,
}
_FFN_GATE_UP_ROLES = frozenset({"gate", "up"})
_CONSTANT_VIEW_TARGETS = frozenset({
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.Tensor.reshape,
    torch.Tensor.view,
})


def _node_shape(node: Node) -> Optional[list[int]]:
    if not isinstance(node, Node):
        return None
    return node.meta.get("shape")


def _linear_bias_arg(node: Node):
    return node.args[2] if len(node.args) > 2 else None


def _terminal_users(node: Node) -> list[Node]:
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

        if user.op == "call_function" and (
            user.target is operator.getitem or user.target in _LAYOUT_USER_TARGETS
        ):
            worklist.extend(user.users.keys())
            continue

        terminals.append(user)

    return terminals


def _has_blocking_terminal_users(node: Node) -> bool:
    for user in _terminal_users(node):
        if user.op == "output":
            continue
        if user.op != "call_function":
            return True
        if user.target in _PACKING_NORM_TARGETS:
            return True
        if _aten_op_name(user.target) in _PACKING_BLOCKERS:
            return True
    return False


def _is_same_ffn_gate_up_region(group: Sequence[Node]) -> bool:
    if len(group) != 2:
        return False

    region_id = group[0].meta.get("region_id")
    if region_id is None:
        return False

    roles = set()
    for node in group:
        if node.meta.get("region_kind") != "ffn":
            return False
        if node.meta.get("region_id") != region_id:
            return False
        role = node.meta.get("region_role")
        if role not in _FFN_GATE_UP_ROLES:
            return False
        roles.add(role)

    return roles == _FFN_GATE_UP_ROLES


def _can_pack_group(group: Sequence[Node]) -> bool:
    if len(group) < 2:
        return False

    first = group[0]
    first_shape = _node_shape(first)
    if first_shape is None or len(first_shape) < 1:
        return False

    phase = first.meta.get("phase", "forward")
    prefix = list(first_shape[:-1])
    has_bias = _linear_bias_arg(first) is not None
    allow_ffn_gate_up = _is_same_ffn_gate_up_region(group)

    for node in group:
        if node.op != "call_function" or node.target is not _LINEAR_TARGET:
            return False
        if node.meta.get("phase", "forward") != phase:
            return False
        shape = _node_shape(node)
        if shape is None or list(shape[:-1]) != prefix:
            return False
        if len(node.args) < 2 or not isinstance(node.args[1], Node):
            return False
        node_has_bias = _linear_bias_arg(node) is not None
        if node_has_bias != has_bias:
            return False
        if node_has_bias and not isinstance(_linear_bias_arg(node), Node):
            return False
        if _has_blocking_terminal_users(node) and not allow_ffn_gate_up:
            return False

    return True


def _matmul_input_arg(node: Node):
    if node.target == torch.ops.aten.mm.default:
        return node.args[0]
    return node.args[1]


def _matmul_weight_arg(node: Node):
    if node.target == torch.ops.aten.mm.default:
        return node.args[1]
    return node.args[2]


def _matmul_bias_arg(node: Node):
    if node.target == torch.ops.aten.addmm.default:
        return node.args[0]
    return None


def _equivalent_input_key(node: Node):
    if not isinstance(node, Node):
        return None

    base = node
    while (
        base.op == "call_function"
        and base.target in _PACKABLE_INPUT_VIEW_TARGETS
        and base.args
        and isinstance(base.args[0], Node)
    ):
        base = base.args[0]

    shape = _node_shape(node)
    if shape is None:
        return None
    return (base, tuple(shape))


def _can_pack_matmul_group(group: Sequence[Node]) -> bool:
    if len(group) < 2:
        return False

    first = group[0]
    first_shape = _node_shape(first)
    first_input = _matmul_input_arg(first)
    first_input_shape = _node_shape(first_input) if isinstance(first_input, Node) else None
    first_input_key = _equivalent_input_key(first_input) if isinstance(first_input, Node) else None
    if (
        first_shape is None
        or len(first_shape) != 2
        or first_input_shape is None
        or len(first_input_shape) != 2
        or first_input_key is None
    ):
        return False

    phase = first.meta.get("phase", "forward")
    target = first.target
    rows = first_shape[0]
    K = first_input_shape[1]
    has_bias = _matmul_bias_arg(first) is not None
    allow_ffn_gate_up = _is_same_ffn_gate_up_region(group)

    for node in group:
        if node.op != "call_function" or node.target != target:
            return False
        if node.meta.get("phase", "forward") != phase:
            return False

        shape = _node_shape(node)
        if shape is None or len(shape) != 2 or shape[0] != rows:
            return False

        input_node = _matmul_input_arg(node)
        if (
            not isinstance(input_node, Node)
            or _node_shape(input_node) != first_input_shape
            or _equivalent_input_key(input_node) != first_input_key
        ):
            return False

        weight_node = _matmul_weight_arg(node)
        weight_shape = _node_shape(weight_node) if isinstance(weight_node, Node) else None
        if (
            not isinstance(weight_node, Node)
            or weight_shape is None
            or len(weight_shape) != 2
            or weight_shape[0] != K
        ):
            return False

        node_has_bias = _matmul_bias_arg(node) is not None
        if node_has_bias != has_bias:
            return False
        if node_has_bias and not isinstance(_matmul_bias_arg(node), Node):
            return False
        if _has_blocking_terminal_users(node) and not allow_ffn_gate_up:
            return False

    return True


def _materialize_packed_attr(
    gm: GraphModule,
    graph,
    name_hint: str,
    tensors: Sequence[torch.Tensor],
    dim: int,
) -> Node:
    attr_name = _create_name(graph, name_hint)
    packed = torch.cat(tuple(tensors), dim=dim)
    gm.register_buffer(attr_name, packed, persistent=False)
    node = graph.get_attr(attr_name)
    node.meta["shape"] = list(packed.shape)
    node.meta["dtype"] = packed.dtype
    return node


def _try_materialize_constant(gm: GraphModule, node: Node) -> Optional[torch.Tensor]:
    if node.op == "get_attr":
        tensor = _deep_getattr(gm, node.target)
        return tensor if isinstance(tensor, torch.Tensor) else None

    if node.op != "call_function":
        return None

    if node.target == torch.ops.aten.t.default:
        src = node.args[0] if node.args else None
        if not isinstance(src, Node):
            return None
        tensor = _try_materialize_constant(gm, src)
        return tensor.t() if tensor is not None else None

    if node.target == torch.ops.aten.transpose.int:
        src = node.args[0] if node.args else None
        if (
            not isinstance(src, Node)
            or len(node.args) < 3
            or not isinstance(node.args[1], int)
            or not isinstance(node.args[2], int)
        ):
            return None
        tensor = _try_materialize_constant(gm, src)
        if tensor is None:
            return None
        return tensor.transpose(node.args[1], node.args[2])

    if node.target in _CONSTANT_VIEW_TARGETS:
        src = node.args[0] if node.args else None
        if not isinstance(src, Node) or len(node.args) < 2:
            return None
        shape_arg = node.args[1]
        if not (
            isinstance(shape_arg, (list, tuple))
            and all(isinstance(dim, int) for dim in shape_arg)
        ):
            return None
        tensor = _try_materialize_constant(gm, src)
        if tensor is None:
            return None
        return tensor.reshape(tuple(shape_arg))

    if node.target == torch.ops.aten.contiguous.default:
        src = node.args[0] if node.args else None
        if not isinstance(src, Node):
            return None
        tensor = _try_materialize_constant(gm, src)
        return tensor.contiguous() if tensor is not None else None

    return None


def _make_packed_input(
    gm: GraphModule,
    graph,
    name_hint: str,
    nodes: Sequence[Node],
    dim: int,
    *,
    materialize_constants: bool,
) -> Node:
    if materialize_constants:
        tensors = []
        for node in nodes:
            tensor = _try_materialize_constant(gm, node)
            if tensor is None:
                break
            tensors.append(tensor)
        else:
            return _materialize_packed_attr(gm, graph, name_hint, tensors, dim)

    packed = graph.call_function(_CAT_TARGET, ([*nodes], dim))
    packed.name = _create_name(graph, name_hint)

    first_shape = _node_shape(nodes[0])
    if first_shape is not None:
        packed_shape = list(first_shape)
        packed_shape[dim] = sum(
            _node_shape(node)[dim] for node in nodes if _node_shape(node) is not None
        )
        packed.meta["shape"] = packed_shape
    packed.meta["dtype"] = nodes[0].meta.get("dtype", torch.float32)
    _set_phase(packed, nodes[0].meta.get("phase", "forward"))
    _copy_region_meta(packed, nodes[0], role="packed_input")
    return packed


def pack_parallel_linears(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    min_group_size: int = 2,
    materialize_constants: bool = False,
) -> PassResult:
    """Pack sibling linear projections that share the same input tensor.

    Rewrites groups such as Q/K/V or gate/up projections into one packed
    linear followed by slices along the last dimension.

    With ``materialize_constants=True``, constant weights/biases are packed
    once at compile time into non-persistent buffers on the GraphModule.
    Otherwise the pass emits runtime ``aten.cat`` nodes so gradients can flow.

    By default the pass avoids projections that immediately feed pointwise
    epilogues, but it makes an explicit exception for gate/up siblings that
    were tagged as the same FFN region.
    """

    graph = gm.graph
    groups: defaultdict[tuple[Node, str, bool], list[Node]] = defaultdict(list)

    for node in graph.nodes:
        if node.op != "call_function" or node.target is not _LINEAR_TARGET:
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        phase = node.meta.get("phase", "forward")
        has_bias = _linear_bias_arg(node) is not None
        groups[(input_node, phase, has_bias)].append(node)

    changed = False

    for (input_node, phase, has_bias), group in groups.items():
        if len(group) < min_group_size or not _can_pack_group(group):
            continue

        first = group[0]
        out_shapes = [_node_shape(node) for node in group]
        if any(shape is None for shape in out_shapes):
            continue

        out_dims = [shape[-1] for shape in out_shapes]
        packed_last_dim = sum(out_dims)
        packed_shape = list(out_shapes[0][:-1]) + [packed_last_dim]

        weight_nodes = [node.args[1] for node in group]
        bias_nodes = [_linear_bias_arg(node) for node in group] if has_bias else []

        graph.inserting_before(first)
        packed_weight = _make_packed_input(
            gm,
            graph,
            "packed_linear_weight",
            weight_nodes,
            0,
            materialize_constants=materialize_constants,
        )
        packed_bias = None
        if has_bias:
            packed_bias = _make_packed_input(
                gm,
                graph,
                "packed_linear_bias",
                bias_nodes,
                0,
                materialize_constants=materialize_constants,
            )

        packed_args = (input_node, packed_weight, packed_bias)
        packed_linear = graph.call_function(_LINEAR_TARGET, packed_args)
        packed_linear.name = _create_name(graph, "packed_linear")
        packed_linear.meta["shape"] = packed_shape
        packed_linear.meta["dtype"] = first.meta.get("dtype", torch.float32)
        _set_phase(packed_linear, phase)
        _copy_region_meta(packed_linear, first, role="packed_projection")

        offset = 0
        for node, out_dim, out_shape in zip(group, out_dims, out_shapes):
            index = tuple(
                [slice(None)] * (len(out_shape) - 1) + [slice(offset, offset + out_dim)]
            )
            slice_node = graph.call_function(operator.getitem, (packed_linear, index))
            slice_node.name = _create_name(graph, f"{node.name}_packed")
            slice_node.meta["shape"] = list(out_shape)
            slice_node.meta["dtype"] = node.meta.get("dtype", torch.float32)
            _set_phase(slice_node, phase)
            _copy_region_meta(slice_node, node)

            node.replace_all_uses_with(slice_node)
            offset += out_dim
            changed = True

        for node in reversed(group):
            if not node.users:
                graph.erase_node(node)

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)


def pack_parallel_matmuls(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    min_group_size: int = 2,
    materialize_constants: bool = False,
) -> PassResult:
    """Pack sibling mm/addmm projections that share one 2D input.

    This is the post-decomposition analogue of ``pack_parallel_linears`` and
    targets graphs produced from raw parameter matmuls like ``x @ wq`` /
    ``x @ wk`` / ``x @ wv``. Like the linear version, it makes a targeted
    exception for FFN gate/up siblings from the same marked region.
    """

    del example_inputs

    graph = gm.graph
    groups: defaultdict[tuple[object, object, str], list[Node]] = defaultdict(list)

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _MM_TARGETS:
            continue
        input_node = _matmul_input_arg(node)
        if not isinstance(input_node, Node):
            continue
        input_key = _equivalent_input_key(input_node)
        if input_key is None:
            continue
        phase = node.meta.get("phase", "forward")
        groups[(node.target, input_key, phase)].append(node)

    changed = False

    for (_, _input_key, phase), group in groups.items():
        if len(group) < min_group_size or not _can_pack_matmul_group(group):
            continue

        first = group[0]
        input_node = _matmul_input_arg(first)
        out_shapes = [_node_shape(node) for node in group]
        if any(shape is None for shape in out_shapes):
            continue

        out_dims = [shape[-1] for shape in out_shapes]
        packed_shape = [out_shapes[0][0], sum(out_dims)]

        weight_nodes = [_matmul_weight_arg(node) for node in group]
        bias_nodes = [_matmul_bias_arg(node) for node in group]
        has_bias = bias_nodes[0] is not None

        graph.inserting_before(first)
        packed_weight = _make_packed_input(
            gm,
            graph,
            "packed_mm_weight",
            weight_nodes,
            1,
            materialize_constants=materialize_constants,
        )
        packed_bias = None
        if has_bias:
            packed_bias = _make_packed_input(
                gm,
                graph,
                "packed_mm_bias",
                bias_nodes,
                0,
                materialize_constants=materialize_constants,
            )

        if first.target == torch.ops.aten.mm.default:
            packed_args = (input_node, packed_weight)
        else:
            packed_args = (packed_bias, input_node, packed_weight)

        packed_mm = graph.call_function(first.target, packed_args)
        packed_mm.name = _create_name(graph, "packed_mm")
        packed_mm.meta["shape"] = packed_shape
        packed_mm.meta["dtype"] = first.meta.get("dtype", torch.float32)
        _set_phase(packed_mm, phase)
        _copy_region_meta(packed_mm, first, role="packed_projection")

        offset = 0
        for node, out_dim, out_shape in zip(group, out_dims, out_shapes):
            index = (slice(None), slice(offset, offset + out_dim))
            slice_node = graph.call_function(operator.getitem, (packed_mm, index))
            slice_node.name = _create_name(graph, f"{node.name}_packed")
            slice_node.meta["shape"] = list(out_shape)
            slice_node.meta["dtype"] = node.meta.get("dtype", torch.float32)
            _set_phase(slice_node, phase)
            _copy_region_meta(slice_node, node)

            node.replace_all_uses_with(slice_node)
            offset += out_dim
            changed = True

        for node in reversed(group):
            if not node.users:
                graph.erase_node(node)

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)


def _match_packed_lastdim_half_slice(node: Node) -> Optional[tuple[Node, str, int]]:
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    if len(node.args) != 2:
        return None

    source, index = node.args
    if not isinstance(source, Node) or not isinstance(index, tuple):
        return None

    source_shape = _node_shape(source)
    slice_shape = _node_shape(node)
    if (
        source_shape is None
        or slice_shape is None
        or len(source_shape) < 1
        or len(index) != len(source_shape)
    ):
        return None

    packed_last_dim = source_shape[-1]
    if not isinstance(packed_last_dim, int) or packed_last_dim % 2 != 0:
        return None
    half = packed_last_dim // 2

    if any(dim != slice(None) for dim in index[:-1]):
        return None
    last = index[-1]
    if not isinstance(last, slice) or last.step is not None:
        return None

    if last.start == 0 and last.stop == half and slice_shape[-1] == half:
        return source, "gate", half
    if last.start == half and last.stop == packed_last_dim and slice_shape[-1] == half:
        return source, "up", half
    return None


def fuse_packed_silu_mul(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
) -> PassResult:
    """Collapse packed FFN gate/up slice chains into one fused op.

    Matches the common packed LLM FFN pattern:
    ``packed_proj -> reshape -> slice(gate) -> silu`` and
    ``packed_proj -> reshape -> slice(up)`` followed by ``mul``.
    Rewrites it to ``packed_silu_mul(packed_proj_reshaped)`` so later
    execution can lower the gate/up combine to a single CUDA kernel.
    """

    del example_inputs

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or _aten_op_name(node.target) != "mul"
            or len(node.args) != 2
            or node.meta.get("region_kind") != "ffn"
            or node.meta.get("region_role") != "combine"
        ):
            continue

        arg0, arg1 = node.args
        silu_node = None
        other_node = None
        if (
            isinstance(arg0, Node)
            and arg0.op == "call_function"
            and _aten_op_name(arg0.target) == "silu"
            and isinstance(arg1, Node)
        ):
            silu_node = arg0
            other_node = arg1
        elif (
            isinstance(arg1, Node)
            and arg1.op == "call_function"
            and _aten_op_name(arg1.target) == "silu"
            and isinstance(arg0, Node)
        ):
            silu_node = arg1
            other_node = arg0
        else:
            continue

        if not silu_node.args or not isinstance(silu_node.args[0], Node):
            continue
        gate_slice = silu_node.args[0]
        up_slice = other_node

        gate_match = _match_packed_lastdim_half_slice(gate_slice)
        up_match = _match_packed_lastdim_half_slice(up_slice)
        if gate_match is None or up_match is None:
            continue

        gate_source, gate_role, gate_half = gate_match
        up_source, up_role, up_half = up_match
        if (
            gate_source is not up_source
            or gate_role != "gate"
            or up_role != "up"
            or gate_half != up_half
        ):
            continue

        if gate_slice.meta.get("region_id") != up_slice.meta.get("region_id"):
            continue

        graph.inserting_before(node)
        fused_node = graph.call_function(packed_silu_mul, (gate_source,))
        fused_node.name = _create_name(graph, "packed_silu_mul")
        fused_node.meta["shape"] = list(node.meta.get("shape", []))
        fused_node.meta["dtype"] = node.meta.get("dtype", torch.float32)
        _set_phase(fused_node, node.meta.get("phase", "forward"))
        _copy_region_meta(fused_node, node)

        node.replace_all_uses_with(fused_node)
        graph.erase_node(node)
        if not silu_node.users:
            graph.erase_node(silu_node)
        if not gate_slice.users:
            graph.erase_node(gate_slice)
        if not up_slice.users:
            graph.erase_node(up_slice)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return PassResult(gm=gm, changed=changed)
