# mypy: allow-untyped-defs
"""
Custom-op-based FSDP2 hooks for torch.compile compatibility.

The standard FSDP2 forward hooks use ``torch._dynamo.disable`` which causes
graph breaks under ``fullgraph=True``.  This module replaces those hooks with
Dynamo-traceable hooks that delegate to custom ops (``fsdp_compile::unshard``,
``fsdp_compile::reshard``, ``fsdp_compile::post_backward``).  The custom ops
are opaque to Dynamo and appear as single nodes in the FX graph.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ._fsdp_common import is_bw, TrainingState
from ._fsdp_param_group import FSDPParamGroup
from ._fsdp_state import FSDPState


# ── Context registry ──
# Maps stable integer IDs to FSDPParamGroups.  IDs are assigned at
# install time and reused across compiled invocations (Dynamo bakes them
# as constants).

_ctx_store: dict[int, FSDPParamGroup] = {}
_ctx_counter = 0


def _register_pg(pg: FSDPParamGroup) -> int:
    global _ctx_counter
    _ctx_counter += 1
    _ctx_store[_ctx_counter] = pg
    return _ctx_counter


def _get_pg(ctx_id: int) -> FSDPParamGroup | None:
    return _ctx_store.get(ctx_id, None)


# ── Shape encoding helpers ──
# Custom ops only accept tensors and primitives.  We encode the original
# (unsharded) parameter shapes as flat int lists so the ``register_fake``
# implementations can reconstruct them.

def _encode_orig_sizes(fsdp_params):
    flat: list[int] = []
    ndims: list[int] = []
    for fp in fsdp_params:
        mi = fp._module_info
        orig = getattr(mi.module, mi.param_name).shape
        flat.extend(list(orig))
        ndims.append(len(orig))
    return flat, ndims


def _decode_orig_sizes(flat, ndims):
    sizes: list[torch.Size] = []
    idx = 0
    for nd in ndims:
        sizes.append(torch.Size(flat[idx : idx + nd]))
        idx += nd
    return sizes


# ── Custom ops ──

@torch.library.custom_op("fsdp_compile::unshard", mutates_args=())
def unshard_op(
    sharded_datas: list[torch.Tensor],
    ctx_id: int,
    group_size: int,
    orig_sizes_flat: list[int],
    orig_ndims: list[int],
    sharded_numels: list[int],
) -> list[torch.Tensor]:
    pg = _get_pg(ctx_id)
    if pg is not None:
        pg._training_state = TrainingState.FORWARD
        pg.unshard(getattr(pg, "unshard_async_op", False))
        pg.wait_for_unshard()
        return [
            getattr(fp._module_info.module, fp._module_info.param_name).data.clone()
            for fp in pg.fsdp_params
        ]
    orig_sizes = _decode_orig_sizes(orig_sizes_flat, orig_ndims)
    return [sd.new_zeros(orig) for sd, orig in zip(sharded_datas, orig_sizes)]


@unshard_op.register_fake
def _unshard_fake(sharded_datas, ctx_id, group_size, orig_sizes_flat, orig_ndims, sharded_numels):
    orig_sizes = _decode_orig_sizes(orig_sizes_flat, orig_ndims)
    return [sd.new_empty(orig) for sd, orig in zip(sharded_datas, orig_sizes)]


@torch.library.custom_op("fsdp_compile::post_backward", mutates_args=())
def post_backward_op(
    grad_unshardeds: list[torch.Tensor],
    ctx_id: int,
    group_size: int,
    sharded_numels: list[int],
) -> list[torch.Tensor]:
    pg = _get_pg(ctx_id)
    if pg is not None:
        for fp, g in zip(pg.fsdp_params, grad_unshardeds):
            if hasattr(fp, "_unsharded_param"):
                fp.unsharded_param.grad = g
        pg.post_backward()
        return [
            fp.sharded_param.grad.clone()
            if fp.sharded_param.grad is not None
            else g.new_zeros(sn)
            for fp, g, sn in zip(pg.fsdp_params, grad_unshardeds, sharded_numels)
        ]
    return [g.clone() for g in grad_unshardeds]


@post_backward_op.register_fake
def _post_backward_fake(grad_unshardeds, ctx_id, group_size, sharded_numels):
    return [g.new_empty(g.shape) for g in grad_unshardeds]


def _unshard_setup_context(ctx, inputs, output):
    sharded_datas, ctx_id, group_size, orig_sizes_flat, orig_ndims, sharded_numels = inputs
    ctx.group_size = group_size
    ctx.sharded_numels = sharded_numels
    ctx.input_shapes = [sd.shape for sd in sharded_datas]


def _unshard_backward(ctx, grad_unshardeds):
    sharded_grads = torch.ops.fsdp_compile.post_backward(
        grad_unshardeds, 0, ctx.group_size, ctx.sharded_numels,
    )
    sharded_grads = [
        g.view(shape) for g, shape in zip(sharded_grads, ctx.input_shapes)
    ]
    return sharded_grads, None, None, None, None, None


unshard_op.register_autograd(
    _unshard_backward, setup_context=_unshard_setup_context
)


@torch.library.custom_op("fsdp_compile::reshard", mutates_args=())
def reshard_op(
    unshardeds: list[torch.Tensor],
    ctx_id: int,
    sharded_numels: list[int],
) -> list[torch.Tensor]:
    pg = _get_pg(ctx_id)
    if pg is not None:
        pg.reshard()
        pg._record_post_forward()
        pg._training_state = TrainingState.IDLE
        return [fp._sharded_param_data.clone() for fp in pg.fsdp_params]
    return [u.new_zeros(sn) for u, sn in zip(unshardeds, sharded_numels)]


@reshard_op.register_fake
def _reshard_fake(unshardeds, ctx_id, sharded_numels):
    return [u.new_empty(sn) for u, sn in zip(unshardeds, sharded_numels)]


# ── Hook helpers ──

def _get_param_groups(state: FSDPState) -> list[FSDPParamGroup]:
    if hasattr(state, "_fsdp_param_groups"):
        return state._fsdp_param_groups
    pg = state._fsdp_param_group
    return [pg] if pg is not None else []


def _unshard_via_custom_op(pg: FSDPParamGroup, ctx_id: int) -> list[torch.Tensor]:
    try:
        group_size = pg._all_gather_process_group.size()
    except AttributeError:
        group_size = 1
    orig_sizes_flat, orig_ndims = _encode_orig_sizes(pg.fsdp_params)
    sharded_numels = [fp._sharded_param_data.numel() for fp in pg.fsdp_params]
    sharded_params = []
    for fp in pg.fsdp_params:
        p = getattr(fp._module_info.module, fp._module_info.param_name)
        if isinstance(p, DTensor):
            sharded_params.append(p.to_local())
        else:
            sharded_params.append(p)

    unshardeds = torch.ops.fsdp_compile.unshard(
        sharded_params, ctx_id, group_size, orig_sizes_flat, orig_ndims, sharded_numels,
    )
    for fp, u in zip(pg.fsdp_params, unshardeds):
        mi = fp._module_info
        mi.module._parameters[mi.param_name] = u
        for sm, sn in zip(mi.shared_modules, mi.shared_param_names):
            sm._parameters[sn] = u
    return unshardeds


def _reshard_via_custom_op(
    pg: FSDPParamGroup, unshardeds: list[Any], ctx_id: int
) -> None:
    sharded_numels = [fp._sharded_param_data.numel() for fp in pg.fsdp_params]
    torch.ops.fsdp_compile.reshard(unshardeds, ctx_id, sharded_numels)


# ── Dynamo-traceable replacement hooks ──

def _create_pre_forward_hook(
    state: FSDPState, pg_ctx_ids: list[tuple[FSDPParamGroup, int]]
):
    def hook(module: nn.Module, args, kwargs):
        state._lazy_init()
        state._training_state = TrainingState.FORWARD
        for pg, ctx_id in pg_ctx_ids:
            if pg is not None and pg.fsdp_params:
                pg.lazy_init()
                pg._training_state = TrainingState.FORWARD
                _unshard_via_custom_op(pg, ctx_id)
        return args, kwargs
    return hook


def _create_post_forward_hook(
    state: FSDPState, pg_ctx_ids: list[tuple[FSDPParamGroup, int]]
):
    def hook(module: nn.Module, input, output):
        for pg, ctx_id in pg_ctx_ids:
            if pg is not None and pg.fsdp_params:
                in_forward = torch.compiler.is_compiling() or not is_bw()
                if in_forward:
                    unshardeds = [
                        getattr(fp._module_info.module, fp._module_info.param_name)
                        for fp in pg.fsdp_params
                    ]
                    _reshard_via_custom_op(pg, unshardeds, ctx_id)
                pg._training_state = TrainingState.IDLE
        state._training_state = TrainingState.IDLE
        return output
    return hook


# ── Hook identification ──

def _is_fsdp_hook(hook_fn, method_name: str) -> bool:
    if not isinstance(hook_fn, types.MethodType):
        return False
    fn_name = getattr(hook_fn.__func__, "__name__", "")
    if fn_name == method_name and hasattr(hook_fn.__self__, "_fsdp_param_groups"):
        return True
    wrapped = getattr(hook_fn.__func__, "__wrapped__", None)
    if wrapped is not None and getattr(wrapped, "__name__", "") == method_name:
        return hasattr(hook_fn.__self__, "_fsdp_param_groups")
    return False


# ── Public API ──

_hook_originals: dict = {}


def install_fsdp_custom_ops(model: nn.Module) -> None:
    """Replace FSDP2 forward hooks with compile-friendly custom-op hooks.

    Must be called after ``fully_shard`` and after one eager forward pass
    (to trigger lazy initialization).  Call before ``torch.compile``.
    """
    _ctx_store.clear()
    _hook_originals.clear()
    for module in model.modules():
        for hook_id in list(module._forward_pre_hooks.keys()):
            hook_fn = module._forward_pre_hooks[hook_id]
            if _is_fsdp_hook(hook_fn, "_pre_forward"):
                state = hook_fn.__self__
                pg_ctx_ids = [
                    (pg, _register_pg(pg)) for pg in _get_param_groups(state)
                ]
                _hook_originals[(id(module), "pre", hook_id)] = hook_fn
                module._forward_pre_hooks[hook_id] = _create_pre_forward_hook(
                    state, pg_ctx_ids
                )
        for hook_id in list(module._forward_hooks.keys()):
            hook_fn = module._forward_hooks[hook_id]
            if _is_fsdp_hook(hook_fn, "_post_forward"):
                state = hook_fn.__self__
                pg_ctx_ids = [
                    (pg, _register_pg(pg)) for pg in _get_param_groups(state)
                ]
                _hook_originals[(id(module), "post", hook_id)] = hook_fn
                module._forward_hooks[hook_id] = _create_post_forward_hook(
                    state, pg_ctx_ids
                )


def uninstall_fsdp_custom_ops(model: nn.Module) -> None:
    """Restore the original FSDP2 hooks."""
    for module in model.modules():
        for hook_id in list(module._forward_pre_hooks.keys()):
            key = (id(module), "pre", hook_id)
            if key in _hook_originals:
                module._forward_pre_hooks[hook_id] = _hook_originals[key]
        for hook_id in list(module._forward_hooks.keys()):
            key = (id(module), "post", hook_id)
            if key in _hook_originals:
                module._forward_hooks[hook_id] = _hook_originals[key]
    _hook_originals.clear()
    _ctx_store.clear()
