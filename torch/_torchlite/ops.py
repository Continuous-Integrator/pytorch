"""Custom callable ops that appear as nodes in the FX graph.

These are non-torch operations used by torchlite passes:
save_for_backward (activation checkpointing), RNG state management
(save_rng_state, load_rng_state), and optimizer steps (sgd_step,
adamw_step, param_update).
"""
import torch
import torch.nn.functional as F


def _named(name):
    def decorator(fn):
        fn.__name__ = name
        fn.__qualname__ = name
        return fn
    return decorator


def param_update(param, new_value):
    param.data.copy_(new_value)


@_named("sgd_step")
def sgd_step(param, grad, lr):
    param.data.add_(grad, alpha=-lr)


@_named("adamw_step")
def adamw_step(param, grad, m, v, step_t, lr, beta1, beta2, eps, weight_decay):
    # AdamW: decoupled weight decay + bias-corrected moment update.
    # step_t stays as a tensor (no .item()) so the entire computation
    # remains on-device and is capturable by CUDA graphs.
    if weight_decay != 0.0:
        param.data.mul_(1.0 - lr * weight_decay)
    m.data.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.data.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bc1 = torch.clamp(1.0 - beta1 ** step_t, min=1e-8)
    bc2 = torch.clamp(1.0 - beta2 ** step_t, min=1e-8)
    m_hat = m / bc1
    v_hat = v / bc2
    param.data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


@_named("save_rng_state")
def _save_rng_state():
    return torch.random.get_rng_state()


@_named("load_rng_state")
def _load_rng_state(state):
    torch.random.set_rng_state(state)


@_named("save_for_backward")
def _save_for_backward(x):
    return x


_packed_silu_mul_kernel = None


def _get_packed_silu_mul_kernel():
    global _packed_silu_mul_kernel
    if _packed_silu_mul_kernel is not None:
        return _packed_silu_mul_kernel

    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["H"],
    )
    @triton.jit
    def _kernel(
        packed_ptr,
        out_ptr,
        rows,
        H,
        stride_pr,
        stride_pc,
        stride_or,
        stride_oc,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_per_row = tl.cdiv(H, BLOCK_SIZE)
        row = pid // blocks_per_row
        block = pid % blocks_per_row
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < rows) & (offs < H)

        gate = tl.load(
            packed_ptr + row * stride_pr + offs * stride_pc,
            mask=mask,
            other=0.0,
        )
        up = tl.load(
            packed_ptr + row * stride_pr + (offs + H) * stride_pc,
            mask=mask,
            other=0.0,
        )
        gate_f = gate.to(tl.float32)
        up_f = up.to(tl.float32)
        out = gate_f * tl.sigmoid(gate_f) * up_f
        tl.store(
            out_ptr + row * stride_or + offs * stride_oc,
            out.to(gate.dtype),
            mask=mask,
        )

    _packed_silu_mul_kernel = _kernel
    return _packed_silu_mul_kernel


@_named("packed_silu_mul")
def packed_silu_mul(x):
    last_dim = x.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError("packed_silu_mul expects an even last dimension")

    H = last_dim // 2
    if x.is_cuda:
        kernel = _get_packed_silu_mul_kernel()
        if kernel is not None:
            rows = x.numel() // last_dim
            packed_2d = x.reshape(rows, last_dim)
            out = torch.empty((*x.shape[:-1], H), dtype=x.dtype, device=x.device)
            out_2d = out.reshape(rows, H)

            import triton

            grid = lambda meta, r=rows, h=H: (r * triton.cdiv(h, meta["BLOCK_SIZE"]),)  # noqa: E731
            kernel[grid](
                packed_2d,
                out_2d,
                rows,
                H,
                packed_2d.stride(0),
                packed_2d.stride(1),
                out_2d.stride(0),
                out_2d.stride(1),
            )
            return out

    gate, up = x.split(H, dim=-1)
    return F.silu(gate) * up


# FX codegen resolves call_function targets by __name__ / __qualname__
# within the module, so we need public aliases matching those names.
save_rng_state = _save_rng_state
load_rng_state = _load_rng_state
save_for_backward = _save_for_backward

_UNARY_POINTWISE_OPS = frozenset({
    "sin", "cos", "neg", "abs",
    "relu", "sigmoid", "tanh",
    "rsqrt", "sqrt", "exp", "log", "reciprocal",
    "silu", "gelu",
})
