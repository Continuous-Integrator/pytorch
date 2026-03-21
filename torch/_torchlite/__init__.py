"""Torchlite: a from-scratch compiler for PyTorch models.

The compiler has three phases:
  trace()      - capture a model into an FX graph
  run_passes() - run all graph transformation passes
  codegen()    - convert the transformed graph into a callable
compile() = trace() + run_passes() + codegen().
"""

from torch._torchlite import passes  # noqa: F401
from torch._torchlite.api import (
    codegen,
    codegen_inference,
    compile,
    default_passes,
    inference_passes,
    precompile_load,
    precompile_save,
    run_passes,
    timed_run_passes,
    trace,
)
from torch._torchlite.passes import (
    activation_checkpoint,
    annotate_dtensor,
    attention_canonicalize,
    autograd_per_op,
    canonicalize_layouts,
    canonicalize_pointwise_kwargs,
    cudagraph_partition,
    decompose,
    decompose_attention_projections,
    decompose_inference,
    dynamize,
    expand_gqa_projections,
    extract_attention_regions,
    extract_ffn_regions,
    fuse_packed_silu_mul,
    fsdp_unwrap,
    functionalize,
    fuse,
    fuse_matmul_add_rms_norm,
    FusedKernel,
    FusedOp,
    FusionGroup,
    MatmulAddRmsNormKernel,
    memory_plan,
    normalize,
    optimizer,
    pack_parallel_linears,
    pack_parallel_matmuls,
    PassResult,
    precompile,
    rng_functionalize,
    save_activations,
    subclass_unwrap,
    triton_codegen,
    verify_graph,
)


__all__ = [
    # Entry points (trace → run_passes → codegen)
    "trace",
    "run_passes",
    "timed_run_passes",
    "default_passes",
    "inference_passes",
    "codegen",
    "codegen_inference",
    "compile",
    "precompile_save",
    "precompile_load",
    # Passes submodule
    "passes",
    # Pass result type
    "PassResult",
    # Individual graph passes (canonical pipeline order)
    "cudagraph_partition",
    "normalize",
    "verify_graph",
    "functionalize",
    "dynamize",
    "canonicalize_layouts",
    "canonicalize_pointwise_kwargs",
    "annotate_dtensor",
    "subclass_unwrap",
    "fsdp_unwrap",
    "extract_attention_regions",
    "extract_ffn_regions",
    "expand_gqa_projections",
    "fuse_packed_silu_mul",
    "pack_parallel_linears",
    "pack_parallel_matmuls",
    "attention_canonicalize",
    "autograd_per_op",
    "rng_functionalize",
    "save_activations",
    "activation_checkpoint",
    "optimizer",
    "memory_plan",
    "decompose",
    "decompose_attention_projections",
    "decompose_inference",
    "fuse",
    "fuse_matmul_add_rms_norm",
    "triton_codegen",
    "precompile",
    # Data types
    "FusedKernel",
    "FusedOp",
    "FusionGroup",
    "MatmulAddRmsNormKernel",
]
