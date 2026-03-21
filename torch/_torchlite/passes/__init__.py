"""FX graph passes for the torchlite compiler.

Every transformation after trace() is an FX graph pass with the signature
(gm, example_inputs, **kwargs) -> PassResult. This package contains all
passes that transform the graph, from initial verification through
decomposition, fusion, and code generation.

All public symbols are re-exported here so that existing imports
``from torch._torchlite.passes import X`` continue to work.
"""

# Re-export ops that passes.py used to re-export
from torch._torchlite.ops import (
    _load_rng_state,
    _save_for_backward,
    _save_rng_state,
    adamw_step,
    param_update,
    sgd_step,
)
from torch._torchlite.passes.attention import (
    attention_canonicalize,
    decompose_attention_projections,
    expand_gqa_projections,
    extract_attention_regions,
    extract_ffn_regions,
)
from torch._torchlite.passes.autograd import (
    _BackwardRecorder,
    _ForwardDecomposer,
    _storage_key,
    autograd_per_op,
)
from torch._torchlite.passes.checkpoint import activation_checkpoint, save_activations
from torch._torchlite.passes.common import (
    _aten_op_name,
    _create_name,
    _deep_getattr,
    _deep_setattr,
    _DUNDER_INPLACE,
    _DUNDER_TO_OP,
    _graph_meta,
    _graph_meta_store,
    _is_torch_op,
    _iter_node_args,
    _node_shape,
    _PROVENANCE_KEYS,
    _REVERSE_DUNDERS,
    _set_phase,
    _VARARGS_TENSOR_METHODS,
    FusedKernel,
    FusedOp,
    FusionGroup,
    MatmulAddRmsNormKernel,
    MatmulEpilogueKernel,
    PassResult,
)
from torch._torchlite.passes.cudagraph import (
    _CUDAGRAPH_NON_CAPTURABLE,
    cudagraph_partition,
)
from torch._torchlite.passes.decompose import (
    _DecompRecorder,
    decompose,
    decompose_inference,
)
from torch._torchlite.passes.dtensor import fsdp_unwrap, subclass_unwrap
from torch._torchlite.passes.dynamize import _align_reshape, dynamize
from torch._torchlite.passes.functionalize import (
    _find_functional_variant,
    functionalize,
)
from torch._torchlite.passes.fusion import (
    _POINTWISE_OPS,
    fuse,
    fuse_add_layer_norm,
    fuse_matmul_add_rms_norm,
    fuse_add_rms_norm,
    matmul_epilogue,
)
from torch._torchlite.passes.layout import canonicalize_layouts
from torch._torchlite.passes.memory import memory_plan
from torch._torchlite.passes.normalize import (
    _normalize_target,
    _set_dtensor_meta,
    annotate_dtensor,
    normalize,
    verify_graph,
)
from torch._torchlite.passes.optimizer_pass import _emit_sgd_update, optimizer
from torch._torchlite.passes.packing import (
    fuse_packed_silu_mul,
    pack_parallel_linears,
    pack_parallel_matmuls,
)
from torch._torchlite.passes.precompile import precompile
from torch._torchlite.passes.rng import rng_functionalize
from torch._torchlite.passes.sdpa import sdpa_pattern
from torch._torchlite.passes.simplify import (
    canonicalize_pointwise_kwargs,
    simplify_views,
)
from torch._torchlite.passes.training import (
    common_subexpression_elimination,
    decompose_training_backward,
)
from torch._torchlite.passes.triton import _TRITON_OP_MAP, triton_codegen, triton_lower


__all__ = [
    "PassResult",
    "FusedKernel",
    "FusedOp",
    "FusionGroup",
    "MatmulAddRmsNormKernel",
    "MatmulEpilogueKernel",
    "matmul_epilogue",
    "annotate_dtensor",
    "normalize",
    "verify_graph",
    "functionalize",
    "autograd_per_op",
    "activation_checkpoint",
    "save_activations",
    "optimizer",
    "dynamize",
    "decompose",
    "decompose_inference",
    "pack_parallel_linears",
    "pack_parallel_matmuls",
    "fuse_packed_silu_mul",
    "extract_attention_regions",
    "extract_ffn_regions",
    "expand_gqa_projections",
    "decompose_attention_projections",
    "canonicalize_layouts",
    "attention_canonicalize",
    "fuse",
    "fuse_matmul_add_rms_norm",
    "fuse_add_rms_norm",
    "fuse_add_layer_norm",
    "sdpa_pattern",
    "triton_codegen",
    "triton_lower",
    "cudagraph_partition",
    "precompile",
    "fsdp_unwrap",
    "subclass_unwrap",
    "memory_plan",
    "rng_functionalize",
    "canonicalize_pointwise_kwargs",
    "simplify_views",
    "decompose_training_backward",
    "common_subexpression_elimination",
]
