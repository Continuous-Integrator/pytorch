# OpenReg Test File Classification

Files are classified as:
- **Core**: Include in openreg CI (don't blocklist)
- **Later**: Blocklist for now, revisit in a future pass
- **Don't care**: Blocklist permanently

## Currently in allowlist (33 files)

All core. These run in `run_openreg_tests.py`:

```
nn/test_convolution.py
nn/test_dropout.py
nn/test_embedding.py
nn/test_init.py
nn/test_multihead_attention.py
nn/test_parametrization.py
nn/test_pooling.py
test_autograd.py
test_binary_ufuncs.py
test_custom_ops.py
test_dataloader.py
test_indexing.py
test_modules.py
test_native_mha.py
test_nn.py
test_ops.py
test_ops_fwd_gradients.py
test_ops_gradients.py
test_optim.py
test_reductions.py
test_scatter_gather_ops.py
test_segment_reductions.py
test_serialization.py
test_shape_ops.py
test_sort_and_select.py
test_tensor_creation_ops.py
test_testing.py
test_torch.py
test_transformers.py
test_type_promotion.py
test_unary_ufuncs.py
test_utils.py
test_view_ops.py
```

## Core (not yet in allowlist)

```
nn/test_packed_sequence.py          # has device logic (CUDA .to() tests), not device-generic
test_numpy_interop.py               # device-generic, NumPy interop
test_python_dispatch.py             # device-generic, __torch_dispatch__ tests
```

## Later — by theme

### PT2 Infra
```
test_content_store.py               # content-addressable storage (AOTAutograd)
test_decomp.py                      # operator decompositions
test_fake_tensor.py                 # FakeTensor / meta tensor
test_functionalization_of_rng_ops.py # RNG op functionalization
test_fx.py                          # torch.fx tracing
test_ops_unbacked.py                # ops with unbacked SymInts
test_proxy_tensor.py                # ProxyTensor tracing
test_dynamic_shapes.py              # dynamic shape / SymInt
test_compile_benchmark_util.py      # compile benchmark utils
```

### Revisit for device-genericity
```
test_matmul_cuda.py                 # need to check if sufficiently device-generic
test_scaled_matmul_cuda.py          # need to check if sufficiently device-generic
test_varlen_attention.py            # CUDA-specific (?)
test_cpp_extensions_stream_and_event.py  # revisit for openreg relevance
```

### Low priority
```
test_complex.py                     # complex tensor ops
test_linalg.py                      # linear algebra ops
test_spectral_ops.py                # FFT / spectral ops
test_foreach.py                     # torch._foreach_* fused ops
test_prims.py                       # primitive ops
test_dlpack.py                      # DLPack tensor interop
test_fx_experimental.py             # experimental fx features
```

### Autograd / functional
```
test_autograd_fallback.py           # autograd fallback tests
test_functional_autograd_benchmark.py
test_functionalization.py
```

### Module / misc
```
test_opaque_obj_v2.py               # custom ops with opaque objects
test_stateless.py                   # functional_call / stateless module
test_out_dtype_op.py                # out_dtype custom op
test_torchfuzz_repros.py            # fuzzer-found bug reproductions
nn/test_lazy_modules.py             # lazy module init (low-pri)
nn/test_load_state_dict.py          # state dict loading (no accelerator logic)
nn/test_module_hooks.py             # module hooks (no accelerator logic)
```

### Categories (later)
```
distributed/                        # FSDP, DTensor, NCCL, etc.
dynamo/                             # torch.compile internals
inductor/                           # codegen, triton, compiled ops
export/                             # torch.export
profiler/                           # profiler/tracing
cpp_extensions/                     # C++ extension tests
benchmark_utils/                    # benchmark utilities
distributions/                     # probability distributions
higher_order_ops/                   # cond, while_loop, map, scan
```

### Utils (later)
```
test_module_tracker.py              # module tracking
test_flop_counter.py                # FLOP counting
```

## Core (needs refactoring to be device-generic)

These are core test areas but don't yet use `instantiate_device_type_tests`.
They need refactoring before they can run against openreg.

```
autograd/                           # autograd functional/complex/logging (0 DG files)
custom_operator/                    # torch.library custom op registration (0 DG files)
optim/                              # optimizer subdirectory tests (0 DG files)
test_autocast.py                    # device-parameterized autocast
test_extension_utils.py             # references PrivateUse1
```

## Don't care

### Hardware-specific
```
test_cuda.py                        # CUDA-specific
test_cuda_compatibility.py
test_cuda_expandable_segments.py
test_cuda_multigpu.py
test_cuda_nvml_based_avail.py
test_cuda_primary_ctx.py
test_cuda_sanitizer.py
test_cuda_trace.py
test_jiterator.py                   # CUDA jiterator
test_kernel_launch_checks.py        # CUDA kernel launch validation
test_matmul_cuda.py                 # (also in later/revisit)
test_metal.py                       # Apple Metal
test_mkldnn.py                      # Intel MKL-DNN
test_mps.py                         # Apple MPS
test_vulkan.py                      # Vulkan
test_xpu.py                         # Intel XPU
test_xpu_expandable_segments.py
test_numba_integration.py           # Numba/CUDA interop
test_sparse_semi_structured.py      # 2:4 sparsity (CUDA-specific)
test_mkl_verbose.py
test_mkldnn_fusion.py
test_mkldnn_verbose.py
test_xnnpack_integration.py
test_nnapi.py
test_openmp.py
test_numa_binding.py
```

### JIT / TorchScript (legacy)
```
test_jit.py
test_jit_autocast.py
test_jit_disabled.py
test_jit_fuser.py
test_jit_fuser_legacy.py
test_jit_fuser_te.py
test_jit_legacy.py
test_jit_llga_fuser.py
test_jit_profiling.py
test_jit_simple.py
test_jit_string.py
test_ops_jit.py
jit/                                # entire directory
```

### Legacy features
```
test_masked.py                      # masked tensor ops (legacy)
test_maskedtensor.py                # MaskedTensor subclass (legacy)
test_nestedtensor.py                # nested tensor (legacy)
test_legacy_vmap.py                 # legacy vmap
test_namedtensor.py                 # named tensors (legacy)
test_sparse.py                      # sparse COO tensors (legacy)
test_sparse_csr.py                  # sparse CSR/CSC/BSR/BSC (legacy)
nn/test_pruning.py                  # module pruning
```

### Core infra (test once on CPU)
```
test_dispatch.py                    # dispatch table mechanics (cuda refs are strings)
test_overrides.py                   # __torch_function__ overrides
test_schema_check.py                # op schema validation
test_meta.py                        # meta tensor dispatch
test_expanded_weights.py            # per-sample gradients
test_function_schema.py             # schema validation
test_native_functions.py            # native function validation
test_per_overload_api.py            # per-overload API
test_accelerator.py                 # generic accelerator API
test_privateuseone_python_backend.py # PrivateUse1 infra
test_rename_privateuse1_to_existing_device.py
test_public_bindings.py             # API binding validation (cuda refs are strings)
test_comparison_utils.py            # tensor metadata assertion utils
test_subclass.py                    # tensor subclass (CPU-only)
test_pytree.py                      # pytree (CPU-only)
test_type_hints.py                  # type annotations (CPU-only)
test_type_info.py                   # type info (CPU-only)
test_typing.py                      # typing (CPU-only)
```


### Other don't care
```
functorch/                          # entire directory
quantization/                       # entire directory
onnx/                               # entire directory
xpu/                                # entire directory
complex_tensor/                     # entire directory
lazy/                               # entire directory
ao/                                 # entire directory
backends/                           # entire directory
custom_backend/                     # entire directory
fx/                                 # entire directory (all CPU-only)
mobile/                             # entire directory
package/                            # entire directory
python_native/                      # entire directory
torch_np/                           # entire directory
typing/                             # entire directory
```

### No device logic at all
```
test_ao_sparsity.py
test_appending_byte_serializer.py
test_as_strided.py
test_autoload.py
test_bundled_images.py
test_bundled_inputs.py
test_ci_sanity_check_fail.py
test_datapipe.py
test_determination.py
test_file_check.py
test_functional_optim.py
test_futures.py
test_hop_infra.py
test_hub.py
test_import_stats.py
test_itt.py
test_license.py
test_logging.py
test_model_exports_to_core_aten.py
test_monitor.py
test_multiprocessing.py
test_multiprocessing_spawn.py
test_namedtuple_return_api.py
test_package.py
test_pruning_op.py
test_quantization.py
test_set_default_mobile_cpu_allocator.py
test_show_pickle.py
test_sympy_utils.py
test_tensorboard.py
test_tensorexpr.py
test_tensorexpr_pybind.py
test_throughput_benchmark.py
test_torch_config_hash_determinism.py
test_utils_config_module.py
test_utils_filelock.py
test_weak.py
test_cpp_api_parity.py
test_cpp_extensions_aot.py
test_cpp_extensions_jit.py
test_cpp_extensions_mtia_backend.py
test_static_runtime.py
test_mobile_optimizer.py
test_fx_graph_print.py
test_fx_passes.py
test_fx_reinplace_pass.py
```
