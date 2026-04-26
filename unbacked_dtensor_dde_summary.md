# Analysis of `ops_unbacked_dtensor_dde` xfails in `test_dtensor_ops.py`

This summarizes where data-dependent errors (DDEs) come from for the ~142 ops
xfailed in `ops_unbacked_dtensor_dde` (test/distributed/tensor/test_dtensor_ops.py:805-947).

These ops are distinct from `ops_dde_xfail` (common_ops_unbacked.py) which are
base-tensor-level DDEs independent of DTensor. The ops here pass with base
tensors but fail when wrapped as DTensors with unbacked symbolic dimensions.

## Error taxonomy

### DTensor-internal DDEs (fixable within DTensor)

#### 1. `infer_broadcast_dims_map` -- 51 ops (BIGGEST GROUP)

**Location:** `torch/distributed/tensor/_ops/utils.py:286`
**Expression:** `if input_shape[idx] == common_shape[idx]:`
**Cause:** Direct equality comparison between potentially unbacked symbolic
shape dimensions during broadcast dimension mapping. When one dim is literal 1
(broadcast) and the other is unbacked, this triggers a DDE.
**Affected ops:** All elementwise binary ops with broadcasting -- add, mul, sub,
div, eq, gt, lt, pow, where, xlogy, copysign, fmod, etc.
**Fix:** Use `guard_or_false` or `statically_known_true` instead of `==`.

#### 2. `_try_replicate_spec_for_scalar_tensor` -- 6 ops

**Location:** `torch/distributed/tensor/_dispatch.py:674,683`
**Expression:** `tensor_arg.numel() == 1`
**Cause:** Checks if a non-DTensor tensor argument is scalar to auto-replicate
it. With unbacked dims, `numel()` is symbolic.
**Affected ops:** nn.functional.interpolate.{bicubic,bilinear,linear,nearest,nearest-exact,trilinear}
**Fix:** Use `guard_or_false` on the numel check.

#### 3. `is_tensor_shardable` -- 5 ops

**Location:** `torch/distributed/tensor/_ops/utils.py:198-220`
**Expression:** `guard_fn(shape[shard_dim] < num_shards[shard_dim])` where
`guard_fn = bool` when `allow_unbacked_sharding=None` (the default).
**Cause:** `expand_to_full_mesh_op_strategy` (line 486) calls
`is_tensor_shardable(inp.shape, s)` without setting `allow_unbacked_sharding`,
so it defaults to `None` which uses bare `bool()` on the symbolic comparison.
**Affected ops:** argsort, histc, msort, sort, topk
**Fix:** Pass `allow_unbacked_sharding=True` from callers, or change the
default from `None` to `True`.

#### 4. `new_factory_strategy` -- 5 ops

**Location:** `torch/distributed/tensor/_ops/_tensor_ops.py:251`
**Expression:** `if tuple(input_shape) == tuple(output_shape) and input_spec.is_sharded():`
**Cause:** Tuple comparison with unbacked symbolic ints.
**Affected ops:** new_empty, new_empty_strided, new_full, new_ones, new_zeros
**Fix:** Use `statically_known_true` for the shape comparison.

#### 5. `gen_single_dim_einsum_strategies` (bias shape) -- 3 ops

**Location:** `torch/distributed/tensor/_ops/_matrix_ops.py:324`
**Expression:** `if bias_shape[bias_dim_idx] == 1:`
**Cause:** Direct comparison of bias dimension size to 1 for broadcast detection.
**Affected ops:** addmm, addmm.decomposed, nn.functional.linear
**Fix:** Use `guard_or_false` on the shape comparison.

#### 6. `argminmax_handler` -- 2 ops

**Location:** `torch/distributed/tensor/_nonlinear_redux.py:222`
**Expression:** `local_tensor.flatten()[local_idx]`
**Cause:** `extract_int` DDE -- indexing with an unbacked symint index
(the result of argmax/argmin is itself data-dependent).
**Affected ops:** argmax, argmin
**Fix:** Harder to fix -- the index value is inherently data-dependent.

#### 7. `as_strided_handler` -- 2 ops

**Location:** `torch/distributed/tensor/_dispatch.py:62-64`
**Expression:** `tensor.size() == tuple(size) and tensor.stride() == tuple(stride)`
**Cause:** Direct size/stride comparison with potentially symbolic values.
**Affected ops:** as_strided, as_strided.partial_views
**Fix:** Use `statically_known_true` for the comparisons.

#### 8. `gather_strategy` -- 1 op

**Location:** `torch/distributed/tensor/_ops/_tensor_ops.py:681`
**Expression:** `index_shape[dim] == 1`
**Cause:** Direct shape comparison for gather strategy selection.
**Affected ops:** gather
**Fix:** Use `guard_or_false` or `statically_known_true`.

#### 9. `normalize_dim` -- 1 op

**Location:** `torch/distributed/tensor/_ops/utils.py:154`
**Affected ops:** narrow

### Meta-kernel DDEs (NOT DTensor-specific, arguably should be in `ops_dde_xfail`)

These DDEs originate in PyTorch core meta kernels / decompositions, not DTensor
code. They happen to surface through the DTensor test because DTensor dispatches
through these meta kernels during tracing.

#### 10. `_padding_check_valid_input` -- 3 ops

**Location:** `torch/_meta_registrations.py:1859,1862,1894`
**Expression:** `input.size(d) != 0` and related padding validity checks.
**Affected ops:** nn.functional.pad.reflect, nn.functional.pad.replicate,
nn.functional.pad.replicate_negative

#### 11. `_compute_scale` -- 4 ops

**Location:** `torch/_decomp/decompositions.py:4009`
**Cause:** Interpolation scale computation with unbacked symbolic sizes.
**Affected ops:** nn.functional.interpolate.{bicubic,bilinear,linear,trilinear}

#### 12. `interpolate` / `adaptive_avg_pool` -- 2 ops

**Location:** `torch/nn/functional.py:4836,1399,1415`
**Cause:** extract_int or adaptive pool dispatch with symbolic sizes.
**Affected ops:** nn.functional.interpolate.area (multiple errors)

#### 13. `run_node` -- 1 op

**Location:** `torch/_dynamo/utils.py:3815`
**Affected ops:** nn.functional.pad.circular

### Non-DDE errors (different root causes, misclassified in this xfail set)

#### 14. GraphBreak -- 21 ops

Not DDEs at all. These are graph breaks during `fullgraph=True` compilation,
typically from dynamo/decomposition issues.
**Affected ops:** __rsub__, _segment_reduce.{lengths,offsets}, _unsafe_masked_index,
addr, argwhere, block_diag, cumprod, floor_divide, masked_scatter, masked_select,
nn.functional.{cosine_embedding_loss,hinge_embedding_loss,logsigmoid,
margin_ranking_loss,multilabel_soft_margin_loss}, nn.functional.pad.replicate,
nonzero_static, rsub, squeeze.multiple

#### 15. MixedTensorDTensor -- 7 ops

Raised by `_try_replicate_spec_for_scalar_tensor` when a non-DTensor tensor
with numel > 1 is passed alongside DTensors. Not a DDE.
**Affected ops:** __getitem__, nn.functional.interpolate.{bicubic,bilinear,
linear,nearest,nearest-exact,trilinear}

#### 16. PendingUnbackedSymbolNotFound -- 2 ops (slice)

Unbacked symbols get lost during the slice operation output.
**Affected ops:** slice (2 samples)

#### 17. ShardingProp misc -- 2 ops

**Affected ops:** __getitem__ (unknown error), index_select ("max() arg is an
empty sequence")

### Ops that now PASS (false xfails, can be removed)

Many ops in the xfail list currently pass all their samples:
__rmatmul__, _softmax_backward_data, alias_copy, bmm, cartesian_prod, cat,
constant_pad_nd, dstack, fill, flatten, hstack, matmul, mm, mv,
nn.functional.{celu,elu,hardsigmoid,hardtanh,mish,poisson_nll_loss,relu6,selu,
softplus,triplet_margin_loss,triplet_margin_with_distance_loss}, permute_copy,
prod, ravel, reshape, reshape_as, slice_scatter, squeeze, transpose_copy,
unflatten, unsqueeze_copy, vdot, view, view_as, view_copy, vstack

## Summary by fixability

| Fix | Ops | Location | Approach |
|-----|-----|----------|----------|
| `infer_broadcast_dims_map` | 51 | utils.py:286 | `guard_or_false` on shape comparison |
| `_try_replicate_spec_for_scalar_tensor` | 6 | _dispatch.py:674 | `guard_or_false` on numel check |
| `is_tensor_shardable` callers | 5 | utils.py:486 | Pass `allow_unbacked_sharding=True` |
| `new_factory_strategy` | 5 | _tensor_ops.py:251 | `statically_known_true` for shape eq |
| `gen_single_dim_einsum_strategies` | 3 | _matrix_ops.py:324 | `guard_or_false` on bias shape |
| `as_strided_handler` | 2 | _dispatch.py:62 | `statically_known_true` for size/stride |
| `gather_strategy` | 1 | _tensor_ops.py:681 | `guard_or_false` on index shape |
| `normalize_dim` | 1 | utils.py:154 | Guard the dim comparison |
| Remove false xfails | ~40 | test_dtensor_ops.py | Remove from xfail set |
| Move to `ops_dde_xfail` | ~10 | test_dtensor_ops.py | Meta-kernel DDEs belong there |
| GraphBreak / MixedTensor | ~30 | various | Different error category |

Fixing just `infer_broadcast_dims_map` (one line change) would resolve 51 ops.
The top 6 DTensor-internal fixes would resolve ~73 ops total.
