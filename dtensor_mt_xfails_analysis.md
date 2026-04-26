# TestMultiThreadedDTensorOps xfails categorization

Analysis of `dtensor_fails` and `dtensor_multi_threaded_fails` xfails (excluding
`dtensor_fails_no_strategy`, which are straightforward missing strategy registrations).

## Category 1: View/Reshape with Sharded Dimensions

`aten.view.default` strategy rejects flattening or splitting dimensions that include
a sharded dim, since it would require redistribution.

| Op | Error |
|---|---|
| `view` | "flatten multiple dimensions, with dimension N being sharded" |
| `view_as` | same |
| `reshape` | same (decomposes to `aten.view`) |
| `reshape_as` | same |
| `flatten` | same |
| `unflatten` | "split the sharded dimension 0 into multiple subdimensions" |
| `ravel` | flatten sharded dim |
| `take_along_dim` | flatten sharded dim (internally views) |
| `repeat_interleave` | flatten sharded dim |
| `kron` | flatten sharded dim (via `_unsafe_view`) |

**Fix:** Improve `aten.view.default` sharding strategy to handle more cases, or add
automatic redistribution before view ops that need it.

## Category 2: Factory / Creation Ops (No Tensor Inputs)

The test harness (`DTensorConverter`) can't convert args to DTensors because these
ops take shapes/scalars, not tensors. Not a DTensor runtime bug.

| Op |
|---|
| `arange`, `eye`, `full`, `ones`, `zeros`, `scalar_tensor` |
| `linspace`, `logspace` (scalar overloads) |
| `signal.windows.*` (bartlett, blackman, cosine, exponential, gaussian, general_cosine, general_hamming, hamming, hann, nuttall, kaiser) |

**Fix:** Test harness issue — these ops need special handling in the test (they don't
take tensor inputs, so the crossref DTensor test doesn't apply in the normal way).

## Category 3: Random/Stochastic Ops (RNG Mismatch)

DTensor path and reference path use different RNG states, so random outputs don't
match in the crossref comparison. Not a DTensor runtime bug.

| Op |
|---|
| `bernoulli`, `rand_like`, `randint_like`, `randint`, `randn`, `randn_like`, `uniform`, `cauchy` |
| `normal`, `normal` (number_mean), `normal` (in_place) |
| `nn.functional.dropout`, `nn.functional.dropout2d`, `nn.functional.dropout3d`, `nn.functional.alpha_dropout` |

**Fix:** Test harness issue — these need seeding or should skip the value comparison
and only check shapes/placements.

## Category 4: Missing Sharding Strategy (Miscategorized)

These are in `dtensor_fails` but the root cause is `NotImplementedError: Operator
aten.X does not have a sharding strategy registered`. They should arguably be in
`dtensor_fails_no_strategy`.

| Op | Missing strategy for |
|---|---|
| `roll` | `aten.roll.default` |
| `searchsorted` | `aten.searchsorted.Tensor` |
| `_chunk_cat` | `aten._chunk_cat.default` |
| `block_diag` | `aten.block_diag.default` |
| `masked_scatter` | `aten.masked_scatter.default` |
| `_unsafe_masked_index` | `aten._unsafe_masked_index.default` |
| `_unsafe_masked_index_put_accumulate` | `aten._unsafe_masked_index_put_accumulate.default` |
| `allclose` | `aten.allclose.default` |
| `nn.functional.interpolate` (area) | `aten._adaptive_avg_pool2d.default` |
| `nn.functional.interpolate` (nearest, nearest-exact) | `aten.upsample_nearest1d.default` |
| `nn.functional.pad` (reflect) | `aten.reflection_pad1d.default` |
| `nn.functional.pad` (replicate, replicate_negative) | `aten.replication_pad1d.default` |
| `nn.functional.unfold` | `aten.im2col.default` |
| `nn.functional.adaptive_avg_pool1d/3d` | `aten._adaptive_avg_pool2d.default` |
| `nanquantile` | `aten.unsqueeze_.default` |
| `masked.median` | `aten.nanmedian.dim` |
| `masked.cumprod` | `aten.cumprod.default` |
| `as_strided` (all variants) | `aten.as_strided_scatter.default` |

**Fix:** Register sharding strategies for these aten ops. Each new strategy
unblocks the corresponding high-level ops.

## Category 5: Mixed Tensor/DTensor Inputs

An op internally creates plain `torch.Tensor`s (e.g., index tensors, random samples)
that get mixed with DTensor inputs, triggering `"got mixed torch.Tensor and DTensor"`.

| Op | Detail |
|---|---|
| `__getitem__` | Index tensors are plain Tensor |
| `quantile` | `aten.masked_fill.Scalar` sees mixed inputs |
| `nn.functional.fractional_max_pool2d/3d` | Internal random samples tensor |

**Fix:** Either convert internal tensors to DTensor automatically, or handle
mixed-input cases in the dispatch path.

## Category 6: Dynamic Output Shape

`DynamicOutputShapeException` — the op's output shape depends on data values,
incompatible with fake tensor propagation.

| Op |
|---|
| `masked_select` |
| `nn.functional.ctc_loss` |
| `combinations` (uses `masked_select` internally) |

**Fix:** Fundamental limitation — these ops need special DTensor handling or must
require Replicate placement.

## Category 7: 0-dim / Scalar Tensor Edge Cases

Sharding strategies don't handle 0-dimensional tensors.

| Op | Error |
|---|---|
| `logsumexp` | "Expected reduce_dims to not be None" on 0-dim input |
| `masked.logsumexp` | same |
| `transpose` | "Expected dim1 < ndim, got 0 >= 0" on 0-dim input |

**Fix:** Add 0-dim guards in the respective sharding strategy functions (small,
targeted fixes).

## Category 8: Convolution Stride+Padding Constraint

The tensor-parallel convolution strategy explicitly rejects `stride != 1` when
padding is non-zero.

| Op |
|---|
| `nn.functional.conv1d`, `conv2d`, `conv3d` |
| `nn.functional.conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d` |

**Fix:** Extend the convolution sharding strategy to handle strided+padded cases,
or fall back to replicate for those configurations.

## Category 9: In-place Op Placement Change

An in-place op during decomposition would require changing placement (e.g.,
`_NormPartial` → `Replicate`), which violates aliasing semantics.

| Op | Detail |
|---|---|
| `nn.functional.cosine_similarity` | `aten.clamp_min_` would need placement change |

**Fix:** Use out-of-place decomposition path, or register a dedicated strategy.

## Category 10: Miscellaneous / One-off Issues

| Op | Error | Notes |
|---|---|---|
| `full_like` | Incorrect output values (50% mismatch) | Correctness bug — fill value not applied properly on sharded tensor |
| `nn.functional.huber_loss` | Wrong mean reduction on mixed placements | `mean` computed over local shard size, not global |
| `squeeze` (multiple) | `'>=' not supported between 'list' and 'int'` | Bug in squeeze.dims strategy with empty `()` dims arg |
| `tensor_split` | "tensor data is not allocated yet" | Meta tensor / lazy allocation issue |
| `unbind` | "Cannot unbind along sharded dim" | Needs redistribution or dedicated strategy |
| `unsafe_split` | "output_specs has 1 output, op has != 1" | Bug in the registered strategy's output count |
| `index_put` | "value tensor shape cannot be broadcast" | Shard(0) causes local tensor shape mismatch |
| `resize_` | "cannot resize variables that require grad" | Test harness passes requires_grad=True tensors |

## Summary: Which Fixes Target Which Ops

| Fix | Ops Unblocked | Effort |
|---|---|---|
| Improve view/reshape sharding | 10 ops | Medium-High |
| Register missing aten strategies | ~19 ops | Medium per-strategy, high total |
| Fix test harness for factory ops | ~21 ops | Low |
| Fix test harness for random ops | ~14 ops | Low |
| Fix 0-dim tensor guards | 3 ops | Low |
| Handle mixed Tensor/DTensor | 3 ops | Medium |
| Fix conv stride+padding | 6 ops | Medium |
| Fix individual correctness bugs | full_like, huber_loss, squeeze_multiple, unsafe_split, index_put | Low-Medium each |
