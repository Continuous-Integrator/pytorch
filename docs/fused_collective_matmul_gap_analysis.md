# Fused Collective + MatMul Gap Analysis for vLLM Models (DeepSeek & Qwen)

**Author**: Tianren Gao (tianren@meta.com), MSL Infra PyTorch Team
**Oncall**: vLLM-compile
**Date**: 2026-03-13
**Status**: Draft for discussion

---

## Executive Summary

This document identifies the gaps for enabling **fused All-Gather + MatMul (AG+MM)**, **All-Reduce + MatMul (AR+MM)**, and **MatMul + Reduce-Scatter (MM+RS)** on vLLM models (DeepSeek V2/V3, Qwen2/2.5/3, Qwen3-MoE). These fusions overlap communication with computation and are critical for TP-bound inference latency. We focus especially on **AR+MM** and **AG+MM** as the highest-impact patterns, and analyze how quantized dtypes (FP8, INT8, INT4) affect fusibility.

**Bottom line**: FP8 models (DeepSeek-R1 FP8, Qwen3-235B-FP8) can benefit from existing PyTorch `symm_mem` fusions today via the Sequence Parallelism + AsyncTP compile passes in vLLM, but there are **6 concrete gaps** blocking full coverage across models and quantization formats.

---

## 1. How DeepSeek & Qwen Use TP Today

### 1.1 Standard TP Pattern (All Models)

All models use the same TP primitives from vLLM's `linear.py`:

| Layer | Linear Type | Collective After MatMul | Location |
|-------|-------------|------------------------|----------|
| **Attention QKV** | `QKVParallelLinear` (column-parallel) | None — each rank computes its shard | `linear.py:557` |
| **Attention O** | `RowParallelLinear` | **all_reduce** | `linear.py:1388` |
| **MLP gate_up** | `MergedColumnParallelLinear` | None — each rank computes its shard | `linear.py:557` |
| **MLP down** | `RowParallelLinear` | **all_reduce** | `linear.py:1388` |

**Key files**:
- DeepSeek V2/V3: `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/deepseek_v2.py`
- Qwen2: `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/qwen2.py`
- Qwen3: `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/qwen3.py`
- Qwen2 MoE: `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/qwen2_moe.py`
- Qwen3 MoE: `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/qwen3_moe.py`

### 1.2 MoE-Specific TP Patterns (DeepSeek V2/V3, Qwen MoE)

DeepSeek and Qwen MoE layers add extra collectives around expert computation:

```
DeepSeek V2/V3 MoE (deepseek_v2.py:337-387):
  hidden_states  ─┬─►  sequence_parallel_chunk()     # manual shard
                   │         │
                   │    SharedFusedMoE(experts)        # routed + shared experts
                   │         │
                   ├─►  tensor_model_parallel_all_gather()  # if sequence_parallel
                   └─►  maybe_all_reduce()                   # if not sequence_parallel

Qwen3 MoE (qwen3_moe.py:163-210):
  hidden_states  ─►  sequence_parallel_chunk()
                        │
                     FusedMoE(experts)
                        │
                     tensor_model_parallel_all_gather()
```

### 1.3 The Sequence Parallelism Decomposition (How AR Becomes RS + AG)

vLLM's compile pass (`sequence_parallelism.py`) decomposes:
```
all_reduce(x) + RMSNorm(x)  →  reduce_scatter(x) + RMSNorm(x_shard) + all_gather(norm_out)
```

This transforms the unfusable `AR + norm` pattern into:
```
[MM + RS]  →  [RMSNorm]  →  [AG + MM]
  fusable!                     fusable!
```

The AsyncTP pass (`collective_fusion.py`) then matches the MM+RS and AG+MM patterns and replaces them with `symm_mem` fused ops.

**Pass execution order** (from `pass_manager.py:115-125`):
1. `SequenceParallelismPass` — decomposes `all_reduce → reduce_scatter + all_gather`
2. `AsyncTPPass` — fuses `MM + reduce_scatter` and `all_gather + MM` via `symm_mem`
3. `AllReduceFusionPass` — fuses remaining `all_reduce + RMSNorm`

---

## 2. What PyTorch Supports Today

### 2.1 Fused Ops in `symm_mem` (torch/distributed/_symmetric_memory/__init__.py)

| Fused Op | Regular MatMul | FP8 `_scaled_mm` | Implementation |
|----------|---------------|-------------------|----------------|
| **AG + MM** | ✅ `fused_all_gather_matmul` | ✅ `fused_all_gather_scaled_matmul` | Lines 545-684, 1022-1180 |
| **MM + RS** | ✅ `fused_matmul_reduce_scatter` | ✅ `fused_scaled_matmul_reduce_scatter` | Lines 1254-1575 |
| **AR + MM** | ❌ Not implemented | ❌ Not implemented | Only `one_shot_all_reduce` exists |

### 2.2 Inductor FX Pass for Fusion (torch/_inductor/fx_passes/micro_pipeline_tp.py)

PyTorch's **Inductor micro-pipeline TP pass** detects and fuses these patterns in the FX graph:
- `_Matmul` (regular `aten.mm.default`) — lines 376-450
- `_ScaledMatmul` (FP8 `aten._scaled_mm.default`) — lines 453-506

Gated by `config._micro_pipeline_tp = False` (disabled by default).

### 2.3 vLLM's Own AsyncTP Pass (collective_fusion.py)

vLLM has its **own pattern-matching pass** that matches 6 pattern variants:

| Pattern | Matches | Replaces With |
|---------|---------|---------------|
| `GEMMReduceScatterPattern` | `aten.mm` + `vllm.reduce_scatter` | `symm_mem.fused_matmul_reduce_scatter` |
| `AllGatherGEMMPattern` | `vllm.all_gather` + `aten.mm` | `symm_mem.fused_all_gather_matmul` |
| `ScaledMMReduceScatterPattern` | `aten._scaled_mm` + `vllm.reduce_scatter` | `vllm.patched_fused_scaled_matmul_reduce_scatter` |
| `AllGatherScaledMMPattern` | `vllm.all_gather` + `aten._scaled_mm` | `symm_mem.fused_all_gather_scaled_matmul` |
| `CutlassScaledMMReduceScatterPattern` | `cutlass_scaled_mm` + `vllm.reduce_scatter` | `vllm.patched_fused_scaled_matmul_reduce_scatter` |
| `AllGatherCutlassScaledMMPattern` | `vllm.all_gather` + `cutlass_scaled_mm` | `symm_mem.fused_all_gather_scaled_matmul` |

**Restriction**: FP8 patterns only enabled when `model_dtype == torch.bfloat16` (line 391 of `collective_fusion.py`).

---

## 3. Model × Quantization Coverage Matrix

Which model + quantization combinations can fuse today:

| Model | Quant | Matmul Op | AG+MM Fusable? | MM+RS Fusable? | AR+MM? | Notes |
|-------|-------|-----------|---------------|---------------|--------|-------|
| **Qwen3-30B-A3B BF16** | None (BF16) | `aten.mm` | ✅ Yes | ✅ Yes | ❌ Decompose first | Dense model, cleanest pattern |
| **Qwen3-235B-A22B FP8** | FP8 | `aten._scaled_mm` | ✅ Yes | ✅ Yes | ❌ Decompose first | Row-wise scales work |
| **DeepSeek-R1 FP8** | FP8 (block `[128,128]`) | `aten._scaled_mm` | ⚠️ **Gap 3** | ⚠️ **Gap 3** | ❌ Decompose first | Block-wise scales may not match |
| **Qwen2.5-72B GPTQ-INT4** | GPTQ INT4 | Marlin kernel | ❌ **Gap 1** | ❌ **Gap 1** | ❌ | Custom op, opaque to Inductor |
| **DeepSeek-V3 AWQ-INT4** | AWQ INT4 | Marlin kernel | ❌ **Gap 1** | ❌ **Gap 1** | ❌ | Custom op, opaque to Inductor |
| **Qwen2.5-72B W8A8** | INT8 | Marlin/CUTLASS INT8 | ❌ **Gap 1** | ❌ **Gap 1** | ❌ | Custom op, opaque to Inductor |
| **DeepSeek MoE layers** | Any | FusedMoE (Triton) | ⚠️ **Gap 4** | ⚠️ **Gap 4** | ❌ | MoE is opaque Triton custom op |
| **Qwen3-MoE layers** | Any | FusedMoE (Triton) | ⚠️ **Gap 4** | ⚠️ **Gap 4** | ❌ | Same MoE issue |

---

## 4. Gap Analysis — Detailed Breakdown

### Gap 1: INT8 / INT4 / GPTQ / AWQ — No Collective Fusion Support

**Impact**: 🔴 **High** — GPTQ-INT4 and AWQ-INT4 are the most popular quantization formats for cost-efficient deployment. W8A8 INT8 is used for accuracy-sensitive workloads.

**Affected Models**:
- Qwen2.5-72B-Instruct-GPTQ-Int4
- Qwen2.5-72B-Instruct-AWQ
- DeepSeek-V3 AWQ variants
- Any W8A8-INT8 checkpoint of these models

**Root Cause**:
- Marlin/GPTQ/AWQ kernels are **custom ops** (`_C.cutlass_scaled_mm`, `apply_gptq_marlin_linear`, `apply_awq_marlin_linear`) that are opaque to Inductor's fusion framework.
- PyTorch `symm_mem` fused ops only support `aten.mm` (BF16/FP16) and `aten._scaled_mm` (FP8). No `fused_all_gather_marlin_mm` or equivalent exists.
- vLLM-compile notes explicitly flag: *"Quantization ops don't fuse with Inductor patterns today because they're custom ops."*

**Where It Lives**: The quant kernels are in vLLM:
- FP8: `vllm/model_executor/layers/quantization/fp8.py` (uses `aten._scaled_mm` — ✅ fusable)
- GPTQ: `vllm/model_executor/layers/quantization/gptq_marlin.py` (uses Marlin — ❌ not fusable)
- AWQ: `vllm/model_executor/layers/quantization/awq_marlin.py` (uses Marlin — ❌ not fusable)
- W8A8: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py` (uses CUTLASS INT8 — ❌ not fusable)
- NVFP4: `vllm/model_executor/layers/quantization/modelopt.py` (uses FP4 custom kernel — ❌ not fusable)

**Performance Impact**:
- Without fusion, TP communication cannot overlap with computation for these quant formats.
- For decode (small batch), TP communication is a significant fraction of per-layer latency. Estimate: **10-20% latency reduction** possible if fused.
- For prefill (large batch), compute dominates and fusion benefit is smaller (~5%).

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Custom fused kernels | **PyTorch** `symm_mem` + **vLLM** pattern | 🔴 High (4-6 weeks) | Implement `fused_all_gather_marlin_mm` and `fused_marlin_mm_reduce_scatter` that combine Marlin dequant+matmul with symmetric memory collectives. Requires CUDA kernel development. |
| B. Make quant ops decomposable | **PyTorch** Inductor + **torchao** | 🔴 Very High (8+ weeks) | Make Marlin/INT8 kernels traceable via `torch.library` so Inductor can decompose them into `dequant → aten.mm → ...`. Once decomposed to `aten.mm`, existing fusion patterns work. Depends on torchao roadmap. |
| C. Layer-level pipelining | **vLLM** scheduler | 🟡 Medium (2-3 weeks) | Don't fuse within a layer. Instead, overlap layer N's collective with layer N+1's quant+matmul compute. Requires exposing async collectives in vLLM's execution graph. Less optimal than true fusion but works for any quant format. |

**Recommendation**: Option C (layer-level pipelining) as short-term pragmatic fix, Option B as the long-term strategic fix aligned with torchao direction.

---

### Gap 2: AR+MM Fusion Does Not Exist in PyTorch

**Impact**: 🟡 **Medium-High** — `all_reduce` after `RowParallelLinear` is the **most common TP collective** in every transformer model. It fires twice per layer (after `o_proj` and after `down_proj`).

**Affected Models**: ALL models (DeepSeek V2/V3, Qwen2/3/MoE, every model using RowParallelLinear with TP > 1).

**Root Cause**:
- No `fused_allreduce_matmul` op exists in PyTorch `symm_mem`. The `all_reduce` is performed as a standalone op (`one_shot_all_reduce` or NCCL `allReduce`).
- The design rationale is that all-reduce is implemented as reduce-scatter + all-gather internally, so the "right" decomposition for fusion is to expose this and fuse at the RS/AG level instead.

**Current Workaround (Already Implemented)**:
vLLM's `SequenceParallelismPass` decomposes `all_reduce → reduce_scatter + all_gather` so the patterns become fusable:
```
Before:  [o_proj MM] → [all_reduce] → [RMSNorm] → [gate_up MM]
After:   [o_proj MM + RS] → [RMSNorm on shard] → [AG + gate_up MM]
              ↑ fused                                    ↑ fused
```

**The Gap in the Workaround**:
The decomposition introduces **extra communication volume** and **overhead** that may hurt small-batch decode:

| Metric | all_reduce | reduce_scatter + all_gather |
|--------|-----------|----------------------------|
| Total data moved | 2 × (N-1)/N × size | 2 × (N-1)/N × size (same total) |
| Number of kernel launches | 1 | 2 (or 3 with fusion ops) |
| Fusion overhead | N/A | Decomposition + graph manipulation |
| Profitable when... | Batch small, compute < comm | Batch large enough to overlap |

For **decode with batch_size=1-4**, the TP comm is already very small and the overhead of decomposition + fusion setup may exceed the overlap benefit. For **prefill** or larger decode batches, fusion wins.

**Where It Lives**:
- Decomposition pass: `vllm/compilation/passes/fusion/sequence_parallelism.py`
- Fusion pass: `vllm/compilation/passes/fusion/collective_fusion.py`
- PyTorch `one_shot_all_reduce`: `torch/distributed/_symmetric_memory/__init__.py:2128+`

**Performance Impact**:
- **Prefill (batch_size ≥ 32)**: Expect **15-25% TP latency improvement** from overlapping RS/AG with matmul.
- **Decode (batch_size 1-8)**: Benefit depends on heuristic — may be **0-10%** or even negative if decomposition overhead dominates.
- This is exactly what the AutoHeuristic work (D176943, D176944, D177068) targets — choosing when to fuse vs. not.

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Keep decomposition + improve heuristics | **PyTorch** Inductor + **vLLM** | 🟢 Low (2-3 weeks) | Improve the AutoHeuristic to make better fuse/no-fuse decisions for small decode. Already in progress (D176943). |
| B. Native AR+MM in symm_mem | **PyTorch** `symm_mem` | 🟡 Medium (3-4 weeks) | Implement `fused_allreduce_matmul` that overlaps the reduce phase with the subsequent matmul. Avoids decomposition overhead. Would need new CUDA kernel. |
| C. Adaptive decomposition | **vLLM** compile passes | 🟢 Low (1-2 weeks) | Only decompose `all_reduce → RS + AG` for compile ranges where batch_size exceeds a threshold. Keep plain `all_reduce` for small decode. |

**Recommendation**: Option A (AutoHeuristic, already in progress) + Option C (adaptive decomposition threshold) for immediate wins. Option B as a future exploration if heuristic approach shows persistent gaps.

---

### Gap 3: Block-Wise FP8 Scales Not Handled in Fused Ops

**Impact**: 🔴 **High** — DeepSeek-R1 FP8 is one of the most important production models and uses block-wise quantization.

**Affected Models**:
- **DeepSeek-R1 FP8** (block scales: `[1,128]` for activations, `[128,128]` for weights)
- Any future model using block-wise FP8 (likely an increasing trend)

**Root Cause**:
PyTorch `symm_mem`'s `_check_and_verify_fp8_all_gather_scale_mode` (line 512-542) only supports three scale modes:

```python
class _ScaleMode(Enum):
    UNSCALED = "unscaled"
    TENSOR_WISE = "tensor-wise"        # 1 scale per tensor — works ✅
    ROW_WISE_SHARDED = "row-wise-sharded"    # 1 scale per row, sharded — works ✅
    ROW_WISE_REPLICATED = "row-wise-replicated"  # 1 scale per row, full — works ✅
```

Block-wise scales (`[128,128]`) don't fit any of these. When the scale tensor has shape `[M/128, K/128]`, it's not tensor-wise (not scalar) and not row-wise (not `[M, 1]`). The check will fail or produce incorrect results.

**Where It Lives**:
- Scale mode check: `torch/distributed/_symmetric_memory/__init__.py:512-542`
- FP8 all-gather implementation: `torch/distributed/_symmetric_memory/__init__.py:611-684`
- DeepSeek FP8 config: block-wise scales set by checkpoint `quantization_config`

**Performance Impact**:
- If block-wise scales aren't handled, DeepSeek-R1 FP8 **cannot use AG+MM or MM+RS fusions at all**.
- Falls back to unfused `all_gather → scaled_mm` and `scaled_mm → reduce_scatter`.
- Estimate: **15-25% TP latency improvement lost** for this model.

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Add BLOCK_WISE scale mode | **PyTorch** `symm_mem` | 🟡 Medium (2-3 weeks) | Add `_ScaleMode.BLOCK_WISE` to `_check_and_verify_fp8_all_gather_scale_mode`. For AG: all-gather the block scales alongside the FP8 data (or replicate if small enough). For RS: slice block scales per shard. Math is straightforward, but needs careful alignment handling (cuBLAS 16-byte alignment at line 1512). |
| B. Convert block → row-wise before fusion | **vLLM** quant layer | 🟢 Low (1 week) | In vLLM's FP8 linear layer, convert block-wise scales to row-wise (take max across the K-block dimension) before the collective. Slight accuracy loss but preserves fusibility. |

**Recommendation**: Option A is the correct fix. Option B as a temporary workaround if needed faster.

---

### Gap 4: MoE Layers — FusedMoE is Opaque to Collective Fusion

**Impact**: 🟡 **Medium** — MoE layers dominate compute in DeepSeek V2/V3 and Qwen3-MoE. The MoE itself isn't fused with collectives.

**Affected Models**:
- DeepSeek V2/V3 (MoE with shared experts, `SharedFusedMoE`)
- Qwen2-MoE, Qwen3-MoE (`FusedMoE`)

**Root Cause**:
The `FusedMoE` kernel is a Triton custom op that combines gating, expert routing, and expert matmul. It's opaque to Inductor. The collectives around it are:

```
Current DeepSeek MoE flow:
  [attn o_proj MM → all_reduce]         # can't fuse (all_reduce blocks)
       ↓
  sequence_parallel_chunk(hidden)       # manual shard
       ↓
  SharedFusedMoE(hidden, router)        # opaque Triton kernel
       ↓
  tensor_model_parallel_all_gather()    # can't fuse (MoE output, not matmul)
```

There's a TODO in `deepseek_v2.py:343-344`:
```python
# TODO: We can replace the all_reduce at the end of attn with a
# reduce_scatter instead of chunking here.
```

**Where It Lives**:
- MoE forward: `vllm/model_executor/models/deepseek_v2.py:337-387`
- FusedMoE impl: `vllm/model_executor/layers/fused_moe/fused_moe.py`
- SharedFusedMoE: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`

**Performance Impact**:
- The `all_gather` after MoE cannot be fused with the next layer's QKV matmul because the MoE output isn't from an `aten.mm` node.
- The `sequence_parallel_chunk` is essentially a manual reduce-scatter that could be fused with the preceding `o_proj` matmul.
- Estimate: **5-10% per-layer improvement** if the attention→MoE boundary is optimized.

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Implement the TODO: attn all_reduce → reduce_scatter | **vLLM** model code | 🟢 Low (1-2 weeks) | Replace `all_reduce` after `o_proj` with `reduce_scatter`, eliminating `sequence_parallel_chunk`. The MoE layer receives already-scattered input. Enables `MM + RS` fusion on `o_proj`. |
| B. Fuse MoE all_gather with next QKV | **vLLM** compile pass | 🟡 Medium (2-3 weeks) | Add pattern matching for `all_gather(FusedMoE_output) → QKV_matmul` even though the all_gather source isn't from an `aten.mm`. Requires extending the pattern matcher to handle non-matmul all-gather sources. |

**Recommendation**: Option A — implement the existing TODO. It's low effort, well-understood, and directly enables MM+RS fusion on `o_proj`.

---

### Gap 5: FP8 Last-Dim Gather/Scatter Explicitly Filtered Out

**Impact**: 🟡 **Medium** — Affects column-parallel patterns where gather_dim is along the feature dimension.

**Affected Models**: Any FP8 model using `ColumnParallelLinear` with `gather_output=True` (less common in practice — most models don't gather output after column-parallel layers).

**Root Cause**:
In `micro_pipeline_tp.py:658-662` and `908-912`:
```python
if _is_last_dim(_get_tensor(shard_node), gather_dim):
    # scaled_mm is not supported yet for last dim
    def _filter_out_scaled_matmul(matmul: _Matmul):
        return not isinstance(matmul, _ScaledMatmul)
    filter_matmul = _filter_out_scaled_matmul
```

This means if TP shards along the last dimension (K-dim of weight), FP8 `_scaled_mm` is excluded from fusion. The issue is that gathering along the non-row dimension interacts with FP8 row-wise scale handling.

**Where It Lives**: `torch/_inductor/fx_passes/micro_pipeline_tp.py:658-662, 908-912`

**Performance Impact**: Limited in practice, since most TP patterns in vLLM use `gather_dim=0` (sequence dimension). However, some embedding or head layers may use last-dim gather.

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Implement last-dim scale handling | **PyTorch** `symm_mem` | 🟡 Medium (2-3 weeks) | Handle scale transposition and alignment when gathering along the last dimension. Need to properly slice/gather row-wise scales when the "row" changes meaning after gathering. |

**Recommendation**: Lower priority — address after Gaps 1-4. Most production models don't hit this path.

---

### Gap 6: Multi-User MatMul Results Block MM+RS Fusion

**Impact**: 🟡 **Medium** — Affects patterns where the matmul output feeds both the reduce-scatter and another consumer.

**Affected Models**: DeepSeek shared expert patterns, residual connections that branch before the collective.

**Root Cause**:
In `micro_pipeline_tp.py:914-922`:
```python
# Currently fused_matmul_reduce_scatter doesn't return the matmul result,
# so we can't apply the fusion if the matmul result is used by multiple users.
if len(input_node.users) != 1:
    log.warning("matmul result has more than one user, skipping ...")
    return
```

The code itself says *"This is not a fundamental limitation of the fused op and can be addressed if needed."*

**Where It Lives**: `torch/_inductor/fx_passes/micro_pipeline_tp.py:914-922`

**Performance Impact**: Blocks MM+RS fusion when the same matmul output is used for both reduction and a local computation. In DeepSeek, the shared expert's output is `all_reduced` separately and then added to the routed expert output — this branching may trigger the multi-user restriction.

**Fix Options**:

| Option | Where to Fix | Effort | Description |
|--------|-------------|--------|-------------|
| A. Return full matmul result from fused op | **PyTorch** `symm_mem` | 🟢 Low (1 week) | Modify `fused_matmul_reduce_scatter` and `fused_scaled_matmul_reduce_scatter` to return both the reduced shard and the full pre-reduction matmul result. Update the FX pass to use both outputs. |

**Recommendation**: Easy fix, do it. The code even acknowledges it should be done.

---

## 5. Quantitative Performance Analysis

### 5.1 Model Architecture Parameters

| Parameter | DeepSeek-V3/R1 (671B) | Qwen2.5-72B | Qwen3-32B | Qwen3-235B-A22B | Qwen3-30B-A3B |
|-----------|----------------------|-------------|-----------|-----------------|---------------|
| hidden_size (H) | 7168 | 8192 | 5120 | 4096 | 2048 |
| intermediate_size (FFN) | 18432 (dense) / 2048 (MoE) | 29568 | 27648 | MoE-based (~1408) | MoE-based (768) |
| num_attention_heads | 128 | 64 | 64 | 64 | 32 |
| num_kv_heads | 128 | 8 | 8 | 8 | 4 |
| num_layers (L) | 61 | 80 | 64 | 94 | 48 |
| num_experts | 256 | — (dense) | — (dense) | 128 | 128 |
| experts_per_tok | 8 | — | — | 8 | 8 |
| n_shared_experts | 1 | — | — | 0 | 0 |

### 5.2 Hardware Constants (H100 SXM, 8-GPU DGX)

| Parameter | Value |
|-----------|-------|
| BF16 GEMM peak | 989 TFLOPS |
| FP8 GEMM peak | 1,979 TFLOPS |
| HBM bandwidth | 3.35 TB/s |
| NVLink bandwidth (per GPU, per direction) | 450 GB/s |
| NVLink bandwidth (bidirectional) | 900 GB/s |
| All-reduce algo BW (ring, TP=8) | ~400 GB/s effective |
| Reduce-scatter algo BW (TP=8) | ~394 GB/s (7/8 × 450) |
| All-gather algo BW (TP=8) | ~394 GB/s (7/8 × 450) |
| CUDA kernel launch overhead | ~5 µs (with CUDA graphs) |
| one_shot_all_reduce latency (small msg) | ~5 µs |
| DDA all-reduce latency (small msg) | ~9 µs (6 µs launch + 3 µs actual) |

### 5.3 Per-Layer Communication Cost

Each standard transformer layer has **2 all-reduces** (after `o_proj` and after `down_proj`). With SP decomposition, each becomes **1 reduce-scatter + 1 all-gather** = 4 collectives per layer.

**Communication data size per collective** = `batch_tokens × hidden_size × bytes_per_element`

Where `batch_tokens = batch_size × seq_len` for prefill, `batch_size × 1` for decode.

| Scenario | batch_tokens | DeepSeek-V3 (H=7168, BF16) | Qwen2.5-72B (H=8192, BF16) | Qwen3-32B (H=5120, BF16) |
|----------|-------------|---------------------------|---------------------------|-------------------------|
| **Decode bs=1** | 1 | 14.3 KB | 16.4 KB | 10.2 KB |
| **Decode bs=8** | 8 | 114.7 KB | 131.1 KB | 81.9 KB |
| **Decode bs=32** | 32 | 458.8 KB | 524.3 KB | 327.7 KB |
| **Prefill seq=2048** | 2048 | 28.0 MB | 32.0 MB | 20.0 MB |
| **Prefill seq=8192** | 8192 | 112.0 MB | 128.0 MB | 80.0 MB |

**Communication latency per collective** (reduce-scatter or all-gather, `msg_size × 7/8 / 450 GB/s + 5µs launch`):

| Scenario | batch_tokens | RS or AG time | AR time (ring) | Collectives per layer | Total comm/layer |
|----------|-------------|---------------|----------------|----------------------|-----------------|
| **Decode bs=1** | 1 | **~5 µs** (launch-dominated) | **~5-9 µs** | 4 (SP) or 2 (AR) | **20 µs** (SP) / **10-18 µs** (AR) |
| **Decode bs=8** | 8 | **~5.2 µs** | **~5.5 µs** | 4 / 2 | **21 µs** / **11 µs** |
| **Decode bs=32** | 32 | **~5.9 µs** | **~7.0 µs** | 4 / 2 | **24 µs** / **14 µs** |
| **Prefill seq=2048** | 2048 | **~60 µs** | **~127 µs** | 4 / 2 | **240 µs** / **254 µs** |
| **Prefill seq=8192** | 8192 | **~223 µs** | **~497 µs** | 4 / 2 | **892 µs** / **994 µs** |

**Key insight**: At decode bs=1-8, communication is **kernel-launch dominated** (~5 µs per collective regardless of data size). At prefill, communication is **bandwidth-bound** and scales linearly with batch_tokens.

### 5.4 Per-Layer Computation Cost

For a dense layer, the two largest matmuls are:
- **gate_up_proj**: `[batch_tokens, H] × [H, 2×FFN/TP]` → FLOPs = `2 × batch_tokens × H × 2 × FFN / TP`
- **down_proj**: `[batch_tokens, FFN/TP] × [FFN/TP, H]` → FLOPs = `2 × batch_tokens × FFN/TP × H`

**Computation time per matmul** (BF16 on H100, 989 TFLOPS peak, ~70% utilization for large M):

| Scenario | batch_tokens | gate_up MM (Qwen2.5-72B) | down_proj MM (Qwen2.5-72B) | gate_up MM (Qwen3-32B) | down_proj MM (Qwen3-32B) |
|----------|-------------|--------------------------|---------------------------|------------------------|-------------------------|
| **Decode bs=1** | 1 | ~0.1 µs (mem-bound, ~14 µs actual) | ~0.05 µs (~8.5 µs actual) | ~0.06 µs (~8.5 µs actual) | ~0.03 µs (~5.3 µs actual) |
| **Decode bs=8** | 8 | ~14 µs (still mem-bound) | ~8.5 µs | ~8.5 µs | ~5.3 µs |
| **Decode bs=32** | 32 | ~16 µs | ~10 µs | ~10 µs | ~6 µs |
| **Prefill seq=2048** | 2048 | **~198 µs** | **~99 µs** | **~82 µs** | **~41 µs** |
| **Prefill seq=8192** | 8192 | **~790 µs** | **~395 µs** | **~328 µs** | **~164 µs** |

> **Note**: At decode bs=1-8, matmuls are **memory-bandwidth bound** (GEMV regime). Actual times are dominated by weight loading from HBM, not compute. For Qwen2.5-72B with H=8192, FFN=29568, TP=8: weight size for gate_up = 8192 × 29568 × 2 / 8 = ~72 MB, load time = 72 MB / 3.35 TB/s ≈ 21.5 µs. This is the effective matmul time.

### 5.5 Fusion Overlap Analysis

**Fusion benefit** = communication time that can be hidden behind computation.

The overlap potential depends on `min(comm_time, compute_time)`:
- If `compute_time >> comm_time`: communication fully hidden → **gain ≈ comm_time saved**
- If `compute_time << comm_time`: compute can overlap with only a fraction → **gain ≈ compute_time**
- If `compute_time ≈ comm_time`: partial overlap → gain depends on pipelining efficiency

#### Gap 2: AR+MM → SP Decomposition (RS + AG) + Fusion

**What it does**: Decomposes each `all_reduce` into `reduce_scatter + all_gather`, then fuses:
- `down_proj MM + RS` (overlaps RS with the tail of the matmul)
- `AG + gate_up MM` (overlaps AG with the head of the next matmul)

**Savings = communication time hidden per layer:**

| Scenario | Comm hidden (ideal) | Compute available to overlap | Actual savings | % of layer time saved |
|----------|--------------------|-----------------------------|---------------|----------------------|
| **Decode bs=1, Qwen2.5-72B** | 2 × (5+5) = 20 µs (4 collectives) | ~14+8.5 = 22.5 µs (very tight, mem-bound MMs) | **~0-5 µs** (overhead may eat gains) | **0-3%** |
| **Decode bs=8, Qwen2.5-72B** | 2 × (5.2+5.2) = 20.8 µs | ~14+8.5 = 22.5 µs | **~8-12 µs** | **5-8%** |
| **Decode bs=32, Qwen2.5-72B** | 2 × (5.9+5.9) = 23.6 µs | ~16+10 = 26 µs | **~15-20 µs** | **8-12%** |
| **Prefill seq=2048, Qwen2.5-72B** | 2 × (60+60) = 240 µs | 198+99 = 297 µs | **~180-220 µs** | **15-22%** |
| **Prefill seq=8192, Qwen2.5-72B** | 2 × (223+223) = 892 µs | 790+395 = 1185 µs | **~700-850 µs** | **18-25%** |

**⚠️ Critical: Decode bs=1 regression risk!** The SP decomposition replaces 2 collectives with 4 collectives per layer, adding 2 extra kernel launches (~10 µs). At bs=1, compute is too small to hide anything, so the **net effect can be negative** (-5 to -10 µs per layer). This is exactly what Gap 2 (AutoHeuristic) targets.

**Validated by benchmark data**: The CustomOp Autotuning doc [a] on Llama3-70B (similar dimensions) confirms:
- M_shard=32 (decode-like): Baseline **216.9 µs** vs Full Fused **575.6 µs** → fused is **2.65× slower**
- M_shard=128: Fused AG+MM **495.4 µs** wins over Baseline **534.6 µs** → **1.08× speedup**
- M_shard=1024+: Full Fused wins → **1.08-1.13× speedup**

#### Gap 3: Block-Wise FP8 Scales (DeepSeek-R1)

**What it unlocks**: FP8 matmul runs at ~2× FLOPS of BF16 (1979 vs 989 TFLOPS), so matmuls are ~2× faster. But the communication data is also ~2× smaller (FP8 = 1 byte vs BF16 = 2 bytes). The **ratio stays similar, but absolute times shrink**, making fusion even more critical to avoid becoming comm-bound.

| Scenario | BF16 MM time | FP8 MM time | BF16 comm time | FP8 comm time | Comm as % of layer |
|----------|-------------|-------------|----------------|---------------|-------------------|
| Prefill 2048, Qwen2.5-72B | ~297 µs | ~149 µs | ~240 µs | ~125 µs | BF16: 45%, **FP8: 46%** |
| Prefill 8192, Qwen2.5-72B | ~1185 µs | ~593 µs | ~892 µs | ~451 µs | BF16: 43%, **FP8: 43%** |

**Without fusion for FP8** (Gap 3 for DeepSeek-R1): 46% of layer time is wasted on communication.
**With fusion for FP8**: Can hide ~80-90% of comm → save ~100-400 µs per layer.

**End-to-end impact for DeepSeek-R1 FP8** (61 layers, prefill seq=2048):
- Comm time exposed without fusion: 61 × ~125 µs = **~7.6 ms**
- Comm time hidden with fusion: 61 × ~15 µs = **~0.9 ms**
- **Savings: ~6.7 ms** → on TTFT of 8.4s, this is ~0.08% for prefill. But for **decode** (53 ms TPOT), saving ~0.6 ms per token is **~1.1%** reduction.

**However**, the real gain compounds: DeepSeek-V3 at TP=16 has 15/16 × data per collective, and collectives are latency-bound. At TP=16 decode bs=4, each token budget is ~53 ms / 61 layers ≈ 0.87 ms per layer. Exposed comm of ~20 µs × 4 collectives = 80 µs per layer = **9.2% of per-layer budget**. Hiding this is meaningful.

#### Gap 1: INT8/INT4 Quant — No Fusion at All

**Impact depends on how much comm cost exists**. Same data sizes as BF16 (communication is on activations, which are BF16 even with quantized weights). But quantized matmuls (Marlin) can be **faster** than BF16 matmuls for small batch sizes (weight decompression is pipelined), making the **comm/compute ratio worse**.

| Scenario | Marlin INT4 MM time (est.) | BF16 comm time | Comm as % of layer |
|----------|--------------------------|----------------|-------------------|
| Decode bs=1, Qwen2.5-72B | ~8-10 µs (faster weight decomp) | ~10-18 µs (AR) | **55-65%** |
| Decode bs=8, Qwen2.5-72B | ~10-12 µs | ~11 µs (AR) | **45-50%** |
| Prefill 2048, Qwen2.5-72B | ~100-150 µs | ~254 µs (AR) | **60-70%** |

**Comm fraction is even higher with INT4!** Because the matmul is faster (less data to load from HBM for 4-bit weights), communication becomes a larger bottleneck. Without any fusion ability, INT4 models leave **45-70% of per-layer time** as pure communication waste.

**Potential savings (layer-level pipelining, Gap 1 Option C)**:
Can overlap ~50-70% of communication with next layer's compute → save ~25-45% of comm time.
- Decode bs=8: save ~5-6 µs per layer × 80 layers = **~400-480 µs per token**
- With TPOT of ~30 ms (72B at TP=8 INT4), that's **~1.5% end-to-end**

#### Gap 4: MoE Attn→MoE Boundary (DeepSeek)

**What it fixes**: The `o_proj → all_reduce → sequence_parallel_chunk → MoE → all_gather` pipeline has an unnecessary all_reduce that could be a reduce_scatter fused with `o_proj`.

For DeepSeek-V3 MoE layers (58 out of 61 layers are MoE):
- Current: `o_proj MM` (compute) → `all_reduce` (comm, ~5-9 µs at decode) → `chunk` (free)
- Fixed: `o_proj MM + RS` (comm hidden behind compute) → direct to MoE input

**Savings per MoE layer**: One fewer collective exposed = **~5-9 µs at decode**, **~60-223 µs at prefill**
**End-to-end**: 58 layers × 5-9 µs = **~290-522 µs per decode token** → on 53 ms TPOT, that's **~0.5-1.0%**

At prefill: 58 × 60 µs = **~3.5 ms** → on 8.4s TTFT, that's **~0.04%** (tiny for prefill)

The real win is that this enables MM+RS fusion on `o_proj` which was previously impossible because MoE used manual chunking.

#### Gap 6: Multi-User MM+RS

**Affects**: DeepSeek shared expert pattern where `down_proj` output feeds both reduce-scatter and the shared expert addition. Blocks fusion on ~58 MoE layers.

**Savings**: Same as MM+RS fusion — one more collective hidden per affected layer.
- **~5-9 µs per layer at decode** × 58 layers = **~290-522 µs per token**

### 5.6 End-to-End Performance Gain Summary

#### Decode Latency Savings (per output token)

| Gap | Fix | DeepSeek-V3 TP=16 (53 ms TPOT) | Qwen2.5-72B TP=8 (~30 ms TPOT) | Qwen3-32B TP=4 (~20 ms TPOT) |
|-----|-----|-------------------------------|-------------------------------|------------------------------|
| **Gap 2** (AR→SP+fusion heuristic) | AutoHeuristic (bs-adaptive) | bs=1: **avoid regression** (0→-5%); bs≥4: **+3-8%** | bs=1: **avoid regression**; bs≥8: **+5-10%** | bs=1: **avoid regression**; bs≥8: **+5-8%** |
| **Gap 3** (block FP8 scales) | Add BLOCK_WISE scale mode | FP8: **+5-9%** (unlock fusion entirely) | N/A (row-wise scales work) | N/A |
| **Gap 4** (MoE attn→RS) | Implement TODO in deepseek_v2.py | **+0.5-1.0%** | N/A (no MoE) | N/A |
| **Gap 6** (multi-user MM+RS) | Return full result from fused op | **+0.5-1.0%** (DeepSeek shared experts) | **<0.5%** | **<0.5%** |
| **Gap 1** (INT4/INT8 fusion) | Layer-level pipeline | N/A (use FP8) | INT4: **+1-2%**; INT8: **+1-2%** | INT4: **+1-2%** |
| **Gap 5** (last-dim FP8) | Last-dim scale handling | **<0.5%** | **<0.5%** | **<0.5%** |
| | **Combined (all gaps)** | **+6-11%** (FP8, bs≥4) | **+5-12%** (BF16/FP8, bs≥8) | **+5-10%** (bs≥8) |

#### Prefill Latency Savings (TTFT)

| Gap | Fix | DeepSeek-V3 TP=16 (8.4s TTFT) | Qwen2.5-72B TP=8 | Qwen3-235B TP=4 (16.9s) |
|-----|-----|-------------------------------|-------------------|------------------------|
| **Gap 2** (SP+fusion) | Decompose + fuse always | **+15-22%** (always profitable at prefill) | **+15-25%** | **+12-20%** |
| **Gap 3** (block FP8) | BLOCK_WISE scale mode | FP8: **+12-20%** (unlock fusion) | N/A | FP8: **+10-18%** |
| **Gap 1** (INT4/INT8) | Layer-level pipeline | N/A | INT4: **+8-15%** | N/A |
| | **Combined** | **+15-22%** | **+15-25%** (BF16/FP8) | **+12-20%** |

#### Throughput Improvement (tokens/sec at saturation)

| Gap | Fix | Impact on throughput |
|-----|-----|---------------------|
| **Gap 2** (SP+fusion) | Auto-adaptive | **+10-20%** at high batch (compute-comm overlap frees GPU cycles) |
| **Gap 3** (block FP8) | BLOCK_WISE | **+10-18%** for DeepSeek-R1 FP8 (unlocks fusion that was completely blocked) |
| **Gap 1** (INT4/INT8) | Layer pipeline | **+5-10%** for quantized models |

### 5.7 Crossover Analysis: When Fusion Helps vs Hurts

**This is the most important insight for AR+MM and AG+MM**.

From the internal CustomOp Autotuning benchmarks (8×H100, TP=8, Llama3-70B FFN):

| M_shard (per GPU) | M_total | Winner | Speedup vs Baseline |
|-------------------|---------|--------|---------------------|
| 32 | 256 | ❌ **Baseline** (no fusion) | Fused is **2.65× slower** |
| 64 | 512 | ❌ **Baseline** | Fused AG+MM is **1.03× slower** |
| 128 | 1024 | ✅ **Fused AG+MM** | **1.08× faster** |
| 256 | 2048 | ✅ **Fused AG+MM** | **1.07× faster** |
| 1024 | 8192 | ✅ **Full Fused** (AG+MM+RS) | **1.14× faster** |
| 2048 | 16384 | ✅ **Full Fused** | **1.10× faster** |
| 4096 | 32768 | ✅ **Full Fused** | **1.08× faster** |

**Crossover point**: **M_shard ≈ 100-128** (M_total ≈ 800-1024 for TP=8)

In practice:
- **Decode bs=1**: M_shard = 1 → **way below crossover** → fusion hurts (2-3× slower)
- **Decode bs=8**: M_shard = 8 → **below crossover** → fusion hurts
- **Decode bs=32**: M_shard = 32 → **at crossover boundary** → roughly neutral
- **Decode bs=128+**: M_shard = 128+ → **above crossover** → fusion wins (5-14%)
- **Prefill**: M_shard = seq_len → **always above crossover** → fusion always wins

**Implication**: The AutoHeuristic (D176943) must learn this crossover point per model and per GPU. The crossover depends on:
1. **Hidden size** (larger H → more compute per token → crossover shifts left)
2. **FFN/intermediate size** (larger FFN → more compute → crossover shifts left)
3. **NVLink bandwidth** (faster NVLink → less comm time → crossover shifts right)
4. **GPU generation** (B200 has ~2× NVLink BW vs H100 → crossover shifts right)
5. **Quantization** (FP8 halves compute time → crossover shifts right by ~2×)

### 5.8 Benchmarked Reference Points

From internal Meta docs:

| Source | Model | Config | Metric | Value |
|--------|-------|--------|--------|-------|
| DeepSeek-V3 Runbook | DeepSeek-V3 671B | TP=16, H100×16, bs=4, 2K input | TTFT | **8.4 s** |
| DeepSeek-V3 Runbook | DeepSeek-V3 671B | TP=16, H100×16, bs=4, 2K input | TPOT | **53 ms** |
| DeepSeek-R1 Runbook | DeepSeek-R1 671B | TP=16, H100×16, bs=4, 2K input | QPS | **0.17** |
| GenAI Benchmarks | Qwen3-235B-A22B | TP=4, bs=128, 8K input | Avg latency | **16.9 s** |
| CustomOp Autotuning | Llama3-70B FFN | TP=8, H100×8, M=8192 | Full Fused speedup | **1.14×** |
| CustomOp Autotuning | DeepSeek-V2 FFN | TP=8, H100×8, M=8192 | CustomOp speedup | **1.09×** |
| Jongsoo Park analysis | Llama3.1-70B | TP=8, H100 | Comm per token (decode) | **~1.44 ms** (160 all-reduces × 9 µs) |
| Jongsoo Park analysis | Llama3.1-70B | TP=8, H100 | one_shot_all_reduce | **~5 µs** |

---

## 6. Prioritized Action Plan

### Priority 1 — Highest Impact, Already Feasible

| # | Gap | Action | Where | Effort | Expected Impact | Owner |
|---|-----|--------|-------|--------|-----------------|-------|
| 1 | **Gap 2** | Improve AutoHeuristic for fuse/no-fuse decisions on small decode | PyTorch Inductor | 🟢 Low | 5-15% decode latency | Already in progress (D176943/D176944/D177068) |
| 2 | **Gap 4** | Implement DeepSeek TODO: attn `all_reduce` → `reduce_scatter` | vLLM model | 🟢 Low (1-2w) | 5-10% per-layer, enables MM+RS on o_proj | vLLM-compile |
| 3 | **Gap 6** | Return full matmul result from fused MM+RS op | PyTorch `symm_mem` | 🟢 Low (1w) | Unblocks fusion for shared expert + residual patterns | PyTorch distributed |

### Priority 2 — High Impact, Medium Effort

| # | Gap | Action | Where | Effort | Expected Impact | Owner |
|---|-----|--------|-------|--------|-----------------|-------|
| 4 | **Gap 3** | Add `BLOCK_WISE` scale mode to `symm_mem` FP8 ops | PyTorch `symm_mem` | 🟡 Medium (2-3w) | Unlocks AG+MM/MM+RS for DeepSeek-R1 FP8 (~15-25% TP improvement) | PyTorch distributed |
| 5 | **Gap 2** | Adaptive SP decomposition: only decompose when batch_size > threshold | vLLM compile pass | 🟢 Low (1-2w) | Avoids regression on small decode batches | vLLM-compile |

### Priority 3 — Important but Higher Effort

| # | Gap | Action | Where | Effort | Expected Impact | Owner |
|---|-----|--------|-------|--------|-----------------|-------|
| 6 | **Gap 1** | Layer-level pipelining for opaque quant kernels | vLLM execution | 🟡 Medium (2-3w) | 10-15% for INT4/INT8 models | vLLM-compile |
| 7 | **Gap 1** | Make quant ops decomposable (long-term) | PyTorch + torchao | 🔴 High (8+w) | Enables all fusion patterns for all quant formats | torchao team |
| 8 | **Gap 5** | Last-dim FP8 scale handling | PyTorch `symm_mem` | 🟡 Medium (2-3w) | Low real-world impact | PyTorch distributed |

---

## 6. Performance Projection

### Expected Per-Layer TP Latency Savings (Decode, TP=8)

| Model | Today | After P1 Fixes | After P1+P2 Fixes | After All Fixes |
|-------|-------|----------------|--------------------| --------------- |
| **Qwen3-30B BF16** | ❌ No fusion | ✅ AG+MM, MM+RS | ✅ + adaptive heuristic | ✅ Same (already maxed) |
| **Qwen3-235B FP8** | ✅ AG+MM, MM+RS already | ✅ + heuristic tuning | ✅ Same | ✅ Same |
| **DeepSeek-R1 FP8** | ⚠️ Partial (block scales may fail) | ⚠️ + MoE o_proj fix | ✅ Block scales fixed | ✅ Full coverage |
| **Qwen2.5-72B GPTQ-INT4** | ❌ No fusion | ❌ Same (quant gap) | ❌ Same | ✅ Layer-level pipeline |
| **DeepSeek-V3 AWQ-INT4** | ❌ No fusion | ❌ Same (quant gap) | ❌ Same | ✅ Layer-level pipeline |

### Estimated Latency Improvement (End-to-End Inference)

| Scenario | Model | Improvement |
|----------|-------|-------------|
| Decode, TP=8, batch=1 | Qwen3-235B FP8 | 5-10% (heuristic tuning) |
| Decode, TP=8, batch=32 | Qwen3-235B FP8 | 15-25% (overlap benefit) |
| Prefill, TP=8, seq=4096 | DeepSeek-R1 FP8 | 10-20% (after block scale fix) |
| Decode, TP=4, batch=8 | Qwen2.5-72B GPTQ-INT4 | 0% today → 10-15% with layer pipeline |

---

## 7. Key Code References

| Component | Path | Lines |
|-----------|------|-------|
| **vLLM ColumnParallelLinear** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/layers/linear.py` | 557-575 |
| **vLLM RowParallelLinear** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/layers/linear.py` | 1388-1416 |
| **vLLM DeepSeek V2/V3 MoE** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/deepseek_v2.py` | 337-387 |
| **vLLM Qwen3 MoE** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/models/qwen3_moe.py` | 163-210 |
| **vLLM Sequence Parallelism pass** | `genai/msl/rl/vllm/omni/vllm/compilation/passes/fusion/sequence_parallelism.py` | 1-372 |
| **vLLM AsyncTP pass** | `genai/msl/rl/vllm/omni/vllm/compilation/passes/fusion/collective_fusion.py` | 1-424 |
| **vLLM Pass Manager** | `genai/msl/rl/vllm/omni/vllm/compilation/passes/pass_manager.py` | 115-125 |
| **vLLM FP8 quant** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/layers/quantization/fp8.py` | 364-646 |
| **vLLM GPTQ Marlin** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/layers/quantization/gptq_marlin.py` | — |
| **vLLM AWQ Marlin** | `genai/msl/rl/vllm/avocado_v1/vllm/model_executor/layers/quantization/awq_marlin.py` | — |
| **PyTorch symm_mem fused ops** | `torch/distributed/_symmetric_memory/__init__.py` | 443-1575 |
| **PyTorch micro_pipeline_tp pass** | `torch/_inductor/fx_passes/micro_pipeline_tp.py` | 1-1114 |
| **PyTorch FP8 scale mode check** | `torch/distributed/_symmetric_memory/__init__.py` | 512-542 |
| **PyTorch Inductor config** | `torch/_inductor/config.py` | 1001 (`_micro_pipeline_tp`) |

---

## 8. Open Questions for Discussion

1. **DeepSeek-R1 block-wise scales**: Has anyone validated whether the current `symm_mem` FP8 paths actually work or silently produce wrong results for block-wise scales? We need a correctness test.

2. **AutoHeuristic training data**: For the fuse/no-fuse decision (D176943), what decode batch sizes and TP sizes should we collect training data for? The breakeven point likely differs per GPU generation (H100 vs B200 vs H200).

3. **Marlin fusion priority**: How many production deployments use GPTQ-INT4 vs FP8 for DeepSeek/Qwen? If FP8 dominates, Gap 1 (INT4/INT8) drops in priority.

4. **MoE expert parallelism**: For DeepSeek-V3 at large TP sizes, expert parallelism (EP) may be used instead of or in addition to TP. How do the fused collectives interact with EP's all-to-all communication?

5. **B200 symm_mem performance**: The `symm_mem` fused ops use pipelined multi-stream execution. Has this been benchmarked on B200 with NVLink5? The tiling heuristics (line 904-924 in `symm_mem`) may need retuning.

---

## Appendix A: Architecture Diagram

```
Standard Transformer Layer with TP (DeepSeek/Qwen dense layers):
═══════════════════════════════════════════════════════════════

Input (replicated across TP ranks)
  │
  ├─► QKV Projection (ColumnParallel) ─── each rank computes shard
  │        │
  │   Attention (local compute)
  │        │
  │   O Projection (RowParallel)
  │        │
  │   ┌────┴──── all_reduce ────┐     ◄── Today: plain all_reduce
  │   │                         │         Ideal:  MM + RS (fused)
  │   └────┬────────────────────┘
  │        │
  │   RMSNorm + Residual
  │        │
  ├─► Gate+Up Projection (ColumnParallel) ─── each rank computes shard
  │        │                                      Ideal: AG + MM (fused)
  │   SiLU + Mul (local)
  │        │
  │   Down Projection (RowParallel)
  │        │
  │   ┌────┴──── all_reduce ────┐     ◄── Today: plain all_reduce
  │   │                         │         Ideal:  MM + RS (fused)
  │   └────┬────────────────────┘
  │        │
  │   RMSNorm + Residual
  │        │
  └─► Next Layer


With Sequence Parallelism Decomposition (vLLM compile pass):
═══════════════════════════════════════════════════════════

  O Projection (RowParallel)
       │
  ┌────┴── reduce_scatter ──┐     ◄── Fused: MM + RS ✅
  │                          │
  │   RMSNorm (on shard)     │
  │        │                 │
  │   ┌────┴── all_gather ──┐│    ◄── Fused: AG + MM ✅
  │   │                     ││
  │   Gate+Up (ColumnParallel)│
  │        │                 │
  │   SiLU + Mul (local)     │
  │        │                 │
  │   Down Projection         │
  │        │                 │
  │   ┌────┴── reduce_scatter│    ◄── Fused: MM + RS ✅
  │   │                     │
  │   RMSNorm (on shard)     │
  │        │                 │
  │   ┌────┴── all_gather   │    ◄── Fused: AG + MM ✅
  │   │                     │
  └── Next Layer QKV         │
```

## Appendix B: FP8 Scale Mode Interaction with Collectives

```
Tensor-wise scale (1 scalar):
  Scale: [1]
  All-gather: replicate scale, gather FP8 data only
  Status: ✅ Works

Row-wise sharded scale (1 scale per row, sharded across ranks):
  Scale: [M_local, 1]
  All-gather: gather both FP8 data AND scales
  Status: ✅ Works

Row-wise replicated scale (1 scale per row, full tensor):
  Scale: [M_global, 1]
  All-gather: gather FP8 data, index into full scale
  Status: ✅ Works

Block-wise scale (DeepSeek-R1):
  Activation scale: [M/128, K/128]  (e.g., [1, 128] for activations)
  Weight scale: [K/128, N/128]      (e.g., [128, 128] for weights)
  All-gather: need to gather FP8 data, figure out how to slice/gather block scales
  Status: ❌ NOT SUPPORTED — Gap 3
```
