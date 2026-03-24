# Design Doc: Better Fused AG+MM and AR+MM for Quantized Dtypes in vLLM

**Author**: Tianren Gao (tianren@meta.com)
**Team**: MSL Infra PyTorch
**Oncall**: vLLM-compile
**Date**: 2026-03-13
**Status**: Draft — 待讨论

---

## 1. 目标

在 vLLM 的 torch.compile pipeline 里，为 DeepSeek 和 Qwen 等模型实现更好的：
- **AG+MM** (All-Gather + MatMul) fusion
- **AR+MM** (All-Reduce + MatMul) fusion

需要支持的量化格式：**FP8**（包括 block-wise）和 **NVFP4**。

优先级：**Block-wise FP8 先做**（DeepSeek-R1 直接受益）。

---

## 2. 背景：TP 中的通信模式

每个 transformer layer 有 2 次 TP 通信：

```
O_proj (RowParallel) → ██ all_reduce ██ → RMSNorm → gate_up (ColumnParallel)
                          AR                            MM

down_proj (RowParallel) → ██ all_reduce ██ → RMSNorm → 下一层 QKV (ColumnParallel)
                             AR                            MM
```

**AR+MM**：all_reduce 结束后才开始 matmul，GPU 在通信期间空闲。
**AG+MM**：sequence parallelism 场景下，all_gather 结束后才开始 matmul。

---

## 3. 现状

### 3.1 已有的 fusion（在 vLLM `collective_fusion.py`）

| Pattern | 匹配的 matmul op | 替换成的 fused op | 状态 |
|---------|-----------------|------------------|------|
| `AllGatherGEMMPattern` | `aten.mm` (BF16) | `symm_mem.fused_all_gather_matmul` | ✅ |
| `AllGatherScaledMMPattern` | `aten._scaled_mm` (FP8 per-token) | `symm_mem.fused_all_gather_scaled_matmul` | ✅ |
| `AllGatherCutlassScaledMMPattern` | `_C.cutlass_scaled_mm` (FP8 per-token) | `symm_mem.fused_all_gather_scaled_matmul` | ✅ |
| `GEMMReduceScatterPattern` | `aten.mm` | `symm_mem.fused_matmul_reduce_scatter` | ✅ |
| `ScaledMMReduceScatterPattern` | `aten._scaled_mm` | `patched_fused_scaled_matmul_reduce_scatter` | ✅ |
| `CutlassScaledMMReduceScatterPattern` | `_C.cutlass_scaled_mm` | `patched_fused_scaled_matmul_reduce_scatter` | ✅ |

**注意**：这些都是 MM+RS 和 AG+MM 的 pattern。AR+MM 没有直接的 fusion pattern。
当前靠 `SequenceParallelismPass` 把 `all_reduce` 拆成 `reduce_scatter + all_gather`，间接变成 AG+MM。

### 3.2 AG+MM / AR+MM 的 gap

| 量化格式 | matmul op | AG+MM 能 fuse? | AR+MM 能 fuse? | 问题 |
|---------|-----------|---------------|---------------|------|
| BF16 | `aten.mm` | ✅ | 间接（SP分解） | 小 batch 回退 |
| FP8 per-token | `aten._scaled_mm` | ✅ | 间接（SP分解） | 小 batch 回退 |
| **FP8 block-wise** | DeepGEMM: `vllm.fp8_gemm_nt`; CUTLASS: `_C.cutlass_scaled_mm` | ❌ | ❌ | **scale shape `[M/128, K/128]` 不被 `symm_mem` 支持** |
| **NVFP4** | `cutlass_scaled_fp4_mm` / `flashinfer_scaled_fp4_mm` / `fbgemm.f4f4bf16` | ❌ | ❌ | **完全没有 pattern** |

---

## 4. 三个工作项

### 工作项 1（最高优先级）：Block-wise FP8 AG+MM

**目标**：让 DeepSeek-R1 FP8 能用 AG+MM fusion。

**问题根因**：

DeepSeek-R1 的 FP8 linear 用 block-wise 量化：
- activation scale shape: `[M/128, K/128]`（不是 per-token 的 `[M, 1]`）
- weight scale shape: `[N/128, K/128]`（不是 per-channel 的 `[1, N]`）

`symm_mem` 的 `fused_all_gather_scaled_matmul` 里 `_ScaleMode` 只有三种：

```python
class _ScaleMode(Enum):
    TENSOR_WISE       # scale 是标量 [1]
    ROW_WISE_SHARDED  # scale 是 [M_local, 1]，AG 时一起 gather
    ROW_WISE_REPLICATED  # scale 是 [M_global, 1]，已复制
```

Block-wise scale `[M/128, K/128]` 不属于任何一种。

**修改方案**：

**4.1.1 在 PyTorch `symm_mem` 里加 `BLOCK_WISE` scale mode**

文件：`torch/distributed/_symmetric_memory/__init__.py`

```python
class _ScaleMode(Enum):
    UNSCALED = "unscaled"
    TENSOR_WISE = "tensor-wise"
    ROW_WISE_SHARDED = "row-wise-sharded"
    ROW_WISE_REPLICATED = "row-wise-replicated"
    BLOCK_WISE = "block-wise"  # 新增
```

检测逻辑（修改 `_check_and_verify_fp8_all_gather_scale_mode`）：

```python
def _check_and_verify_fp8_all_gather_scale_mode(shard, scale, gather_dim, group_size):
    if scale is None:
        return _ScaleMode.UNSCALED
    if scale.numel() == 1:
        return _ScaleMode.TENSOR_WISE

    # 新增: block-wise 检测
    # block-wise scale 特征：2D tensor，行数 < shard 行数，列数 > 1
    if (scale.dim() == 2
        and scale.shape[0] != shard.shape[0]  # 不是 per-row
        and scale.shape[1] > 1):              # 不是 per-channel
        return _ScaleMode.BLOCK_WISE

    # ... 原有 ROW_WISE 逻辑
```

**4.1.2 AG 时 all-gather block scale**

在 `_fused_all_gather_scaled_matmul` 的 fallback 和 CUDA 实现里处理：

```python
if scale_mode == _ScaleMode.BLOCK_WISE:
    # input shard: [M_local, K]，scale: [M_local/block_m, K/block_k]
    # all-gather input: [M_global, K]
    # all-gather scale: [M_global/block_m, K/block_k]  ← scale 也要 gather
    A_scale = torch.ops._c10d_functional.all_gather_into_tensor(
        A_scale.contiguous(), group_size, group_name
    )
    A_scale = torch.ops._c10d_functional.wait_tensor(A_scale)
```

**4.1.3 Pipeline chunk 切分时的 scale 处理**

`symm_mem` 把 input 切成 `tp_size` 个 chunk 做 pipeline。每个 chunk 的 block scale 也要对应切：

```python
for i in range(tp_size):
    chunk_start = i * chunk_m
    chunk_end = (i + 1) * chunk_m
    A_chunk = A[chunk_start:chunk_end, :]

    if scale_mode == _ScaleMode.BLOCK_WISE:
        block_m = shard.shape[0] // scale.shape[0]  # 推导 block size
        scale_start = chunk_start // block_m
        scale_end = chunk_end // block_m
        A_scale_chunk = A_scale[scale_start:scale_end, :]
        # cuBLAS 16-byte alignment check
        if A_scale_chunk.data_ptr() % 16 != 0:
            A_scale_chunk = A_scale_chunk.clone()
```

**4.1.4 vLLM 侧确认 pattern 匹配**

DeepSeek-R1 走 CUTLASS block FP8 时，FX graph 里是 `_C.cutlass_scaled_mm`，跟现有 `CutlassScaledMMReduceScatterPattern` / `AllGatherCutlassScaledMMPattern` 的 pattern **形式上能匹配**。

需要验证：
1. pattern 里的 `scale_a` / `scale_b` 能接受 `[M/128, K/128]` 形状
2. replacement 里传给 `symm_mem` fused op 的参数正确

如果走 DeepGEMM（`fp8_gemm_nt`），则需要在 vLLM 加新 pattern（后续工作）。

**预期收益**：

| 场景 | 通信隐藏量 | 预估加速 |
|------|-----------|---------|
| Prefill seq=2048, TP=8 | ~120 µs/层 × 61 层 | TTFT -10~20% |
| Prefill seq=8192, TP=8 | ~450 µs/层 × 61 层 | TTFT -15~22% |
| Decode bs≥4, TP=16 | ~5-9 µs/层 × 61 层 | TPOT -5~9% |

---

### 工作项 2：AR+MM 方案 2 — `fused_allreduce_matmul` in PyTorch

**目标**：在 PyTorch `symm_mem` 里实现 `fused_allreduce_matmul`，让 vLLM 不需要 SP 分解就能做 AR+MM fusion。

**设计原理**：

`all_reduce` 在内部等于 `reduce_scatter + all_gather`。
`fused_allreduce_matmul` = `reduce_scatter` (不能 overlap) + `fused_all_gather_matmul` (可以 overlap)。

```
现在 (SP 分解方案):
  vLLM FX pass 1: all_reduce → reduce_scatter + all_gather    (SequenceParallelismPass)
  vLLM FX pass 2: all_gather + mm → fused_all_gather_matmul   (AsyncTPPass)
  问题：两个 pass，FX graph 被大改，小 batch 可能回退

方案 2 (fused_allreduce_matmul):
  vLLM FX pass:   all_reduce + mm → fused_allreduce_matmul    (一步到位)
  内部实现：reduce_scatter + fused_all_gather_matmul
  好处：一个 op，一个 pattern，不需要 SP 分解
```

**4.2.1 Op 定义**

文件：`torch/distributed/_symmetric_memory/__init__.py`

```python
lib.define(
    "fused_allreduce_matmul("
    "Tensor input, Tensor[] Bs, str reduce_op, "
    "str group_name"
    ") -> (Tensor, Tensor[])",
    tags=("pt2_compliant_tag",),
)

lib.define(
    "fused_allreduce_scaled_matmul("
    "Tensor input, Tensor[] Bs, "
    "Tensor A_scale, Tensor[] B_scales, "
    "str reduce_op, str group_name, "
    "Tensor?[] biases, Tensor?[] result_scales, "
    "ScalarType?[] out_dtypes, bool[] use_fast_accum"
    ") -> (Tensor, Tensor[])",
    tags=("pt2_compliant_tag",),
)
```

**4.2.2 Meta impl（用于 tracing）**

```python
@torch.library.impl(lib, "fused_allreduce_matmul", "Meta")
def _fused_allreduce_matmul_fallback(input, Bs, reduce_op, group_name):
    group = c10d._resolve_process_group(group_name)
    # reduce_scatter + all_gather 不改变 shape
    # output shape 跟普通 all_reduce + matmul 一样
    result = input  # shape 不变
    mm_outputs = [input @ B for B in Bs]
    return result, mm_outputs
```

**4.2.3 CUDA impl**

```python
@torch.library.impl(lib, "fused_allreduce_matmul", "CUDA")
def _fused_allreduce_matmul(input, Bs, reduce_op, group_name):
    group = c10d._resolve_process_group(group_name)

    # Step 1: reduce_scatter（同步，必须等完成）
    scatter_dim = 0
    input_shard = torch.ops._c10d_functional.reduce_scatter_tensor(
        input.contiguous(), reduce_op, group.size(), group_name
    )
    input_shard = torch.ops._c10d_functional.wait_tensor(input_shard)

    # Step 2: fused_all_gather_matmul（AG 与 MM pipeline overlap）
    # 复用已有实现
    full_input, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        input_shard, Bs, gather_dim=scatter_dim, group_name=group_name
    )

    return full_input, mm_outputs
```

Scaled 版本类似，Step 2 调 `fused_all_gather_scaled_matmul`。

**4.2.4 vLLM 侧的 pattern**

在 `collective_fusion.py` 里加新 pattern：

```python
class AllReduceGEMMPattern(BasePattern):
    def pattern(input, weight):
        ar = tensor_model_parallel_all_reduce(input)
        return torch.ops.aten.mm.default(ar, weight)

    def replacement(input, weight):
        full_input, mm_outputs = torch.ops.symm_mem.fused_allreduce_matmul(
            input, [weight], "sum", tp_group_name
        )
        return mm_outputs[0]
```

**注意**：这个 pattern 需要跨过 RMSNorm。因为实际的 FX graph 是：
```
all_reduce → RMSNorm → matmul
```
不是直接 `all_reduce → matmul`。

**两种处理方式**：
1. Pattern 匹配 `all_reduce → RMSNorm → matmul` 整体替换（复杂，要把 RMSNorm 也包进去）
2. 保留 SP 分解方式（把 RMSNorm 放在 RS 和 AG 之间），用 `fused_allreduce_matmul` 替代分开的 RS + AG + MM

方式 2 更实际：`fused_allreduce_matmul` 本质就是 RS + fused_AG_MM，可以把 RMSNorm 插在 RS 和 AG 之间：

```python
@torch.library.impl(lib, "fused_allreduce_matmul", "CUDA")
def _fused_allreduce_matmul(input, Bs, reduce_op, group_name):
    # Step 1: reduce_scatter
    input_shard = reduce_scatter(input)

    # 调用方在这里可以插 RMSNorm:
    # normed = rmsnorm(input_shard)

    # Step 2: fused AG+MM
    full_input, mm_outputs = fused_all_gather_matmul(input_shard, Bs, ...)
    return full_input, mm_outputs
```

**实际上这就需要把 op 拆成两步暴露出来**，这样 RMSNorm 能插在中间。

**更现实的方案**：保持 vLLM 的 SP 分解（把 AR 拆成 RS + AG），但把 AG+MM 的 fusion 做好（工作项 1），同时用 AutoHeuristic（你的 D176943）来避免小 batch 回退。这样 AR+MM 就通过 SP 分解 + AG+MM fusion 间接实现了。

**`fused_allreduce_matmul` 作为独立 op 的价值**：当 AR 和 MM 之间**没有 RMSNorm** 或其他计算时（某些模型架构），可以直接用。对大多数 transformer 模型，AR 和 MM 之间都有 RMSNorm，所以直接用的场景有限。

---

### 工作项 3（后续）：NVFP4 AG+MM

**目标**：让 NVFP4 量化模型也能用 AG+MM fusion。

**NVFP4 是什么**：

NVIDIA FP4 量化格式。weight 用 4-bit 浮点存储，两个值打包成 1 个 `uint8`。
- weight: `uint8`, shape `[N, K/2]` (packed FP4)
- weight_scale: `float8_e4m3fn`, shape `[N, K/group_size]`, group_size=16（per-block scale）
- weight_global_scale: `float32` 标量
- input_global_scale_inv: `float32` 标量

运行时：
1. `scaled_fp4_quant(x_bf16, input_global_scale_inv)` → `x_fp4` (uint8) + `x_blockscale` (fp8)
2. `cutlass_scaled_fp4_mm(x_fp4, weight, x_blockscale, weight_scale, alpha, output_dtype)` → output

**FX graph 里的 op**（取决于 backend）：

```
CUTLASS:     vllm._C.cutlass_scaled_fp4_mm(x_fp4, weight, x_blockscale, weight_scale, alpha, dtype)
FlashInfer:  flashinfer_scaled_fp4_mm(x_fp4, weight, x_blockscale, weight_scale, alpha, dtype, backend)
FBGEMM:      fbgemm.f4f4bf16(x_fp4, weight, x_blockscale, weight_scale, alpha)
Marlin:      apply_fp4_marlin_linear(input, weight, weight_scale, ...)
```

**跟 FP8 的关键区别**：

1. **Input 也需要 runtime 量化**。FP8 per-token 的 input 量化是一步 `cast + compute_scale`；NVFP4 的 input 量化是 `scaled_fp4_quant`，产出 packed uint8 + block scale。这意味着 AG+MM pipeline 里每个 chunk 都需要先量化再 matmul。

2. **matmul op 完全不同**。不是 `aten.mm`，不是 `aten._scaled_mm`，是 `cutlass_scaled_fp4_mm`——参数完全不同（多了 alpha、block scale 格式不同）。

**方案选择（Decompose vs Per-op Pattern）**：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **Decompose** | 让 `cutlass_scaled_fp4_mm` decompose 成 `dequant_fp4 → aten.mm` | 通用，现有 AG+MM pattern 自动 work | 丢失 CUTLASS FP4 kernel 性能；Inductor 无法重新 fuse quant+mm |
| **Per-op Pattern** | 写 `AllGatherNvFp4MMPattern` + `fused_all_gather_fp4_matmul` | 保留 CUTLASS FP4 kernel 性能 | 每种 backend 都要写一套；代码量大 |

**建议**：先做 block-wise FP8（工作项 1），积累经验后再做 NVFP4。NVFP4 大概率需要 per-op pattern 方案，因为 FP4 kernel 的 dequant 和 matmul 深度耦合，decompose 性能损失太大。

---

## 5. 实施计划

| 阶段 | 工作项 | 在哪做 | 内容 | 预估时间 |
|------|--------|-------|------|---------|
| **Phase 1** | Block-wise FP8 AG+MM | **PyTorch** `symm_mem` | 加 `_ScaleMode.BLOCK_WISE`，修改 scale gather/slice 逻辑 | 2-3 weeks |
| **Phase 1** | Block-wise FP8 AG+MM | **vLLM** `collective_fusion.py` | 验证现有 pattern 兼容 block scale，必要时修改 | 1 week |
| **Phase 1** | AutoHeuristic | **vLLM** config + passes | 让 AG+MM fusion 对小 batch adaptive（你的 D176943 集成） | 2 weeks |
| **Phase 2** | AR+MM | **PyTorch** `symm_mem` | 加 `fused_allreduce_matmul` op（RS + fused_AG_MM） | 2-3 weeks |
| **Phase 2** | AR+MM | **vLLM** `collective_fusion.py` | 加 `AllReduceGEMMPattern` | 1 week |
| **Phase 3** | NVFP4 AG+MM | **PyTorch** `symm_mem` + **vLLM** | 加 FP4 pipeline + pattern | 4-6 weeks |

---

## 6. Benchmark 计划

### 模型
- **DeepSeek-R1 FP8** (671B, block-wise FP8, TP=8/16)
- **Qwen3-235B-A22B FP8** (MoE, TP=4/8)
- **Qwen2.5-72B BF16** (dense, TP=8) — baseline comparison

### Configs
1. **Baseline**：无 SP，无 AsyncTP（当前默认）
2. **SP + AG+MM only**：有 SP 分解 + AG+MM fusion
3. **SP + AG+MM + AutoHeuristic**：adaptive，小 batch 不 fuse
4. **AR+MM direct**：用 `fused_allreduce_matmul`（Phase 2 之后）

### Batch sizes
- Decode: 1, 4, 8, 32
- Prefill: 128, 512, 2048, 8192

### 指标
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Throughput (tokens/s)

### 命令
```bash
# Baseline
vllm bench latency --model <model> --tp 8 \
  --batch-size <bs> --input-len <len> --output-len 128 --num-iters 30

# SP + AG+MM fusion
vllm bench latency --model <model> --tp 8 \
  --batch-size <bs> --input-len <len> --output-len 128 --num-iters 30 \
  -cc.pass_config.enable_sp=true -cc.pass_config.fuse_gemm_comms=true
```

---

## 7. 关键代码位置

| 组件 | 文件 | 说明 |
|------|------|------|
| symm_mem fused ops | `torch/distributed/_symmetric_memory/__init__.py` | AG+MM, MM+RS 的 fused op 实现 |
| Scale mode 检查 | 同上, lines 512-542 | `_check_and_verify_fp8_all_gather_scale_mode` |
| AG+MM pipeline | 同上, lines 545-684 | `_fused_all_gather_matmul_impl` |
| Inductor micro_pipeline_tp | `torch/_inductor/fx_passes/micro_pipeline_tp.py` | Inductor 的 async TP pass |
| AutoHeuristic fuse/no_fuse | 同上 + `autoheuristic/artifacts/_AsyncTPFuseH100.py` | 你的 D176943 |
| vLLM AsyncTP pass | `vllm/compilation/passes/fusion/collective_fusion.py` | pattern matching + replacement |
| vLLM SP pass | `vllm/compilation/passes/fusion/sequence_parallelism.py` | AR → RS + AG 分解 |
| vLLM pass manager | `vllm/compilation/passes/pass_manager.py` | pass 注册和执行 |
| vLLM PassConfig | `vllm/config/compilation.py` | enable_sp, fuse_gemm_comms |
| FP8 block-wise linear | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `W8A8BlockFp8LinearOp` |
| NVFP4 linear | `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` | `apply_nvfp4_linear` |
| DeepSeek model | `vllm/model_executor/models/deepseek_v2.py` | MoE + TP 结构 |

---

## 8. 待讨论

1. **Block-wise FP8**：DeepSeek-R1 走的是 DeepGEMM 还是 CUTLASS block FP8？如果走 DeepGEMM，需要先确认 `fp8_gemm_nt` 这个 op 在 FX graph 里是否可 trace / 有 pattern 可匹配。

2. **AR+MM**：考虑到 AR 和 MM 之间总有 RMSNorm，`fused_allreduce_matmul` 作为单一 op 的实用价值有限。更现实的路径可能是：优化 SP 分解 + AG+MM（工作项 1）+ AutoHeuristic（D176943）。`fused_allreduce_matmul` 可以作为 API 暴露，但短期内 vLLM 还是用 SP 分解路径。

3. **性能基线**：需要先跑一次 DeepSeek-R1 FP8 在 TP=8 下的 profiling，确认 TP 通信占总时间的比例，验证 fusion 的理论收益。

4. **B200 适配**：B200 NVLink 带宽翻倍，crossover point 会右移（更大的 batch 才值得 fuse）。AutoHeuristic 需要在 B200 上重新训练。
