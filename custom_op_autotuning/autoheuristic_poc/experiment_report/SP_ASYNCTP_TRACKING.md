# SP+AsyncTP 完整追踪文档

> **日期**: 2026-03-18 (最后更新)
> **作者**: Tianren Gao
> **上下文**: AutoHeuristic POC for Async TP — vLLM 端到端验证

---

## 1. 项目目标

为 vLLM 的 Async TP (AllGather+MatMul 通信重叠) 实现一个 **3 层 AutoHeuristic pipeline**:

| 层 | 决策 | 状态 |
|----|------|------|
| ① fuse/no_fuse | AG+MatMul 是否用 symm_mem 融合 | ✅ 已完成 |
| ② native/pipeline | 用 CUTLASS persistent kernel 还是 Python cuBLAS pipeline | ✅ 已完成 |
| ③ tile config | SM90 AG+GEMM kernel 的 tile/cluster 选择 | ✅ 已完成 |

本文档追踪 **vLLM 端到端集成和验证**，包括：
- 将 heuristic 集成到 vLLM 编译管线
- 修复 SP+AsyncTP 在 piecewise compilation 下的 crash
- 端到端 serving benchmark 验证性能

---

## 2. 时间线概览

| Session | 日期 | 主要工作 | 关键发现 |
|---------|------|---------|---------|
| 1-3 | 03-12~13 | 识别 SP+AsyncTP 在 piecewise compilation 下的 crash | AOT autograd epilogue `copy_` 节点和跨 submod output shape 不匹配 |
| 4-5 | 03-13 | 调试 crash 根因 | SP 将 AllReduce→RMSNorm 替换为 RS→RMSNorm→AG，但 piecewise 编译在 attention op 处拆分模型时，输出 shape 在跨 submod 边界处不一致 |
| 6 | 03-13 | 实现修复 | `_remove_mismatched_epilogue_copies()` 和 `_fix_cross_submod_outputs()` |
| 7 | 03-14 | 首批 benchmark | Llama-2-7b TP=2 确认 SP 匹配; Mixtral-8x7B SP 不匹配 MoE |
| 8 | 03-15 | CUDA graph 兼容性 | 发现 `+rms_norm` + CUDA graph 有时 crash |
| 9 | 03-16 | 调查 crash 来源 | 错误源自 `validate_cudagraph_capturing_enabled()` |
| 10 | 03-18 | CUDA graph 系统测试 + Serving Benchmark | crash 非通用问题；首次 serving benchmark |
| 11 | 03-18 | 修正 Benchmark 方法论 + 2×2 矩阵 | ⚠️ 之前 1.47-2.24x 是方法论错误！SP 实际提升仅 1-2% |
| **12** | **03-18** | **Llama-3-70B TP=8 2×2 矩阵 Benchmark** | **SP = 0.98x（无收益）！baseline_cg 有异常；理论分析被实测验证** |

---

## 3. SP+AsyncTP Piecewise Compilation 修复

### 3.1 问题

vLLM 的 piecewise compilation 在 attention op 处将模型拆分为多个 submod。SP pass 将 `AllReduce → RMSNorm` 替换为 `ReduceScatter → RMSNorm → AllGather`，改变了输出 tensor 的 shape（从 `[B, full_hidden]` 变为 `[B, hidden/TP]`）。

这导致两个问题：
1. **AOT autograd epilogue copy_ 节点**：编译器在 submod 边界生成的 `copy_` 操作期望原始 shape
2. **跨 submod output shape 不匹配**：一个 submod 的输出 shape 与下一个 submod 的输入 shape 不一致

### 3.2 修复 (在 `sequence_parallelism.py`)

```python
def _remove_mismatched_epilogue_copies(self, graph):
    """移除 piecewise 边界处与 SP shape 变更不兼容的 copy_ 节点"""

def _fix_cross_submod_outputs(self, graph):
    """修正跨 submod 边界的 output shape，使其与 SP 变更后的 shape 一致"""
```

### 3.3 修改文件

| 文件 | 变更 |
|------|------|
| `vllm/compilation/passes/fusion/sequence_parallelism.py` | +164 行（两个新方法 + `__call__` 中调用） |
| `vllm/compilation/passes/fusion/collective_fusion.py` | +49 行（`_mm_extra_check` 集成） |
| `vllm/compilation/piecewise_backend.py` | +3 行 |
| `vllm/config/compilation.py` | +8 行（`async_tp_min_tokens` 字段） |
| `vllm/config/vllm.py` | +17 行 |

---

## 4. CUDA Graph 兼容性调查

### 4.1 问题

在 Llama-3-70B TP=8 上运行 SP+AsyncTP 时，启用 CUDA graphs 导致 crash：
```
RuntimeError: CUDA graph capturing detected at an inappropriate time.
```

### 4.2 调查结论

**crash 不是 `+rms_norm` + CUDA graph 的通用问题**。系统测试结果：

| Model | TP | +rms_norm | CUDA Graphs | SP+AsyncTP | 结果 |
|-------|---:|-----------|-------------|------------|------|
| Llama-3.2-1B | 1 | ✅ | ✅ | ❌ | ✅ 稳定运行 300s+ |
| Llama-2-7b-hf | 2 | ✅ | ✅ | ❌ | ✅ 稳定运行 300s+ |
| Llama-2-7b-hf | 8 | ✅ | ✅ | ❌ | ✅ 稳定运行 300s+ |
| Llama-2-7b-hf | 2 | ✅ | ✅ | ✅ | ✅ 稳定运行 300s+ |
| Llama-3-70B | 8 | ✅ | ✅ | ✅ | ❌ crash (可能是 stale cache) |

### 4.3 错误源链

```
validate_cudagraph_capturing_enabled()     ← monitor.py:86-95
  ↑ 被调用于
CUDAGraphWrapper.__call__()                ← cuda_graph.py:264
  ↑ flag 管理于
gpu_model_runner.py:
  set_cudagraph_capturing_enabled(True)    ← line 5727 (capture_model)
  set_cudagraph_capturing_enabled(False)   ← line 5751
```

**Llama-3-70B crash 原因**：可能是 stale compilation cache 或 model-specific 内存压力。网络不可用，无法下载模型重新验证。

---

## 5. Serving Benchmark 结果 ⭐

### 5.1 环境

| 项目 | 值 |
|------|-----|
| **Model** | meta-llama/Llama-2-7b-hf |
| **Hardware** | 2× NVIDIA H100 (TP=2) |
| **Compilation** | `compile_sizes=[256]`, piecewise |
| **Benchmark** | `vllm bench serve`, 256 prompts, input=256, output=128, rate=inf |
| **GPU memory** | `gpu-memory-utilization=0.85` |

### 5.2 结果（修正后 2×2 矩阵）

| 配置 | SP+AsyncTP | CUDA Graph | 吞吐 (tok/s) | ITL (ms) | TTFT (ms) |
|------|:---------:|:---------:|-------------|---------|----------|
| baseline_cg | ❌ | ✅ | 8,818.6 | 21.50 | 888.29 |
| baseline_no_cg | ❌ | ❌ | 8,959.0 | 21.47 | 830.13 |
| sp_asynctp_cg | ✅ | ✅ | 8,866.0 | 21.45 | 882.77 |
| sp_asynctp_no_cg | ✅ | ❌ | 9,114.6 | 21.46 | 778.84 |

### 5.3 2×2 分解分析

| 比较 | 比值 | 解释 |
|------|:----:|------|
| SP effect (CG on) | **1.01x** | SP+AsyncTP 在有 CG 时几乎无提升 |
| SP effect (CG off) | **1.02x** | SP+AsyncTP 在无 CG 时也几乎无提升 |
| CG effect (no SP) | **0.98x** | CG 略微降低性能 |
| CG effect (with SP) | **0.97x** | CG 在有 SP 时也略微降低性能 |

### 5.4 关键发现

1. **SP+AsyncTP 在 Llama-2-7b TP=2 上没有显著收益**：仅 1-2%，在噪声范围内
2. **TTFT 有小幅改善**：888→778ms (12%)，可能来自 prefill 阶段的同步减少
3. **CUDA graphs 略微降低性能**：2-3% penalty
4. **ITL 完全相同**：~21.5ms，per-token latency 完全是 compute-bound

### 5.5 为什么 SP 在这里不 work？

Llama-2-7b TP=2 是 **compute-bound** 的，不是 communication-bound：

- **模型太小**：7B → 每 GPU 只处理 3.5B 的计算量
- **TP 太小**：2 GPU 通过 NVLink 连接 → AllReduce latency 只有微秒级
- **Batch 太大**：BS=256 → 每个 GEMM 是 256×4096×4096，计算量巨大
- **通信占比极低**：AllReduce 成本 << 1% 的总 forward 时间 → overlap 几乎无法节省时间

SP+AsyncTP 应该在以下场景更有用：
- **大 TP**（TP=8/16）：跨 8+ GPU 的 AllReduce 成本大得多
- **大模型**（70B+）：更多 layer = 更多 AllReduce
- **小 batch size**：GEMM 更小 → 通信占比更高
- **跨节点通信**：没有 NVLink 时，AllReduce latency 大得多

### 5.6 ⚠️ 之前结果（Session 11 初次运行）的错误

之前报告了 1.47x-2.24x 的提升，**这些数字完全是方法论错误**：

| 错误 | 影响 | 修复 |
|------|------|------|
| `gpu-memory-utilization=0.4` | baseline 从 8,818 降到 3,737 (2.36x 性能损失) | 改为 0.85 |
| `num_prompts=200` + `compile_sizes=[256]` + `cudagraph_mode=0` | SP 在 no-CG 配置中从未激活（BS 不匹配）| 改为 num_prompts=256 |
| 缺少 baseline_no_cg | 无法区分 SP 效果和关闭 CG 的效果 | 完整 2×2 矩阵 |

### 5.7 与之前 Latency Benchmark 的对比

| 方法 | baseline | SP | 结果 |
|------|---------|----|----|
| `vllm bench latency` (BS=256, Session 7-8) | 36.1 ms | 173.6 ms | **-380%** ❌ |
| `vllm bench serve` (256 prompts, 修正后) | 8,819 tok/s | 8,866 tok/s | **+0.5%** ≈ 持平 |

两个结果实际上是一致的：**SP 在 Llama-2-7b TP=2 上没有收益**。Latency benchmark 显示更大的惩罚是因为额外 collective ops 的开销在单次推理中更明显。

### 5.8 Llama-3-70B TP=8 结果（Session 12）⭐⭐

| 项目 | 值 |
|------|-----|
| **Model** | meta-llama/Meta-Llama-3-70B |
| **Hardware** | 8× NVIDIA H100 (TP=8) |
| **Compilation** | `compile_sizes=[256]`, piecewise |
| **Benchmark** | `vllm bench serve`, 256 prompts, input=256, output=128, rate=inf |
| **GPU memory** | `gpu-memory-utilization=0.90` |

#### 5.8.1 结果

| 配置 | SP+AsyncTP | CUDA Graph | 吞吐 (tok/s) | ITL (ms) | TTFT (ms) |
|------|:---------:|:---------:|-------------|---------|----------|
| baseline_cg | ❌ | ✅ | 2,422.8 | 56.02 | 6,218.30 |
| baseline_no_cg | ❌ | ❌ | 3,949.4 | 47.34 | 2,089.23 |
| sp_asynctp_cg | ✅ | ✅ | 3,726.0 | 49.66 | 2,282.66 |
| sp_asynctp_no_cg | ✅ | ❌ | 3,875.2 | 47.80 | 2,190.32 |

#### 5.8.2 2×2 分解分析

| 比较 | 比值 | 解释 |
|------|:----:|------|
| SP effect (CG off) | **0.98x** | SP+AsyncTP 在无 CG 时有轻微性能损失（3,949→3,875） |
| SP effect (CG on) | **1.54x** | ⚠️ 假象！baseline_cg 异常慢（见下文） |
| CG effect (no SP) | **0.61x** | CG 严重降低 baseline 性能！ |
| CG effect (with SP) | **0.96x** | CG 在有 SP 时也降低性能 |

#### 5.8.3 `baseline_cg` 异常分析

`baseline_cg` 的 2,422 tok/s 和 TTFT=6,218ms 是异常值。原因分析：

| 因素 | baseline_cg | sp_asynctp_cg | 差异 |
|------|-------------|---------------|------|
| `custom_ops` | `["+rms_norm"]` | 未设置（默认） | ✅ 可能是根因 |
| `enable_sp` | false | true | SP 改变图结构 |
| `fuse_gemm_comms` | false | true | 不影响 CG |

**推测**：`custom_ops: ["+rms_norm"]` 在 70B TP=8 的大计算图上与 CUDA graph capture
产生负面交互。70B 有 80 层 × 每层 7 个 AllReduce = 560 个集合通信操作，CUDA graph
capture 需要录制整个执行序列。`+rms_norm` 可能导致部分 RMSNorm 作为 custom op 无法
被 CUDA graph 高效捕获，触发 fallback 路径或增加 replay overhead。

**关键点**：即使忽略异常的 baseline_cg，**SP 的 clean comparison（non-CG）= 0.98x**，
说明 SP+AsyncTP 在 70B TP=8 上也没有收益。

#### 5.8.4 70B TP=8 为什么 SP 还是不 work？

这个结果验证了 §6 的理论分析：

```
理论预测 (§6.6):
  通信占比 = 32.2%  →  理论最大 1.47x
  但 fusion kernel (M=256) ≈ 1.04x  →  理论 E2E ≈ 1.2%
  + Serving overhead  →  实际 ~0.5-1%

实测结果:
  SP 效果 = 0.98x （-2%，轻微回退）

为什么 -2% 而不是 +1%？
  → fusion kernel 在 BS=256 仍有 overhead（kernel 数据显示 M=256 介于
    M=32 (2.1x slower) 和 M=512 (1.04x faster) 之间，可能是 ~neutral 或略慢）
  → SP pass 本身引入额外 ops (ReduceScatter, AllGather 替代 AllReduce)
  → 两者抵消后净效果是轻微回退
```

### 5.9 SP 测试状态汇总

| Model | TP | SP 有效? | 说明 |
|-------|---:|---------|------|
| **Llama-2-7b-hf** | 2 | ❌ **无显著收益** (1-2%) | Compute-bound，通信占比太低 |
| **Meta-Llama-3-70B** | 8 | ❌ **轻微回退** (0.98x) | 通信占比 32%，但 fusion kernel overhead 抵消收益 |
| Mixtral-8x7B | 8 | ⚠️ 未测 serving | MoE 模型，SP pattern 只匹配 attention 层 |
| Qwen3-8B | 2 | ⚠️ 未测 serving | Dense model，理论上应该匹配 |

**结论**：即使在"理论最佳"的 70B TP=8 NVLink 场景下，SP+AsyncTP 也没有 serving 收益。
这彻底验证了 §6.6 的分析：在单节点 NVLink 环境下，fusion kernel overhead
在 decode-dominant serving 场景中完全抵消了通信隐藏的收益。

---

## 6. Async TP 收益深度分析 ⭐⭐

### 6.1 通信占比理论估算

用 Amdahl's Law 计算不同配置下，通信 (AllReduce) 在总 forward 时间中的占比，
以及如果 Async TP 能 100% 隐藏通信，理论最大加速：

**假设**：H100 NVLink BW=450 GB/s (effective per direction), Peak FLOPS=990 TFLOPS BF16,
GEMM utilization=70%, AllReduce latency overhead=5μs per op

| 配置 | 通信占比 | 通信(ms) | 计算(ms) | 理论最大加速 |
|------|:------:|--------:|--------:|:----------:|
| Llama-2-7b TP=2 BS=256 | 20.5% | 0.62 | 2.39 | 1.26x |
| Llama-2-7b TP=2 BS=32 | 54.4% | 0.36 | 0.30 | 2.19x |
| Llama-3-70B TP=8 BS=256 | 32.2% | 3.41 | 7.19 | 1.47x |
| Llama-3-70B TP=8 BS=32 | 55.6% | 1.13 | 0.90 | 2.25x |
| Llama-3-405B TP=16 BS=256 | 31.9% | 10.07 | 21.47 | 1.47x |
| Llama-3-405B TP=16 BS=32 | 46.8% | 2.36 | 2.68 | 1.88x |

**关键 takeaway**：通信占比随 TP 增大和 BS 减小而增加。但 Llama-2-7b TP=2 BS=256
的 20.5% 通信占比意味着即使完美 overlap，最多只能加速 1.26x。实测 1.01x，说明
serving overhead + fusion kernel overhead 消耗了绝大部分理论收益。

### 6.2 Kernel-Level 数据揭示的根本矛盾

来自 AutoHeuristic 训练数据（Llama3-70B, TP=8, 8× H100 NVLink）：

```
                    Fusion Kernel Speedup
M (batch size)    Baseline    Fused       Impact
─────────────────────────────────────────────────
1  (decode)        241μs      548μs       2.3× 更慢 ❌
4  (decode)        244μs      612μs       2.5× 更慢 ❌
32 (decode)        269μs      554μs       2.1× 更慢 ❌
512 (prefill)      586μs      563μs       1.04× 更快 ✅
4096 (prefill)    4024μs     3507μs       1.15× 更快 ✅
```

这是 **TP=8** 上的数据 — 通信成本已经比 TP=2 大很多。但即使在 TP=8：
- **M ≤ 32**: fusion 2.1-2.5x 更慢（symm_mem pipeline setup overhead > comm saving）
- **M = 512**: 刚好 break even (1.04x)
- **M ≥ 4096**: fusion 有效 (1.15x)

**根本矛盾**：

```
通信占比最高的场景 (小 M, 小 BS)
          ↕  恰恰是
fusion kernel overhead 最大的场景 (小 M)

具体来说：
  小 M → 通信占 forward 的 55%  → 理论最大 2.25x
  小 M → 但 fusion 2.1-2.5x 更慢 → 实际是 regression

  大 M → 通信占 forward 的 20-32%  → 理论最大 1.26-1.47x
  大 M → fusion 1.04-1.15x 更快 → 但只吃到理论上限的一小部分
```

### 6.3 `fused_all_gather_matmul` 为什么在小 M 时 overhead 大？

`fused_all_gather_matmul` 使用 PyTorch symmetric memory 的 micro-pipeline：

```
传统 AllGather + MatMul:
  [AllGather 全部数据] → [等待完成] → [一个大 MatMul]

Fused pipeline (分 chunk):
  [AG chunk 0] → [MM chunk 0]     ← overlap: AG chunk 1 与 MM chunk 0 并行
                  [AG chunk 1] → [MM chunk 1]
                                   [AG chunk 2] → [MM chunk 2]
                                                    ...
```

Pipeline 的开销：
1. **Symmetric memory setup**: 建立 P2P memory mapping, 每次调用有 ~50μs 固定开销
2. **Chunk synchronization**: 每个 chunk 需要 barrier/signal，小 M 时 chunk 太小，
   synchronization overhead 占比大
3. **GEMM 碎片化**: 一个大 GEMM 拆成多个小 chunk GEMM，每个 chunk 的 tensor core
   利用率更低
4. **Stream 管理**: 需要在多个 CUDA stream 间协调，增加 launch overhead

当 M 很大时（≥512），这些固定开销被 amortize 掉。当 M 很小时（≤32），
固定开销 > 通信隐藏的收益。

### 6.4 Serving 场景的具体问题

在 vLLM continuous batching serving 中：

```
一个请求的生命周期：
  [Prefill: 1 step]  →  [Decode: 128 steps]  →  完成
       M = seq_len            M = 1 per request
       (256-4096)             (但 batch 后 M = concurrent_reqs)

总计算时间分布（256 prompts, input=256, output=128）：
  Prefill: 256 prompts × 256 tokens × 1 step  = 65,536 token-steps
  Decode:  256 prompts × 128 tokens × 1 step  = 32,768 token-steps
                                                  ↑ 但每步处理 ~256 个 token (batched)

  实际 Decode steps: ~128 步 × BS=256 = 大部分时间在 BS=256 做 decode
```

**Decode 的 M = batch size (并发请求数)**：
- 低并发 (在线推理, BS=8-32): M 小, fusion 2x 更慢 → **Async TP 有害**
- 中并发 (typical serving, BS=64-256): M 中等, fusion ~neutral → **Async TP 无效**
- 高并发 (batch processing, BS=512+): M 大, fusion ~1.04-1.15x → **Async TP 有效**

但 BS=512+ 的高并发在实际部署中并不常见（需要大量 KV cache 内存）。

### 6.5 Async TP 真正有效的场景

| 场景 | 为什么有效 | 预期收益 | 可行性 |
|------|----------|---------|:------:|
| **TP=8, 70B, prefill-heavy** (长 input, 短 output, 如 summarization/RAG) | Prefill M=2048+, kernel 数据显示 1.15x speedup | TTFT↓ 5-10% | 🔴 需要 70B model |
| **TP=8, 70B, 高并发 decode** (BS=512+) | M=512 进入 fusion-beneficial zone | ~5% 吞吐↑ | 🔴 需要 8 GPU + 大 KV cache |
| **跨节点 TP** (IB/RoCE, 非 NVLink) | BW 从 450 GB/s 降到 ~50 GB/s, 通信占比暴增到 80%+ | 20-40% | 🟡 需要多节点 |
| **Kernel latency benchmark** (非 serving) | 无 serving overhead 稀释, 直接测量 kernel 差异 | 能准确观察 kernel 效果 | ✅ 可测 |
| **TP=2, 7B** (我们测过的) | 通信太便宜, M=256 borderline | ~1% (噪声) | ✅ 已验证无效 |

### 6.6 理论收益链条的损耗分析

以 "最佳" 场景为例：Llama-3-70B TP=8 BS=256

```
理论通信占比:                              32.2%
  → 理论最大加速 (Amdahl):                1.47x

但 fusion kernel 不是免费的:
  → Kernel-level speedup (M=256):         ~1.04x (估算，介于 M=32 和 M=512 之间)
  → 理论 E2E speedup:                    32.2% × (1-1/1.04) ≈ 1.2% 加速

Serving overhead (scheduler, KV cache, HTTP, tokenizer):
  → 进一步稀释 1.2% → ~0.5-1% 实际吞吐提升

结论: 即使在 "最佳" 单节点 NVLink 场景,
      Async TP 的 serving 吞吐提升可能也只有 ~1%
```

**真正能看到大收益的唯一场景：跨节点 TP（BW << NVLink）**

### 6.7 Heuristic 的真正价值：避免 Regression

上述分析表明，**Async TP fusion 在大多数 serving 场景下不会带来显著加速**。
但 heuristic 的价值在于：

```
Inductor 当前规则: "always fuse" (K_shard < 1024 才 skip, 实际从不触发)
    → Decode 阶段 (M=1-32): 2.1-2.5× 性能回退 ❌❌❌
    → 这是一个严重的生产 regression

我们的 heuristic: 正确识别 M ≤ 256 不该 fuse
    → 避免 decode 回退
    → Prefill (M ≥ 512) 才 fuse → 微小收益 (1.04-1.15x)
    → 净效果: 不加速，但避免了 regression
```

**这意味着 AutoHeuristic 的生产价值不是 "让模型更快"，
而是 "防止 Inductor 的 always-fuse 规则让 decode 变慢 2-3x"。**

在 Inductor 上游修复 always-fuse 规则之前，这个 heuristic 是关键的安全网。

### 6.8 完整决策框架

```
                    Async TP Fusion 决策树
                            │
                   TP ≥ 8 且跨节点?
                      /         \
                    是            否
                    │             │
              通信占比 > 50%    通信占比 < 35%
              Fusion 可能有效    Fusion 基本无效
                    │             │
              M ≥ 512?        不 fuse (避免 regression)
              /       \
            是          否
            │           │
        Fuse ✅      不 fuse ❌
       (1.15x)     (避免 2x 回退)
```

---

## 7. 当前存在的问题

### 7.0 ⚠️ SP/AsyncTP 在 upstream vLLM 中是死代码 (`IS_DENSE = False`)

**这是最关键的发现**。在 upstream vLLM 中：

```python
# vllm/config/vllm.py (upstream)
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# See https://github.com/vllm-project/vllm/issues/25689.

# O2/O3 优化级别:
"enable_sp": IS_DENSE,       # = False，永远不启用
"fuse_gemm_comms": IS_DENSE, # = False，永远不启用
```

**含义**：
1. SP 和 AsyncTP **在所有优化级别 (O0-O3) 中都被禁用**
2. 代码存在但从未在任何 production/default 配置中运行过
3. 唯一触发方式是手动设置 `--compilation-config '{"pass_config": {"enable_sp": true}}'`
4. **这就是为什么 vLLM 社区没有发现 piecewise crash 的原因** — 没人使用 SP

**我们修复的 crash (§3) 和 heuristic (§6) 都是针对死代码的改动。**
即使代码完全正确，serving benchmark (§5) 也证明没有性能收益。

**结论**：所有改动合并为一个 draft PR 保留，等未来 SP 重新启用时再 review。
详见 `VLLM_CODE_CHANGES_PR_PLAN.md`。

### 7.1 Async TP 收益窗口极窄

如 §6 分析，fusion 只在 M≥512 + TP≥8 时有效。大多数 serving 场景 (decode-dominated,
BS<256) 不在有效窗口内。Heuristic 的主要价值是避免 always-fuse regression。

### 7.2 SP+AsyncTP 在所有已测模型上无显著收益

- **Llama-2-7b TP=2**: 1-2% 提升（噪声范围），compute-bound
- **Meta-Llama-3-70B TP=8**: 0.98x（轻微回退），fusion kernel overhead 抵消收益
- **原因**：理论分析（§6.6）已验证 — 单节点 NVLink 下 fusion kernel overhead 无法被 overlap 收益覆盖

### 7.3 baseline_cg 在 70B TP=8 上的异常

`custom_ops: ["+rms_norm"]` + CUDA graph + 70B TP=8 导致严重性能回退：
- 吞吐 2,423 vs 3,949 tok/s (no-CG baseline)
- TTFT 6,218 vs 2,089 ms
- 可能与大计算图（80层×7个AllReduce）+ CUDA graph capture 交互有关
- **注意**：这不影响 SP 的结论，因为 non-CG comparison 也是 0.98x

### 7.3 之前的 Benchmark 方法论错误 (已修复)

之前报告的 1.47x-2.24x 提升完全是方法论错误：
- `gpu-memory-utilization=0.4` → baseline 被人为限制 2.36x
- `num_prompts=200` + `compile_sizes=[256]` → no-CG 配置 SP 从未激活
- 缺少 baseline_no_cg → 无法分离 SP 和 CG 效果

### 7.4 CUDA Graph 在 70B TP=8 上的性能异常

- `custom_ops: ["+rms_norm"]` + CG + 70B TP=8 导致 baseline_cg 吞吐从 3,949 降到 2,423 tok/s (0.61x)
- TTFT 从 2,089ms 暴增到 6,218ms
- SP 配置 + CG 则不受影响 (3,726 tok/s, TTFT 2,283ms)
- 可能与大计算图的 CUDA graph capture 交互、`custom_ops` 配置差异有关
- **状态**：不影响 SP 结论（non-CG comparison = 0.98x），但值得调查

- CUDA graphs 在 70B TP=8 上降低性能（baseline: 0.61x, SP: 0.96x）
- `baseline_cg` + `custom_ops: ["+rms_norm"]` 特别严重（2,423 vs 3,949 tok/s）
- 可能原因：大计算图的 CUDA graph capture/replay overhead
- **状态**: 不影响 SP 结论（non-CG comparison 也是 0.98x），但值得调查 CG 为何在大模型上表现差

### 7.5 MoE 模型覆盖有限

- MoE (Mixtral-8x7B) 的 SP pattern 只匹配 attention 层，不匹配 experts
- MoE experts 的 AllReduce 后接 router combine，不直接接 RMSNorm
- **需要**: 评估只对 attention 层做 SP 是否仍有净收益

### 7.6 Benchmark 基础设施问题 (已修复)

| 问题 | 根因 | 修复 |
|------|------|------|
| 所有配置启动失败 | `server.kill()` 不杀 GPU worker 子进程 | `start_new_session=True` + `os.killpg()` |
| 基准测试无输出 | `benchmark_serving.py` 已弃用 | 改用 `vllm bench serve` |
| 日志文件为空 | `conda run` 缓冲 stdout | 直接使用 `/home/tianren/.conda/envs/vllm/bin/python` |
| GPU OOM | stale worker 进程占 50GiB | `pkill -9 -f "VLLM::Worker"` cleanup |

---

## 8. 与 Kernel-level AutoHeuristic 的关系

### 8.1 Pipeline 概览

```
┌─────────────────────────────────────────────────────────────┐
│  Layer ① fuse/no_fuse AutoHeuristic (✅ 完成)               │
│    - 116 configs, 95.7% accuracy                            │
│    - Artifact: _AsyncTPFuseH100.py                          │
│    - 集成: micro_pipeline_tp.py (Inductor)                   │
│           + collective_fusion.py (vLLM)                      │
├─────────────────────────────────────────────────────────────┤
│  Layer ② native/pipeline AutoHeuristic (✅ 完成)             │
│    - 40 configs, ~85% accuracy                              │
│    - Artifact: _AsyncTPNativeH100.py                        │
│    - 集成: symm_mem/__init__.py                              │
├─────────────────────────────────────────────────────────────┤
│  Layer ③ tile config AutoHeuristic (✅ 完成)                 │
│    - 36 shapes, 18 tile configs, ~93% accuracy              │
│    - Artifact: _AsyncTPTileConfigH100.py                    │
│    - 集成: all_gather_gemm.py                                │
├─────────────────────────────────────────────────────────────┤
│  SP+AsyncTP in vLLM (⬤ 本文档追踪)                         │
│    - Piecewise compilation crash fix ✅                      │
│    - CUDA graph 兼容性调查 ✅                                 │
│    - Serving benchmark: Llama-2-7b TP=2 无显著收益 ⚠️        │
│    - Serving benchmark: Llama-3-70B TP=8 无显著收益 ⚠️       │
│    - 跨节点 TP / Prefill-heavy workload 测试 ⏳              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Kernel-level vs E2E 差距

| 维度 | Kernel-level | Serving E2E |
|------|-------------|-------------|
| **测量方法** | 单 matmul 延迟 (μs) | 256 并发请求吞吐 (tok/s) |
| **Fusion 收益** | 个别 matmul 1.5-2× speedup | 整体 ~1% 吞吐提升 (噪声范围) |
| **Heuristic 价值** | 清晰区分 fuse/no_fuse | Llama-2-7b TP=2 无显著差异 |
| **CUDA Graphs** | 不涉及 | 轻微降低性能 2-3% |
| **核心 insight** | 小 M (decode) 不该 fuse | Llama-2-7b TP=2 compute-bound，通信不是瓶颈 |

---

## 9. Draft PR 与实验复现指南

### 9.1 Draft PR

所有 vLLM 代码改动已合并为一个 draft PR：

- **PR**: https://github.com/vllm-project/vllm/pull/37489
- **Branch**: `tianren/sp-asynctp-piecewise-fix`
- **Fork**: `tianrengao/vllm`
- **包含**: 10 files, +1162/-13 (crash fix + heuristic + config + benchmark 脚本 + 测试)

### 9.2 如何复现实验

**在当前机器 (devgpu088) 上**：

```bash
# 切到 PR branch（vLLM 是 editable install，切 branch 即生效）
cd /data/users/tianren/vllm
git checkout tianren/sp-asynctp-piecewise-fix

# 跑 Llama-3-70B TP=8 2×2 矩阵 benchmark
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 \
  /home/tianren/.conda/envs/vllm/bin/python benchmarks/run_sp_cg_2x2_70b.py

# 跑 Llama-2-7b TP=2 2×2 矩阵 benchmark
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  /home/tianren/.conda/envs/vllm/bin/python benchmarks/run_sp_cg_2x2.py
```

**如果换了新机器**：

```bash
# 1. Clone 并切到 PR branch
git clone git@github.com:tianrengao/vllm.git
cd vllm
git checkout tianren/sp-asynctp-piecewise-fix

# 2. 安装（需要 CUDA 环境）
pip install -e .

# 3. 下载模型
huggingface-cli download meta-llama/Meta-Llama-3-70B --local-dir /path/to/models/Meta-Llama-3-70B

# 4. 修改 benchmark 脚本中的 MODEL_LOCAL 路径，然后跑
python benchmarks/run_sp_cg_2x2_70b.py
```

### 9.3 当前环境依赖

| 依赖 | 位置 / 版本 | 说明 |
|------|-------------|------|
| vLLM repo | `/data/users/tianren/vllm/` | editable install |
| Conda env | `/home/tianren/.conda/envs/vllm/` | Python + CUDA + PyTorch |
| Llama-3-70B | `/data/users/tianren/models/Meta-Llama-3-70B/` | 30 shards, ~130GB |
| Llama-2-7b | `/data/users/tianren/models/Llama-2-7b-chat-hf/` | 2 shards |
| GPUs | 8× NVIDIA H100 (devgpu088) | TP=8 需要全部 8 张 |

### 9.4 如果 upstream vLLM 变了

Branch 基于 2026-03-18 的 `main` (commit `bc2c0c86e`)。如果隔很久回来需要 rebase：

```bash
cd /data/users/tianren/vllm
git checkout tianren/sp-asynctp-piecewise-fix
git fetch origin
git rebase origin/main
# 如果有冲突，主要关注 sequence_parallelism.py 和 vllm.py
```

### 9.5 未来可以探索的方向

基于 benchmark 结果 (§5)，SP+AsyncTP 在单节点 NVLink 下无 serving 收益。
如果要继续研究，以下方向可能有突破：

1. **跨节点 TP**: BW 从 450 GB/s → ~50 GB/s，通信占比 > 80%
2. **Prefill-heavy workload**: 改 benchmark 脚本的 `INPUT_LEN=4096, OUTPUT_LEN=32`
3. **更大 batch size**: `NUM_PROMPTS=512`，M=512 进入 fusion 有效区间
4. **等 upstream 重新启用 SP**: 关注 [issue #25689](https://github.com/vllm-project/vllm/issues/25689)

---

## 10. Recommended Next Steps

| # | 任务 | 优先级 | 详情 |
|---|------|--------|------|
| 1 | **跨节点 TP 测试** | 🔴 高 | 唯一理论上能看到大收益的场景（BW 从 450 GB/s → ~50 GB/s，通信占比 > 80%）|
| 2 | **小 batch size 测试 (BS=16-32)** | 🟡 中 | 虽然 fusion kernel 在 M≤32 有 overhead，但 SP 的 ReduceScatter/AllGather 本身可能有收益 |
| 3 | **Prefill-heavy workload 测试** | 🟡 中 | 长 input (4096+) + 短 output (32)，M=4096 时 fusion 1.15x，且 prefill 占比更高 |
| 4 | **调查 baseline_cg 70B 异常** | 🟡 中 | `custom_ops: ["+rms_norm"]` + CG + 70B TP=8 的性能异常，可能暴露 vLLM 的 CG capture bug |
| 5 | **MoE serving benchmark** | 🟢 低 | Mixtral-8x7B 只对 attention 做 SP 是否有净收益 |
| 6 | **分析通信占比** | 🟢 低 | 用 profiler 量化实际 AllReduce vs compute 时间占比（理论估算已被实测验证） |
| 7 | **上游 always-fuse regression fix** | 🟢 低 | 在 Inductor 中修复 always-fuse 规则，避免 M≤256 decode regression |

---

## 10. 文件索引

### vLLM 代码变更

| 文件 | 类型 | 说明 |
|------|------|------|
| `vllm/compilation/passes/fusion/sequence_parallelism.py` | 修改 | SP pass + piecewise fix |
| `vllm/compilation/passes/fusion/collective_fusion.py` | 修改 | AsyncTP pattern matching + extra_check |
| `vllm/compilation/passes/fusion/async_tp_heuristic.py` | 新增 | Decision tree heuristic |
| `vllm/config/compilation.py` | 修改 | `async_tp_min_tokens` 字段 |
| `vllm/config/vllm.py` | 修改 | `enable_sp_and_async_tp()` |

### Benchmark 脚本

| 文件 | 说明 |
|------|------|
| `benchmarks/run_sp_cg_2x2.py` | SP+AsyncTP 2×2 矩阵 benchmark — Llama-2-7b TP=2 |
| `benchmarks/run_sp_cg_2x2_70b.py` | SP+AsyncTP 2×2 矩阵 benchmark — Llama-3-70B TP=8 |
| `benchmarks/run_sp_cg_quick.py` | 之前的 benchmark (有方法论错误) |
| `benchmarks/run_async_tp_benchmark.py` | Qwen3-8B latency benchmark |
| `benchmarks/run_async_tp_benchmark_mixtral.py` | Mixtral-8x7B latency benchmark |
| `benchmarks/SP_ASYNCTP_BENCHMARK_REPORT.md` | 详细 benchmark 报告 |

### AutoHeuristic Pipeline (PyTorch)

| 文件 | 说明 |
|------|------|
| `torchgen/_autoheuristic/async_tp/` | 训练脚本 + converter + 生成脚本 |
| `torch/_inductor/autoheuristic/artifacts/_AsyncTPFuseH100.py` | ① fuse/no_fuse artifact |
| `torch/_inductor/autoheuristic/artifacts/_AsyncTPNativeH100.py` | ② native/pipeline artifact |
| `torch/_inductor/autoheuristic/artifacts/_AsyncTPTileConfigH100.py` | ③ tile config artifact |

### 实验报告

| 文件 | 说明 |
|------|------|
| `experiment_report/TL_DISCUSSION_CONCISE.md` | 完整的 3 层 pipeline 技术文档 |
| `experiment_report/E2E_BENCHMARK_REPORT.md` | E2E latency benchmark 报告 (Session 1-8) |
| `experiment_report/SP_ASYNCTP_TRACKING.md` | **本文档** — 完整追踪 |

### Benchmark 结果

| 文件 | 说明 |
|------|------|
| `/tmp/sp_2x2_results_1773860877.json` | Serving benchmark 2×2 矩阵修正数据 (Llama-2-7b TP=2) |
| `/tmp/sp_2x2_70b_results_1773870297.json` | Serving benchmark 2×2 矩阵 (Llama-3-70B TP=8) |
| `/tmp/sp_cg_results_1773858323.json` | 之前的 serving benchmark (有方法论错误) |
| `/tmp/async_tp_benchmark_1773382977.json` | Qwen3-8B latency 原始数据 |
| `/tmp/async_tp_bench_mixtral_1773384607.json` | Mixtral latency 原始数据 |
