# vLLM Code Changes 分析与 PR 拆分计划

> **日期**: 2026-03-18
> **作者**: Tianren Gao
> **状态**: 已合并为一个 Draft PR → https://github.com/vllm-project/vllm/pull/37489

---

## 0. 最终决定

所有改动合并为一个 draft PR 保留，不拆分。原因：
- SP 在 upstream vLLM 中是死代码（`IS_DENSE = False`，见 [issue #25689](https://github.com/vllm-project/vllm/issues/25689)）
- Serving benchmark 显示无性能收益（Llama-2-7b TP=2: 1.01x, Llama-3-70B TP=8: 0.98x）
- 改动的价值是未来参考和回归避免，不急于 merge

### Draft PR 信息

| 项目 | 值 |
|------|-----|
| **PR** | https://github.com/vllm-project/vllm/pull/37489 |
| **Branch** | `tianren/sp-asynctp-piecewise-fix` |
| **Fork** | `tianrengao/vllm` |
| **Base** | `main` (commit `bc2c0c86e`, 2026-03-18) |
| **Files** | 10 files, +1162/-13 |

### 如何回来跑实验

**在当前机器 (devgpu088)**：

```bash
# 切到 PR branch（vLLM 是 editable install，切 branch 即生效）
cd /data/users/tianren/vllm
git checkout tianren/sp-asynctp-piecewise-fix

# Llama-3-70B TP=8 benchmark
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 \
  /home/tianren/.conda/envs/vllm/bin/python benchmarks/run_sp_cg_2x2_70b.py

# Llama-2-7b TP=2 benchmark
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  /home/tianren/.conda/envs/vllm/bin/python benchmarks/run_sp_cg_2x2.py
```

**换了新机器**：

```bash
git clone git@github.com:tianrengao/vllm.git && cd vllm
git checkout tianren/sp-asynctp-piecewise-fix
pip install -e .
huggingface-cli download meta-llama/Meta-Llama-3-70B --local-dir ./models/Meta-Llama-3-70B
# 修改 benchmarks/run_sp_cg_2x2_70b.py 中 MODEL_LOCAL 路径后运行
```

**如果 upstream 变了需要 rebase**：

```bash
git fetch origin && git rebase origin/main
# 冲突主要在 sequence_parallelism.py 和 vllm.py
```

### 当前环境依赖

| 依赖 | 位置 | 说明 |
|------|------|------|
| vLLM repo | `/data/users/tianren/vllm/` | editable install，切 branch 即生效 |
| Conda env | `/home/tianren/.conda/envs/vllm/` | Python + CUDA + PyTorch |
| Llama-3-70B | `/data/users/tianren/models/Meta-Llama-3-70B/` | 30 shards, ~130GB |
| Llama-2-7b | `/data/users/tianren/models/Llama-2-7b-chat-hf/` | 2 shards |
| GPUs | 8× NVIDIA H100 (devgpu088) | TP=8 需要全部 8 张 |

---

## 1. 改动总览

所有代码改动都在 **vLLM repo** (`/data/users/tianren/vllm/`)。
PyTorch repo 没有 SP/AsyncTP 相关代码改动（现有 4-commit stack 是 "out variants with decomposition in inductor"，无关）。

### 代码文件改动汇总

| # | 文件 | 类型 | 改动行数 | 改动内容 |
|---|------|------|---------|----------|
| 1 | `vllm/compilation/passes/fusion/sequence_parallelism.py` | M | +164 | Piecewise crash fix + async_tp_min_tokens gating |
| 2 | `vllm/compilation/passes/fusion/collective_fusion.py` | M | +49 | Per-matmul heuristic extra_check |
| 3 | `vllm/compilation/passes/fusion/async_tp_heuristic.py` | **New** | +50 | Decision tree heuristic |
| 4 | `vllm/config/compilation.py` | M | +8 | `async_tp_min_tokens` 配置字段 |
| 5 | `vllm/config/vllm.py` | M | +17 | `enable_sp_and_async_tp()` 启用逻辑 |
| 6 | `vllm/compilation/piecewise_backend.py` | M | +1 | Debug 用，不入 PR |

### 非代码文件

| 类型 | 文件 | 说明 |
|------|------|------|
| Benchmark | `benchmarks/run_sp_cg_2x2.py`, `run_sp_cg_2x2_70b.py` | 2×2 矩阵 benchmark |
| Report | `benchmarks/SP_ASYNCTP_BENCHMARK_REPORT.md` | Benchmark 报告 |
| Test | `tests/compile/distributed/test_async_tp_heuristic.py` | Heuristic 单元测试 |
| Debug | 大量 `verify_*.py`, `verify_*.sh`, `debug_*.py` | 调试用，不入 PR |
| Doc | PyTorch 侧 `SP_ASYNCTP_TRACKING.md`, `E2E_BENCHMARK_REPORT.md` | 追踪文档 |

---

## 2. PR 拆分建议

### PR 1: Fix SP+Piecewise Compilation Crash 🔴 高优先级

**价值**: 修复一个真实的 crash bug。任何用 SP + piecewise compilation 的用户都会遇到。

#### 涉及文件

| 文件 | 改动 |
|------|------|
| `sequence_parallelism.py` | `_remove_mismatched_epilogue_copies()` + `_fix_cross_submod_outputs()` + pre-SP output recording |

#### Bug 描述

SP pass 将 `AllReduce → RMSNorm` 替换为 `ReduceScatter → RMSNorm → AllGather`，
改变了 residual tensor 的 shape（从 `[batch, hidden]` 变为 `[batch/TP, hidden]`）。
但 vLLM 的 piecewise compilation 在 attention op 处将模型拆分为多个 submod，
这导致两个 crash：

**Crash 1: AOT autograd epilogue `copy_` shape mismatch**

```
# AOT autograd 在 submod 边界生成:
copy_(graph_input[batch, hidden], mutated_value[batch/TP, hidden])
#     ↑ 原始 shape                    ↑ SP 修改后的 shape
# → RuntimeError: shape mismatch in copy_
```

**Crash 2: 跨 submod output shape 不匹配**

```
# submod_0 输出: residual[batch/TP, hidden]  (SP 修改后)
# submod_1 输入: 期望 residual[batch, hidden]  (原始 shape)
# → RuntimeError: assert_size_stride failed
```

#### 修复方案

```python
# Fix 1: 移除无效的 epilogue copy_ 节点
def _remove_mismatched_epilogue_copies(self, graph):
    """当 copy_(dst, src) 的 dst.shape != src.shape 时，
    说明 SP 已经改变了 src 的 shape，copy_ 不再有效。
    直接移除，因为 SP 的 in-place op 已经通过 slice view 完成了 mutation。"""
    for node in graph.nodes:
        if node.target is torch.ops.aten.copy_.default:
            dst, src = node.args[0], node.args[1]
            if dst.shape != src.shape:  # SP 造成的 mismatch
                node.replace_all_uses_with(dst)
                graph.erase_node(node)

# Fix 2: 在 submod output 处插入 AllGather 恢复原始 shape
def _fix_cross_submod_outputs(self, graph, output_node, pre_sp_output_info):
    """比较 SP 前后的 output shape。
    任何 dim-0 缩小的 output（被 ReduceScatter 缩小）
    都需要 AllGather 恢复到原始 shape，以保证下一个 submod 能接收正确的输入。"""
    for i, out in enumerate(output_args):
        pre_shape = pre_sp_output_info[i]
        post_shape = out.meta["val"].shape
        if pre_shape[0] != post_shape[0]:  # dim-0 被 SP 缩小
            all_gather = graph.call_function(
                torch.ops.vllm.all_gather.default,
                args=(out, 0, tp_size, tp_group.unique_name),
            )
            new_outputs[i] = all_gather
```

#### 如何复现 Bug

```bash
# 1. 启动 vLLM server with SP enabled + piecewise compilation
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --compilation-config '{"compile_sizes": [256], "pass_config": {"enable_sp": true, "fuse_gemm_comms": true}}'

# 2. 发送请求 → 第一次 forward pass 后 crash
curl http://localhost:8000/v1/completions -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello", "max_tokens": 10}'
```

**Error**: `RuntimeError: assert_size_stride(...)` 或 `copy_: shape mismatch`

#### Before / After

| | Before (crash) | After (fixed) |
|---|---|---|
| SP + piecewise | ❌ RuntimeError crash | ✅ 正常运行 |
| SP without piecewise | ✅ 正常（不涉及 submod 拆分） | ✅ 正常 |
| No SP | ✅ 正常 | ✅ 正常（不触发修复逻辑） |

#### 注意事项

⚠️ **以下改动不应入 PR 1**（是为测试降低门槛用的）：
```python
# 这两行改动需要 revert，保持 upstream 默认值
SP_MIN_HIDDEN_SIZE = {90: 8192}   # 被改为 1，需 revert
SP_MIN_PER_GPU_SIZE_MB = {90: 8}  # 被改为 0.001，需 revert
```
这些参数控制 "SP 最低 hidden_size 门槛"。上游默认 8192（只对 70B+ 模型启用 SP），
我们为了在 7B 模型上测试改成了 1。PR 中应保持 upstream 默认值或单独讨论是否降低。

#### Minimal Test

```python
# tests/compile/distributed/test_sp_piecewise_fix.py
"""Test that SP + piecewise compilation doesn't crash at submod boundaries."""

def test_sp_piecewise_no_crash():
    """Verify SP graph transformations maintain correct output shapes
    across piecewise submod boundaries."""
    # Create a mock graph with AllReduce → RMSNorm pattern
    # Apply SP pass (should transform to ReduceScatter → RMSNorm → AllGather)
    # Verify:
    #   1. No copy_ nodes with mismatched shapes remain
    #   2. Output shapes match pre-SP shapes (AllGather inserted)
    pass
```

实际测试需要 multi-GPU 环境 (TP≥2)，可以用 `torchrun --nproc-per-node=2` 运行。

---

### PR 2: Per-matmul Heuristic Gating for Async TP Fusion 🟡 中优先级

**价值**: 防止 Inductor 的 always-fuse 规则在 decode 阶段造成 2-3x 性能回退。
这是生产环境中的 **regression avoidance**，不是加速。

#### 涉及文件

| 文件 | 改动 |
|------|------|
| `async_tp_heuristic.py` | **New** — `should_fuse_async_tp(M, K, N)` 决策树 |
| `collective_fusion.py` | `_mm_extra_check()` + 6 处 `extra_check=_mm_extra_check` |
| `compilation.py` | `async_tp_min_tokens: int | None = Field(default=128)` |
| `sequence_parallelism.py` | `is_applicable_for_range()` 中添加 `async_tp_min_tokens` 检查 |

#### 问题描述

Inductor 当前的 collective fusion 规则是 "always fuse"（只在 `K_shard < 1024` 时
skip，实际上从不触发）。这导致 `fused_all_gather_matmul` / `fused_matmul_reduce_scatter`
在 **decode 阶段** (M=1~32) 被使用，但这些 fused kernel 在小 M 时
**比 unfused 慢 2.1-2.5 倍**。

```
Fusion Kernel Performance (Llama3-70B, TP=8, H100 NVLink):

M (batch)    Baseline    Fused       Impact
────────────────────────────────────────────
1  (decode)    241μs      548μs     2.3× 更慢 ❌
4  (decode)    244μs      612μs     2.5× 更慢 ❌
32 (decode)    269μs      554μs     2.1× 更慢 ❌
512 (prefill)  586μs      563μs     1.04× 更快 ✅
4096 (prefill) 4024μs     3507μs    1.15× 更快 ✅
```

#### 修复方案

**两层防护**:

1. **Coarse-grained**: `async_tp_min_tokens` — compile_range.end < 128 时跳过 SP+AsyncTP 整体
2. **Fine-grained**: `should_fuse_async_tp(M, K, N)` — per-matmul 决策树，在 pattern matching 时根据 shape 决定是否 fuse

```python
# Layer 1: compilation.py
async_tp_min_tokens: int | None = Field(default=128)

# Layer 2: async_tp_heuristic.py
def should_fuse_async_tp(M: int, K: int, N: int) -> bool:
    """Decision tree from AutoHeuristic (D176943)."""
    m_times_n = M * N
    if m_times_n <= 2_457_600:  # M*N small (decode-like)
        if K <= 9_600:
            return True    # small shapes, fuse
        else:
            return N > 34_816  # large K needs large N to compensate
    else:  # large M*N (prefill-like)
        ...  # more refined checks

# Layer 2 integration: collective_fusion.py
def _mm_extra_check(match: pm.Match) -> bool:
    M, K = lhs_shape[0], lhs_shape[1]
    N = rhs_shape[1]
    return should_fuse_async_tp(M, K, N)

pm.register_replacement(
    pattern, replacement, inputs, pm.fwd_only, pm_pass,
    extra_check=_mm_extra_check,  # ← 新增
)
```

#### Before / After

| | Before (always fuse) | After (heuristic gating) |
|---|---|---|
| Decode (M=1-32) | ❌ Fused → 2.1-2.5x 更慢 | ✅ Not fused → 保持 baseline |
| Prefill (M=512+) | ✅ Fused → 1.04-1.15x 更快 | ✅ Fused → 1.04-1.15x 更快 |
| 净效果 | Decode regression | 无 regression，prefill 微小加速 |

#### 如何复现 Bug (无 heuristic 时的 regression)

```bash
# 在没有 heuristic 的 vLLM 上，启用 async TP:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B \
    --tensor-parallel-size 8 \
    --compilation-config '{"pass_config": {"enable_sp": true, "fuse_gemm_comms": true}}'

# Decode 阶段 (BS=1-32) 会使用 fused kernel，导致 2-3x 延迟增加
# 可通过 vllm bench latency 观察:
python -m vllm.entrypoints.cli.main bench latency \
    --model meta-llama/Meta-Llama-3-70B -tp 8 \
    --batch-size 1 --input-len 256 --output-len 128
```

#### Minimal Test

已有: `tests/compile/distributed/test_async_tp_heuristic.py`

```python
class TestShouldFuseAsyncTP:
    def test_small_batch(self):
        # Decode shapes → should fuse (by decision tree)
        assert should_fuse_async_tp(1, 8192, 28672) == True

    def test_no_fuse_large_m_small_n(self):
        # Large M + small N → should NOT fuse
        assert should_fuse_async_tp(4096, 8192, 4096) == False

class TestRangeGating:
    def test_small_range_blocked(self):
        # Range(1,1) < min_tokens=128 → blocked
        assert _check_threshold(Range(1, 1), 128) == False

    def test_large_range_allowed(self):
        # Range(512,512) >= min_tokens=128 → allowed
        assert _check_threshold(Range(512, 512), 128) == True
```

---

### PR 3: SP/AsyncTP Enablement Config 🟢 低优先级

**价值**: 将 SP/AsyncTP 启用条件从静态 `IS_DENSE` 改为运行时检查。

#### 涉及文件

| 文件 | 改动 |
|------|------|
| `vllm.py` | `enable_sp_and_async_tp()` + O2/O3 优化级别配置 |

#### 改动描述

```python
# Before: 静态常量
"enable_sp": IS_DENSE,       # 所有 dense 模型都启用
"fuse_gemm_comms": IS_DENSE, # 包括 TP=1 也启用

# After: 运行时检查
def enable_sp_and_async_tp(cfg) -> bool:
    return cfg.parallel_config.tensor_parallel_size > 1 and is_cuda()

"enable_sp": enable_sp_and_async_tp,      # 只在 TP>1 + CUDA 时启用
"fuse_gemm_comms": enable_sp_and_async_tp, # 同上
```

**这个改动可以合入 PR 1 或 PR 2**，不需要单独 PR。

---

### 不入 PR 的改动

| 改动 | 原因 |
|------|------|
| `piecewise_backend.py` (+1 line) | Debug 用，将 `return range_entry.runnable(*args)` 拆成两行 |
| `SP_MIN_HIDDEN_SIZE: 8192→1` | 为测试降低门槛，PR 中应保持 upstream 默认 |
| `SP_MIN_PER_GPU_SIZE_MB: 8→0.001` | 同上 |
| 所有 `verify_*.py`, `verify_*.sh` | 调试脚本 |
| `benchmarks/run_sp_cg_*.py` | Benchmark 脚本，可保留但不入 PR |
| `benchmarks/debug_sp_asynctp.py` | 调试脚本 |

---

## 3. Serving Benchmark 结论（为什么 "效果不好"）

即使代码修复和 heuristic 都正确工作，SP+AsyncTP 在 **单节点 NVLink serving** 场景下
**没有显著吞吐提升**：

| Model | TP | SP 效果 | 原因 |
|-------|---:|:------:|------|
| Llama-2-7b | 2 | 1.01-1.02x | Compute-bound，通信占比 <20% |
| **Llama-3-70B** | **8** | **0.98x** | 通信占比 32%，但 fusion kernel overhead 抵消收益 |

**根本原因**: fusion kernel (`fused_all_gather_matmul`) 在 serving decode 的主要 batch size (1-256) 下
有固定 overhead（symm_mem setup ~50μs, chunk sync, GEMM fragmentation），
恰好抵消了通信隐藏的收益。只有在 M≥512+ 时 fusion 才有效 (1.04-1.15x)，
但 serving 的绝大部分时间在 decode (M=BS)，且 BS<512。

**Heuristic 的价值不在加速，在于 regression avoidance**:
- 阻止 Inductor always-fuse 规则在 M≤256 decode 时使用 fused kernel
- 避免 2-3x decode 延迟回退

---

## 4. 建议操作

| # | 操作 | 说明 |
|---|------|------|
| 1 | **PR 1 先开 draft** | Piecewise crash fix 是独立的 bug fix，最容易 review |
| 2 | **PR 2 可等** | Heuristic gating 依赖 PR 1，且 serving 效果不明显 |
| 3 | **保留 working tree** | 不 `git checkout` — 所有改动留在 uncommitted 状态 |
| 4 | **备份实验文档** | `SP_ASYNCTP_TRACKING.md` 记录了 12 个 session 的完整分析 |

---

## 5. 文件地图

```
/data/users/tianren/vllm/
├── vllm/compilation/passes/fusion/
│   ├── sequence_parallelism.py    ← PR 1 (crash fix) + PR 2 (threshold)
│   ├── collective_fusion.py       ← PR 2 (extra_check)
│   └── async_tp_heuristic.py      ← PR 2 (new file, decision tree)
├── vllm/config/
│   ├── compilation.py             ← PR 2 (async_tp_min_tokens)
│   └── vllm.py                    ← PR 3 or merge into PR 1/2
├── tests/compile/distributed/
│   └── test_async_tp_heuristic.py ← PR 2 test
├── benchmarks/
│   ├── run_sp_cg_2x2.py           ← Llama-2-7b benchmark (保留)
│   ├── run_sp_cg_2x2_70b.py       ← Llama-3-70B benchmark (保留)
│   └── SP_ASYNCTP_BENCHMARK_REPORT.md  ← 报告 (保留)
└── verify_*.py, verify_*.sh       ← 调试用 (可清理)

/data/users/tianren/pytorch/
└── custom_op_autotuning/autoheuristic_poc/experiment_report/
    ├── SP_ASYNCTP_TRACKING.md      ← 12 session 完整追踪 (保留)
    ├── E2E_BENCHMARK_REPORT.md     ← E2E benchmark 报告 (保留)
    └── VLLM_CODE_CHANGES_PR_PLAN.md ← 本文档
```
