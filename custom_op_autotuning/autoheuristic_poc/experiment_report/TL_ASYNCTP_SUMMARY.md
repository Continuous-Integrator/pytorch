# Async TP AutoHeuristic — TL 汇报总结

> **Author**: Tianren Gao | **Date**: 2026-03-18
> **Duration**: 13 sessions (03-06 ~ 03-18)
> **Hardware**: 8× H100 NVLink, devgpu088

---

## 一句话总结

我们为 Async TP 构建了完整的 **3 层 AutoHeuristic pipeline**（fuse/no_fuse → native/pipeline → tile config），全部训练完成并集成到 PyTorch Inductor；同时修复了 vLLM 中 SP+AsyncTP 的 piecewise compilation crash。但 E2E serving benchmark 显示 **在单节点 NVLink 环境下，SP+AsyncTP 没有 serving 收益**（Llama-2-7b TP=2: 1-2%，Llama-3-70B TP=8: 0.98x）。更重要的是，发现 **SP 在 upstream vLLM 中是死代码**（`IS_DENSE=False` 硬编码禁用）。

---

## 1. 完成了什么

### 1.1 PyTorch 3 层 AutoHeuristic Pipeline（✅ 全部完成）

| 层 | 决策 | 数据量 | Accuracy | 净 overhead 减少 | 状态 |
|----|------|--------|----------|------------------|------|
| ① fuse/no_fuse | AG+MatMul 是否用 symm_mem 融合 | 116 configs | **95.7%** | +2.6% vs always-fuse | ✅ Artifact + Inductor 集成 |
| ② native/pipeline | CUTLASS persistent kernel vs Python cuBLAS pipeline | 40 configs | ~85% | +3-9% (小模型 decode) | ✅ Artifact + symm_mem 集成 |
| ③ tile config | SM90 AG+GEMM 的 tile/cluster 选择（18 种） | 36 shapes | ~93% | +2.07% avg, **+11% peak** | ✅ PR #177068 |

**核心价值：替换了 3 处硬编码 heuristic，提供了 shape-aware 的编译期决策。**

### 1.2 vLLM SP+AsyncTP 修复（✅ crash fix 完成，❌ 无 serving 收益）

| 工作 | 状态 | 说明 |
|------|------|------|
| Piecewise compilation crash fix | ✅ | `_remove_mismatched_epilogue_copies()` + `_fix_cross_submod_outputs()` |
| Per-matmul heuristic 集成 | ✅ | `should_fuse_async_tp(M,K,N)` decision tree 集成到 `AsyncTPPass` |
| `async_tp_min_tokens` 阈值 | ✅ | 跳过 compile_range.end < 128 的 SP+AsyncTP |
| Llama-2-7b TP=2 serving benchmark | ✅ | **1-2% 提升**（噪声范围） |
| **Llama-3-70B TP=8 serving benchmark** | ✅ | **0.98x（轻微回退）** |
| Draft PR | ✅ | [vllm#37489](https://github.com/vllm-project/vllm/pull/37489) |

---

## 2. 关键发现

### 2.1 Async TP 的根本矛盾

```
通信占比最高的场景 (小 M, decode)     ← Async TP 理论收益最大
          ↕  恰恰是
fusion kernel overhead 最大的场景     ← Async TP 实际 regression 最大 (2-3x)

小 M (decode): 通信占 55%, 理论 2.25x, 但 fusion 2.1-2.5x 更慢 → regression
大 M (prefill): 通信占 32%, 理论 1.47x, 但 fusion 仅 1.04-1.15x → 微小收益
```

这是 `symm_mem` pipeline 的固有开销（P2P setup ~50μs + chunk sync + GEMM 碎片化），不是 heuristic 能解决的问题。

### 2.2 SP 在 upstream vLLM 是死代码

```python
# vllm/config/vllm.py (upstream)
IS_DENSE = False  # 硬编码 False，所有优化级别 (O0-O3) SP/AsyncTP 都禁用
# See https://github.com/vllm-project/vllm/issues/25689
```

- 这就是为什么 **没有人发现 piecewise crash** — 没人使用 SP
- 我们修复的 crash 和写的 heuristic **都是针对死代码**

### 2.3 Heuristic 的真正价值：避免 Regression，不是加速

| Inductor 旧规则 (K<1024) | 我们的 AutoHeuristic |
|---|---|
| **等价于 "always fuse"** (K_shard ≥ 1024 for all LLMs) | 正确区分 decode (不 fuse) 和 prefill (fuse) |
| Decode M=1~32: **2-3x regression** | Decode: no regression |
| 未来如果 async TP 被默认启用，**会造成严重生产问题** | 安全启用 async TP 的前提条件 |

**AutoHeuristic 是"安全网"，让 async TP 可以被默认启用而不造成 decode regression。**

---

## 3. Code 状态

### 3.1 PyTorch Stack（bookmark: `async_tp_autoheuristic`）

3 个 commit，独立于当前工作 stack：

| Commit | 内容 | 关键文件 |
|--------|------|---------|
| `0144b4a4a95b` | [inductor] fuse/no_fuse AutoHeuristic | `_AsyncTPFuseH100.py`, `micro_pipeline_tp.py` |
| `270acc256a0e` | [distributed] native vs pipeline AutoHeuristic | `_AsyncTPNativeH100.py`, `symm_mem/__init__.py` |
| `6a7eabbee735` | [inductor] tile config AutoHeuristic | `_AsyncTPTileConfigH100.py`, training pipeline files |

**状态**：③ tile config 已开 PR #177068。①② 在 bookmark 上，尚未提交 diff。

### 3.2 vLLM Draft PR

- **PR**: https://github.com/vllm-project/vllm/pull/37489
- **Branch**: `tianren/sp-asynctp-piecewise-fix` (fork `tianrengao/vllm`)
- **内容**: 5 modified + 5 new files (crash fix + heuristic + config + benchmark 脚本 + 测试)

### 3.3 训练 Pipeline 文件（已就位）

```
torchgen/_autoheuristic/async_tp/
├── convert_benchmark_data.py          # JSON → AutoHeuristic format
├── convert_native_vs_pipeline_data.py
├── convert_tile_config_data.py
├── train_decision_async_tp.py         # ① fuse/no_fuse
├── train_decision_async_tp_native.py  # ② native/pipeline
├── train_decision_async_tp_tile_config.py  # ③ tile config
├── gen_async_tp_heuristic_h100.sh     # 一键生成脚本
└── README.md
```

---

## 4. 与 TL 讨论的要点

### 4.1 能汇报的成果

| 成果 | 价值 |
|------|------|
| **3 层 AutoHeuristic pipeline 全部完成** | 首个非 GEMM 的 AutoHeuristic 应用；验证了 pipeline 可复用性 |
| **95.7% accuracy decision tree** | vs Inductor "always fuse" 的 ~50% → 防止 2-3x decode regression |
| **tile config ③ 修复 correctness bug** | 硬编码 heuristic 对 33% shapes 选择会编译失败的 config |
| **发现并修复了训练数据 bug** | converter 和 runtime 的 local/global M 不一致，导致决策树 split 阈值错误 |
| **Custom Op Autotuning API 验证** | 95.3% oracle match, -2.4% overhead → compile-time autotuning 质量高 |
| **vLLM piecewise compilation crash fix** | SP 在 piecewise compilation 下的 AOT autograd + 跨 submod shape 不匹配 |
| **深度分析了 async TP 不 work 的原因** | fusion kernel overhead at small M，NVLink 下通信成本太低 |

### 4.2 需要讨论的问题

1. **Async TP 的投入产出**：
   - Pipeline 基础设施价值高（可复用），但 E2E serving 收益为零
   - 是否继续投入？跨节点 TP (IB/RoCE) 是唯一可能看到大收益的场景
   - `IS_DENSE=False` blocker — upstream vLLM 团队是否有重新启用 SP 的计划？

2. **PyTorch stack 处理**：
   - ③ tile config 已有 PR #177068，可以推进 review/land
   - ①② fuse/no_fuse + native/pipeline 是否值得提交 diff？（目前功能 OFF by default）
   - 需要等 `_micro_pipeline_tp` 被默认启用后才有实际影响

3. **Pipeline 复用到其他 kernel**：
   - **最高 ROI**: Helion config picker 自动化（替换手写 `pick_xxx_config()`）
   - **中 ROI**: vLLM AsyncTPPass 加 shape-aware heuristic
   - **低 ROI**: CuTe DSL tile config（path 不在 vLLM 主路径）

4. **下一步方向**：
   - 跨节点 TP 测试（通信成本高 10x，可能看到真正收益）
   - Prefill-heavy workload（M=4096+，fusion 1.15x，占比更高）
   - 回到其他 Custom Op Autotuning 应用（Scaled MM、Pad MM H100 等）

### 4.3 Impact 定位建议

**不要 pitch 为 "async TP 加速 serving"** — 数据不支持。

**可以 pitch 为**：
1. **"AutoHeuristic pipeline 首个非 GEMM 应用的端到端验证"** — 证明 benchmark → train → artifact → deploy 的通用性
2. **"防止 async TP 默认启用时的 decode regression"** — Inductor 旧规则 = always fuse，会让 decode 慢 2-3x
3. **"为未来跨节点 TP 做好基础设施准备"** — pipeline ready，等场景

---

## 5. Benchmark 数据汇总

### Kernel-level（Llama-3-70B, TP=8, 8× H100 NVLink）

| M (batch) | Baseline | Fused | Impact |
|-----------|---------|-------|--------|
| 1 (decode) | 241μs | 548μs | **2.3× 更慢** |
| 32 (decode) | 269μs | 554μs | **2.1× 更慢** |
| 512 (prefill) | 586μs | 563μs | 1.04× 更快 |
| 4096 (prefill) | 4024μs | 3507μs | **1.15× 更快** |

### E2E Serving

| 配置 | 吞吐 (tok/s) | SP 效果 |
|------|-------------|---------|
| **Llama-2-7b TP=2** baseline | 8,819 | — |
| Llama-2-7b TP=2 + SP | 8,866 | **1.01x** (噪声) |
| **Llama-3-70B TP=8** baseline | 3,949 | — |
| Llama-3-70B TP=8 + SP | 3,875 | **0.98x** (轻微回退) |

---

---

## 6. TL Feedback 和 Story Narrative

### 6.1 TL Feedback (2026-03-19)

- **M>2048 的 use case 不多**，但 **M<=1024 一定是好方向** -- Layer 3 在 M<=1024 的 correctness fix（硬编码 crash）正好命中这个方向
- 对 CuTe DSL tile config hardcode 的来源有疑问 -- 可能是 MSL/ops 团队在早期只测了大 M，没覆盖小 M 就写死了

### 6.2 Story Narrative -- 如何讲这个项目

核心卖点不是某一个 kernel 快了多少，而是 **Custom Op Autotuning -> Decision Tree 这条 pipeline 本身**。

**Story**:

PyTorch 里存在大量硬编码的 kernel heuristic（形如 `if M*N < threshold: use config A else use config B`）。这些 heuristic 有三个问题：

1. **准确率低** -- 手写规则覆盖不了真实 workload 的 shape 多样性（Layer 1 的 Inductor always-fuse 规则只有 ~50% accuracy）
2. **有 correctness bug** -- 选了不合法的 config 导致编译 crash（Layer 3 的 C2x1 在 M<=1024 crash，影响 33% shapes）
3. **每台机器不同** -- H100 和 A100 的最优 config 不一样，hardcode 一个值在另一台机器上就是 regression

我们的 pipeline 解决这三个问题：

```
Custom Op Autotuning (per-GPU benchmark, run once)
  -> CSV training data (shape features + latency per config)
  -> sklearn Decision Tree training
  -> AutoHeuristic artifact (Python if-else code, zero runtime cost)
  -> Turn on by default, per-GPU artifact, no manual tuning
```

**用 async TP 和 CuTe DSL 作为两个 proof point**:

| Proof Point | 问题 | 我们的结果 |
|-------------|------|-----------|
| Layer 1: async TP fuse/no-fuse | Inductor always-fuse 导致 decode 2-3x regression | 95.7% accuracy，正确区分 decode (no-fuse) 和 prefill (fuse)，防止 regression |
| Layer 3: CuTe DSL tile config | 硬编码 heuristic 在 M<=1024 crash (33% shapes) | 修复 crash + 可比 shapes 上平均快 2.07%，峰值快 11% |

**卖点总结**:

1. **Correctness** -- 修复了硬编码 heuristic 的 crash/regression bug
2. **Per-GPU adaptation** -- 每台机器跑一次 benchmark，自动生成该机器的最优 decision tree，不需要人工 tune
3. **Zero runtime cost** -- Decision tree 是编译期的 if-else，不增加运行时开销
4. **Pipeline reusability** -- 同一套 pipeline (converter, trainer, artifact) 已经在二分类 (Layer 1,2) 和 18 分类 (Layer 3) 上验证，可以直接复用到其他 kernel
5. **Default-on ready** -- 有 unsafe leaf 保护 + confidence threshold，对不确定的 shape fallback 到保守选择，可以安全地默认启用

**适用场景**:
- 任何有 `if shape > threshold: use config A else use config B` 形式的硬编码 heuristic
- 已识别的候选：Pad MM (H100 artifact 缺失)、Scaled MM (纯 compile-time autotuning，无 heuristic)、AllReduce bucketing (固定 25MB)、Helion config picker (手写 nearest-match)
- 每新增一个 GPU 型号，不需要人工重新 tune，跑一次 benchmark 自动生成 artifact

---

## 7. 文档索引

| 文档 | 用途 |
|------|------|
| **本文档** (`TL_ASYNCTP_SUMMARY.md`) | TL 汇报用 |
| `TL_DISCUSSION_CONCISE.md` | 完整技术细节（754 行，含 3 层 pipeline、架构图、所有数据） |
| `SP_ASYNCTP_TRACKING.md` | 13 session 全程追踪（vLLM E2E 验证） |
| `VLLM_CODE_CHANGES_PR_PLAN.md` | vLLM 代码改动清单和 PR 计划 |
| `TL_DISCUSSION_SUMMARY.md` | 早期 POC 结果（benchmark 数据表、decision tree） |
| `E2E_BENCHMARK_REPORT.md` | E2E latency benchmark 报告 (Sessions 1-8) |
