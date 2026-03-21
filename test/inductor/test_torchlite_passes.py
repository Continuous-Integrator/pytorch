"""Unit tests for individual torchlite graph passes.

Each test creates a small model, traces it, runs a single pass, and verifies
the expected graph property or compares output against eager execution.
"""

import operator
import os
import shutil
import tempfile

from test_torchlite_utils import TrainStep, TwoLayerMLP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._torchlite import (
    compile,
    inference_passes,
    precompile_load,
    precompile_save,
    run_passes,
    trace,
)
from torch._torchlite.ops import (
    _load_rng_state,
    _save_rng_state,
    adamw_step,
    packed_silu_mul,
    param_update,
    sgd_step,
)
from torch._torchlite.passes import (
    _aten_op_name,
    _graph_meta,
    _save_for_backward,
    activation_checkpoint,
    attention_canonicalize,
    autograd_per_op,
    canonicalize_layouts,
    canonicalize_pointwise_kwargs,
    common_subexpression_elimination,
    cudagraph_partition,
    decompose,
    decompose_attention_projections,
    decompose_inference,
    decompose_training_backward,
    dynamize,
    expand_gqa_projections,
    extract_attention_regions,
    extract_ffn_regions,
    fuse_packed_silu_mul,
    functionalize,
    fuse,
    FusedKernel,
    matmul_epilogue,
    MatmulEpilogueKernel,
    memory_plan,
    normalize,
    optimizer,
    pack_parallel_linears,
    pack_parallel_matmuls,
    rng_functionalize,
    save_activations,
    sdpa_pattern,
    simplify_views,
    triton_codegen,
    verify_graph,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNormalize(TestCase):
    def test_dunders_normalized(self):
        def f(x, y):
            return x + y

        gm = trace(f, [torch.randn(4), torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIs(node.target, torch.add)

    def test_tensor_method_normalized(self):
        def f(x):
            return x.abs()

        gm = trace(f, [torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIs(node.target, torch.abs)


class TestVerifyGraph(TestCase):
    def test_valid_graph_passes(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        gm = trace(f, [torch.randn(4)])
        result = verify_graph(gm, [])
        self.assertIsNotNone(result.gm)

    def test_detects_non_callable_target(self):
        gm = trace(lambda x: x + 1, [torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.target = "not_callable"
                break
        with self.assertRaises(ValueError):
            verify_graph(gm, [])


class TestFunctionalize(TestCase):
    def test_inplace_add_removed(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        gm = trace(f, [torch.randn(4)])
        gm = functionalize(gm, []).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = getattr(node.target, "__name__", "")
                # No in-place ops should remain in the main body
                # (copy_back nodes are allowed at the end)
                if name.endswith("_") and not name.startswith("__"):
                    if node.target is not torch.Tensor.copy_:
                        self.fail(f"In-place op {name} not removed by functionalize")

    def test_functionalize_preserves_output(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        inp = torch.randn(4)
        expected = f(inp.clone())

        gm = trace(f, [inp.clone()])
        gm = functionalize(gm, []).gm
        actual = gm(inp.clone())
        self.assertEqual(actual, expected)

    def test_multiple_inplace_ops(self):
        def f(x):
            y = x.clone()
            y.mul_(2.0)
            y.add_(3.0)
            return y

        inp = torch.randn(4)
        expected = f(inp.clone())

        gm = trace(f, [inp.clone()])
        gm = functionalize(gm, []).gm
        actual = gm(inp.clone())
        self.assertEqual(actual, expected)


class TestDynamize(TestCase):
    def test_inserts_size_nodes(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = dynamize(gm, [torch.randn(4, 8)]).gm

        has_size = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.Tensor.size:
                has_size = True
                break
        self.assertTrue(has_size)

    def test_dynamic_batch_dim(self):
        def f(x):
            return x * 2

        inp4 = torch.randn(4, 8)
        inp8 = torch.randn(8, 8)

        gm = trace(f, [inp4])
        gm = dynamize(gm, [inp4]).gm

        out4 = gm(inp4)
        self.assertEqual(out4.shape, (4, 8))
        out8 = gm(inp8)
        self.assertEqual(out8.shape, (8, 8))

    def test_no_dynamic_dims_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm_before = str(gm.graph)
        gm = dynamize(gm, [torch.randn(4, 8)], dynamic_dims={}).gm
        # With empty dynamic_dims, no size nodes should be inserted
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.Tensor.size:
                self.fail("Size nodes inserted when dynamic_dims is empty")


class TestPacking(TestCase):
    def test_pack_parallel_linears_packs_qkv(self):
        class QKVProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(8, 8)
                self.k = nn.Linear(8, 4)
                self.v = nn.Linear(8, 4)

            def forward(self, x):
                return self.q(x), self.k(x), self.v(x)

        torch.manual_seed(0)
        model = QKVProj().eval()
        x = torch.randn(2, 3, 8)

        gm = trace(model, [x])
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]
        packed_slices = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.getitem
        ]

        self.assertEqual(len(linear_nodes), 1)
        self.assertEqual(len(packed_slices), 3)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_matmuls_packs_qkv(self):
        class QKVProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Parameter(torch.randn(8, 8))
                self.wk = nn.Parameter(torch.randn(8, 4))
                self.wv = nn.Parameter(torch.randn(8, 4))

            def forward(self, x):
                return x @ self.wq, x @ self.wk, x @ self.wv

        torch.manual_seed(0)
        model = QKVProj().eval()
        x = torch.randn(6, 8)

        gm = trace(model, [x])
        gm = decompose_inference(gm, [x]).gm
        gm = pack_parallel_matmuls(gm, [x], materialize_constants=True).gm

        mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mm.default
        ]
        packed_slices = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.getitem
        ]

        self.assertEqual(len(mm_nodes), 1)
        self.assertEqual(len(packed_slices), 3)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_linears_skips_pointwise_epilogue(self):
        class GatedProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 16, bias=False)
                self.up = nn.Linear(8, 16, bias=False)

            def forward(self, x):
                return torch.sigmoid(self.gate(x)) * self.up(x)

        torch.manual_seed(0)
        model = GatedProj().eval()
        x = torch.randn(4, 8)

        gm = trace(model, [x])
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]

        self.assertEqual(len(linear_nodes), 2)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_linears_packs_ffn_gate_up_region(self):
        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(8)
                self.gate = nn.Linear(8, 16, bias=False)
                self.up = nn.Linear(8, 16, bias=False)
                self.down = nn.Linear(16, 8, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return self.down(F.silu(self.gate(h)) * self.up(h))

        torch.manual_seed(0)
        model = GatedFFN().eval()
        x = torch.randn(2, 4, 8)

        gm = trace(model, [x])
        gm = extract_ffn_regions(gm, [x]).gm
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm

        packed_linears = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch._C._nn.linear
            and n.meta.get("region_role") == "packed_projection"
        ]
        packed_slices = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is operator.getitem
            and n.meta.get("region_kind") == "ffn"
            and n.meta.get("region_role") in {"gate", "up"}
        ]

        self.assertEqual(len(packed_linears), 1)
        self.assertEqual(len(packed_slices), 2)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_matmuls_skips_pointwise_epilogue(self):
        class GatedProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.wg = nn.Parameter(torch.randn(8, 16))
                self.wu = nn.Parameter(torch.randn(8, 16))

            def forward(self, x):
                return torch.sigmoid(x @ self.wg) * (x @ self.wu)

        torch.manual_seed(0)
        model = GatedProj().eval()
        x = torch.randn(4, 8)

        gm = trace(model, [x])
        gm = decompose_inference(gm, [x]).gm
        gm = pack_parallel_matmuls(gm, [x], materialize_constants=True).gm

        mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mm.default
        ]

        self.assertEqual(len(mm_nodes), 2)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_matmuls_packs_ffn_gate_up_region(self):
        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(8)
                self.gate = nn.Linear(8, 16, bias=False)
                self.up = nn.Linear(8, 16, bias=False)
                self.down = nn.Linear(16, 8, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return self.down(F.silu(self.gate(h)) * self.up(h))

        torch.manual_seed(0)
        model = GatedFFN().eval()
        x = torch.randn(2, 4, 8)

        gm = trace(model, [x])
        gm = extract_ffn_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm
        gm = pack_parallel_matmuls(gm, [x], materialize_constants=True).gm

        packed_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.mm.default
            and n.meta.get("region_role") == "packed_projection"
        ]
        packed_slices = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is operator.getitem
            and n.meta.get("region_kind") == "ffn"
            and n.meta.get("region_role") in {"gate", "up"}
        ]

        self.assertEqual(len(packed_mm), 1)
        self.assertEqual(len(packed_slices), 2)
        self.assertEqual(gm(x), model(x))

    def test_pack_parallel_matmuls_packs_equivalent_reshaped_inputs(self):
        class QKVProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Parameter(torch.randn(8, 8))
                self.wk = nn.Parameter(torch.randn(8, 4))
                self.wv = nn.Parameter(torch.randn(8, 4))

            def forward(self, x):
                q = x @ self.wq
                k = x @ self.wk
                v = x @ self.wv
                return q, k, v

        torch.manual_seed(0)
        model = QKVProj().eval()
        x = torch.randn(2, 3, 8)

        gm = trace(model, [x])
        gm = decompose_inference(gm, [x]).gm
        gm = pack_parallel_matmuls(gm, [x], materialize_constants=True).gm

        mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mm.default
        ]
        packed_slices = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.getitem
        ]

        self.assertEqual(len(mm_nodes), 1)
        self.assertEqual(len(packed_slices), 3)
        self.assertEqual(gm(x), model(x))

    def test_fuse_packed_silu_mul_rewrites_packed_ffn_chain(self):
        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(8)
                self.gate = nn.Linear(8, 16)
                self.up = nn.Linear(8, 16)
                self.down = nn.Linear(16, 8)

            def forward(self, x):
                h = self.norm(x)
                return self.down(F.silu(self.gate(h), inplace=False) * self.up(h))

        torch.manual_seed(0)
        model = GatedFFN().eval()
        x = torch.randn(2, 4, 8)

        gm = trace(model, [x])
        gm = extract_ffn_regions(gm, [x]).gm
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm
        gm = decompose_inference(gm, [x]).gm
        gm = canonicalize_pointwise_kwargs(gm, [x]).gm
        gm = simplify_views(gm, [x]).gm
        gm = fuse_packed_silu_mul(gm, [x]).gm

        packed_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is packed_silu_mul
        ]
        raw_ffn = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "ffn"
            and _aten_op_name(n.target) in {"silu", "mul"}
        ]

        self.assertEqual(len(packed_ops), 1)
        self.assertEqual(raw_ffn, [])
        self.assertEqual(gm(x), model(x), atol=1e-5, rtol=1e-5)


class TestLayoutCanonicalize(TestCase):
    def test_removes_redundant_contiguous_after_mm(self):
        def f(x, w):
            return torch.ops.aten.contiguous.default(x @ w)

        x = torch.randn(4, 8)
        w = torch.randn(8, 16)

        gm = trace(f, [x, w])
        gm = decompose(gm, [x, w]).gm
        gm = canonicalize_layouts(gm, [x, w]).gm

        for node in gm.graph.nodes:
            self.assertFalse(
                node.op == "call_function"
                and node.target == torch.ops.aten.contiguous.default
            )

    def test_collapses_stacked_view_chain(self):
        def f(x, w):
            y = (x @ w).view(2, 4, 8)
            return y.reshape(2, 4, 2, 4)

        x = torch.randn(8, 8)
        w = torch.randn(8, 8)

        gm = trace(f, [x, w])
        gm = canonicalize_layouts(gm, [x, w]).gm

        stacked_views = [
            n
            for n in gm.graph.nodes
            if (
                n.op == "call_function"
                and n.target
                in {
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten._unsafe_view.default,
                    torch.Tensor.reshape,
                    torch.Tensor.view,
                }
                and isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "call_function"
                and n.args[0].target == torch.ops.aten._unsafe_view.default
            )
        ]

        self.assertEqual(len(stacked_views), 0)
        self.assertEqual(gm(x, w), f(x, w))

    def test_inference_pipeline_cleans_transformer_view_chains(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import inference_passes

        class SimpleTransformerBlock(nn.Module):
            def __init__(self, dim, n_heads=4):
                super().__init__()
                self.wq = nn.Parameter(torch.randn(dim, dim) * 0.01)
                self.wk = nn.Parameter(torch.randn(dim, dim) * 0.01)
                self.wv = nn.Parameter(torch.randn(dim, dim) * 0.01)
                self.wo = nn.Parameter(torch.randn(dim, dim) * 0.01)
                self.w1 = nn.Parameter(torch.randn(dim, dim * 4) * 0.01)
                self.w2 = nn.Parameter(torch.randn(dim * 4, dim) * 0.01)
                self.n_heads = n_heads

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                head_dim = dim // self.n_heads
                q = (
                    (x @ self.wq)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                k = (
                    (x @ self.wk)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                v = (
                    (x @ self.wv)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                attn = F.scaled_dot_product_attention(q, k, v)
                out = attn.transpose(1, 2).reshape(bsz, seqlen, dim)
                x = x + out @ self.wo
                h = torch.relu(x @ self.w1)
                x = x + h @ self.w2
                return x

        model = SimpleTransformerBlock(256).cuda().eval()
        x = torch.randn(4, 128, 256, device="cuda")

        gm = trace(model, [x])
        gm = run_passes(gm, [x], pipeline=inference_passes(gm, [x]))

        stacked_views = [
            n
            for n in gm.graph.nodes
            if (
                n.op == "call_function"
                and n.target
                in {
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten._unsafe_view.default,
                    torch.Tensor.reshape,
                    torch.Tensor.view,
                }
                and isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "call_function"
                and n.args[0].target == torch.ops.aten._unsafe_view.default
            )
        ]

        with torch.no_grad():
            actual = gm(x)
            expected = model(x)

        self.assertEqual(len(stacked_views), 0)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)


class TestPointwiseCanonicalize(TestCase):
    def test_strips_default_functional_activation_kwargs(self):
        def f(x):
            return F.silu(F.relu(x, inplace=False), inplace=False)

        x = torch.randn(4, 8)

        gm = trace(f, [x])
        gm = canonicalize_pointwise_kwargs(gm, [x]).gm

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if _aten_op_name(node.target) in {"relu", "silu"}:
                self.assertEqual(node.kwargs, {})

        self.assertEqual(gm(x), f(x))


class TestRegionExtract(TestCase):
    def test_extract_attention_regions_marks_sdpa_qkv(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm

        roles = {
            node.name: node.meta.get("region_role")
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.meta.get("region_kind") == "attention"
        }
        sdpa_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is F.scaled_dot_product_attention
        ]

        self.assertEqual(roles["linear_1"], "q")
        self.assertEqual(roles["linear_3"], "k")
        self.assertEqual(roles["linear_5"], "v")
        self.assertEqual(roles["linear_7"], "out_proj")
        self.assertEqual(len(sdpa_nodes), 1)
        self.assertEqual(sdpa_nodes[0].meta.get("region_kind"), "attention")

    def test_extract_attention_regions_marks_raw_mm_attention(self):
        class RawAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Parameter(torch.randn(32, 32))
                self.wk = nn.Parameter(torch.randn(32, 32))
                self.wv = nn.Parameter(torch.randn(32, 32))
                self.wo = nn.Parameter(torch.randn(32, 32))
                self.n_heads = 4

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                head_dim = dim // self.n_heads
                q = (
                    (x @ self.wq)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                k = (
                    (x @ self.wk)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                v = (
                    (x @ self.wv)
                    .reshape(bsz, seqlen, self.n_heads, head_dim)
                    .transpose(1, 2)
                )
                attn = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
                out = torch.softmax(attn, dim=-1) @ v
                out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
                return out @ self.wo

        torch.manual_seed(0)
        model = RawAttention().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm

        attention_roles = {
            node.meta.get("region_role")
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.meta.get("region_kind") == "attention"
        }

        self.assertTrue(
            {"q", "k", "v", "scores", "softmax", "core", "out_proj"} <= attention_roles
        )

    def test_extract_ffn_regions_marks_gate_up_down(self):
        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(32)
                self.gate_proj = nn.Linear(32, 64, bias=False)
                self.up_proj = nn.Linear(32, 64, bias=False)
                self.down_proj = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        torch.manual_seed(0)
        model = GatedFFN().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_ffn_regions(gm, [x]).gm

        roles = {
            node.meta.get("region_role")
            for node in gm.graph.nodes
            if node.op == "call_function" and node.meta.get("region_kind") == "ffn"
        }

        self.assertTrue(
            {"norm", "gate", "up", "activation", "combine", "down"} <= roles
        )


class TestAttentionCanonicalize(TestCase):
    def test_rewrites_repeat_kv_decomposition(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = decompose(gm, [x]).gm
        gm = simplify_views(gm, [x]).gm
        gm = attention_canonicalize(gm, [x]).gm

        repeat_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.repeat_interleave
        ]
        expand_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.expand.default
        ]
        sdpa_nodes = [
            n
            for n in gm.graph.nodes
            if (n.op == "call_function" and n.target is F.scaled_dot_product_attention)
        ]

        self.assertEqual(len(repeat_nodes), 2)
        self.assertEqual(len(expand_nodes), 0)
        self.assertEqual(len(sdpa_nodes), 1)
        self.assertEqual(sdpa_nodes[0].meta.get("attention_kind"), "gqa")
        self.assertEqual(gm(x), model(x))

    def test_expand_gqa_projections_removes_runtime_repeat(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                return F.scaled_dot_product_attention(q, k, v)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = expand_gqa_projections(gm, [x]).gm

        repeat_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.repeat_interleave
        ]
        linear_nodes = [
            n
            for n in gm.graph.nodes
            if (n.op == "call_function" and n.target is torch._C._nn.linear)
        ]

        self.assertEqual(len(repeat_nodes), 0)
        self.assertEqual(len(linear_nodes), 3)
        self.assertEqual(linear_nodes[1].meta.get("shape")[-1], 32)
        self.assertEqual(linear_nodes[2].meta.get("shape")[-1], 32)
        self.assertEqual(gm(x), model(x))

    def test_expand_gqa_projections_enables_qkv_packing(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = expand_gqa_projections(gm, [x]).gm
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm

        repeat_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.repeat_interleave
        ]
        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]

        self.assertEqual(len(repeat_nodes), 0)
        self.assertEqual(len(linear_nodes), 2)
        self.assertEqual(gm(x), model(x))


class TestAutogradPerOp(TestCase):
    def test_backward_nodes_added(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm

        has_backward = False
        for node in gm.graph.nodes:
            if node.meta.get("phase") == "backward":
                has_backward = True
                break
        self.assertTrue(has_backward)

    def test_no_params_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm_orig = str(gm.graph)
        gm = autograd_per_op(gm, [torch.randn(4)]).gm
        # No params means no backward nodes should be added
        for node in gm.graph.nodes:
            self.assertNotEqual(node.meta.get("phase"), "backward")


class TestSaveActivations(TestCase):
    def test_save_for_backward_inserted(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm

        saves = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _save_for_backward
        )
        self.assertGreater(saves, 0)

    def test_no_backward_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm = save_activations(gm, []).gm
        saves = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _save_for_backward
        )
        self.assertEqual(saves, 0)


class TestActivationCheckpoint(TestCase):
    def test_recompute_nodes_added(self):
        class SinCosModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 4) * 0.01)

            def forward(self, x):
                h = torch.sin(x @ self.w)
                return torch.cos(h)

        model = SinCosModel()
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = activation_checkpoint(gm, inp).gm

        has_recompute = False
        for node in gm.graph.nodes:
            if node.meta.get("phase") == "recompute":
                has_recompute = True
                break
        self.assertTrue(has_recompute)


class TestTrainingCleanup(TestCase):
    def test_decompose_training_backward_lowers_threshold_backward(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = decompose_training_backward(gm, inp).gm

        threshold_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.threshold_backward.default
        ]
        gt_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.gt.Scalar
        ]

        self.assertEqual(len(threshold_nodes), 0)
        self.assertGreater(len(gt_nodes), 0)

    def test_cse_dedupes_identical_pure_ops(self):
        def f(x):
            a = torch.sin(x)
            b = torch.sin(x)
            return a + b

        x = torch.randn(2, 8)
        gm = trace(f, [x])

        sin_before = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) == "sin"
        )

        gm = common_subexpression_elimination(gm, [x]).gm

        sin_after = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) == "sin"
        )

        self.assertLess(sin_after, sin_before)
        self.assertEqual(gm(x), f(x))

    def test_cse_and_backward_decompose_expose_training_matmul_epilogues(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import default_passes
        from torch._torchlite.passes.triton import _TritonMatmulModule

        model = TwoLayerMLP(256, 512, 128).cuda()
        train_step = TrainStep(model)
        inp = [
            torch.randn(32, 256, device="cuda"),
            torch.randn(32, 128, device="cuda"),
        ]

        gm = trace(train_step, inp)
        gm = run_passes(gm, inp, pipeline=default_passes(gm, inp))

        threshold_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.threshold_backward.default
        ]
        raw_mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mm.default
        ]
        triton_matmul_mods = [
            mod
            for _, mod in gm.named_modules()
            if isinstance(mod, _TritonMatmulModule)
        ]

        self.assertEqual(len(threshold_nodes), 0)
        self.assertLessEqual(len(raw_mm_nodes), 2)
        self.assertGreaterEqual(len(triton_matmul_mods), 3)

    def test_cuda_training_pipeline_matches_eager_one_step(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        torch.manual_seed(0)
        eager_model = TwoLayerMLP(32, 64, 16).cuda()
        torch.manual_seed(0)
        compiled_model = TwoLayerMLP(32, 64, 16).cuda()

        train_step_eager = TrainStep(eager_model)
        train_step_compiled = TrainStep(compiled_model)
        x = torch.randn(8, 32, device="cuda")
        target = torch.randn(8, 16, device="cuda")

        eager_loss = train_step_eager(x, target)
        eager_loss.backward()
        with torch.no_grad():
            for p in eager_model.parameters():
                p.sub_(0.01 * p.grad)

        compiled_fn = compile(train_step_compiled, [x, target])
        compiled_loss = compiled_fn(x, target)

        self.assertEqual(compiled_loss, eager_loss, atol=1e-3, rtol=1e-3)
        for eager_param, compiled_param in zip(
            eager_model.parameters(), compiled_model.parameters()
        ):
            self.assertEqual(
                compiled_param, eager_param, atol=2e-3, rtol=2e-3
            )


class TestDecompose(TestCase):
    def test_decompose_preserves_output(self):
        def f(x):
            return torch.nn.functional.gelu(x)

        inp = torch.randn(4, 8)
        expected = f(inp)

        gm = trace(f, [inp.clone()])
        gm = decompose(gm, []).gm
        actual = gm(inp)
        self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_decompose_expands_compound_ops(self):
        def f(x):
            return torch.nn.functional.softmax(x, dim=-1)

        gm = trace(f, [torch.randn(4, 8)])
        nodes_before = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        gm = decompose(gm, []).gm
        nodes_after = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        # softmax should decompose into exp, sum, div, etc.
        self.assertGreaterEqual(nodes_after, nodes_before)

    def test_decompose_inference_preserves_marked_regions(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        class RawMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(32, 64))
                self.w2 = nn.Parameter(torch.randn(64, 32))

            def forward(self, x):
                return torch.relu(x @ self.w1) @ self.w2

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm

        attention_sdpa = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "attention"
            and n.target is F.scaled_dot_product_attention
        ]
        attention_linears = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "attention"
            and n.target is torch._C._nn.linear
        ]
        attention_repeats = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "attention"
            and _aten_op_name(n.target) == "repeat_interleave"
        ]

        self.assertEqual(len(attention_sdpa), 1)
        self.assertEqual(len(attention_linears), 4)
        self.assertEqual(len(attention_repeats), 2)
        self.assertEqual(gm(x), model(x))

        mlp = RawMLP().eval()
        gm = trace(mlp, [x])
        gm = extract_ffn_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm

        ffn_relu = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "ffn"
            and _aten_op_name(n.target) == "relu"
        ]
        ffn_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) in {"mm", "addmm"}
        ]

        self.assertEqual(len(ffn_relu), 1)
        self.assertGreaterEqual(len(ffn_mm), 2)
        self.assertEqual(gm(x), mlp(x))

    def test_decompose_inference_preserves_attention_projections(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm
        gm = decompose_inference(gm, [x]).gm

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]
        repeat_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and getattr(n.target, "__name__", "") == "repeat_interleave"
        ]

        self.assertEqual(len(linear_nodes), 2)
        self.assertEqual(len(repeat_nodes), 2)
        self.assertEqual(gm(x), model(x))

    def test_decompose_attention_projections_rewrites_single_qkv_and_out_proj(self):
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 4
                self.head_dim = 8
                self.qkv = nn.Linear(32, 96)
                self.out_proj = nn.Linear(32, 32)

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                qkv = self.qkv(x).reshape(
                    bsz, seqlen, 3, self.n_heads, self.head_dim
                )
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, dim)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = SimpleAttention().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm
        gm = attention_canonicalize(gm, [x]).gm
        gm = sdpa_pattern(gm, [x]).gm
        gm = decompose_attention_projections(gm, [x]).gm

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]
        mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) in {"mm", "addmm"}
        ]

        self.assertEqual(len(linear_nodes), 0)
        self.assertEqual(len(mm_nodes), 2)
        self.assertEqual(gm(x), model(x))

    def test_decompose_attention_projections_rewrites_packed_gqa_projections(self):
        class GQAAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 16)
                self.v_proj = nn.Linear(32, 16)
                self.out_proj = nn.Linear(32, 32)
                self.n_heads = 4
                self.n_kv_heads = 2
                self.head_dim = 8

            def forward(self, x):
                bsz, seqlen, _ = x.shape
                q = (
                    self.q_proj(x)
                    .view(bsz, seqlen, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = GQAAttn().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm
        gm = pack_parallel_linears(gm, [x], materialize_constants=True).gm
        gm = decompose_inference(gm, [x]).gm
        gm = attention_canonicalize(gm, [x]).gm
        gm = sdpa_pattern(gm, [x]).gm
        gm = decompose_attention_projections(gm, [x]).gm

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]
        attention_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and _aten_op_name(n.target) in {"mm", "addmm"}
            and n.meta.get("region_kind") == "attention"
        ]

        self.assertEqual(len(linear_nodes), 0)
        self.assertEqual(len(attention_mm), 2)
        self.assertEqual(gm(x), model(x))

    def test_inference_pipeline_decomposes_attention_projections(self):
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 4
                self.head_dim = 8
                self.qkv = nn.Linear(32, 96)
                self.out_proj = nn.Linear(32, 32)

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                qkv = self.qkv(x).reshape(
                    bsz, seqlen, 3, self.n_heads, self.head_dim
                )
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, dim)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = SimpleAttention().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = run_passes(gm, [x], pipeline=inference_passes(gm, [x]))

        linear_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch._C._nn.linear
        ]
        mm_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) in {"mm", "addmm"}
        ]

        self.assertEqual(len(linear_nodes), 0)
        self.assertEqual(len(mm_nodes), 2)
        self.assertEqual(gm(x), model(x))

    def test_decompose_attention_projections_preserves_residual_out_proj(self):
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 4
                self.head_dim = 8
                self.qkv = nn.Linear(32, 96)
                self.out_proj = nn.Linear(32, 32)

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                qkv = self.qkv(x).reshape(
                    bsz, seqlen, 3, self.n_heads, self.head_dim
                )
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, dim)
                return self.out_proj(attn)

        class ResidualAttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(32)
                self.attn = SimpleAttention()

            def forward(self, x):
                return x + self.attn(self.norm(x))

        torch.manual_seed(0)
        model = ResidualAttentionBlock().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm
        gm = attention_canonicalize(gm, [x]).gm
        gm = sdpa_pattern(gm, [x]).gm
        gm = decompose_attention_projections(gm, [x]).gm

        qkv_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and _aten_op_name(n.target) in {"mm", "addmm"}
            and n.meta.get("region_kind") != "attention"
        ]
        out_proj_linear = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch._C._nn.linear
            and n.meta.get("region_kind") == "attention"
            and n.meta.get("region_role") == "out_proj"
        ]

        self.assertEqual(len(qkv_mm), 1)
        self.assertEqual(len(out_proj_linear), 1)
        self.assertEqual(gm(x), model(x))

    def test_memory_plan_skips_unbind_view_aliases(self):
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 8
                self.head_dim = 16
                self.qkv = nn.Linear(128, 384)
                self.out_proj = nn.Linear(128, 128)

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                qkv = self.qkv(x).reshape(
                    bsz, seqlen, 3, self.n_heads, self.head_dim
                )
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).reshape(bsz, seqlen, dim)
                return self.out_proj(attn)

        torch.manual_seed(0)
        model = SimpleAttention().eval()
        x = torch.randn(2, 16, 128)

        gm = trace(model, [x])
        gm = extract_attention_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm
        gm = attention_canonicalize(gm, [x]).gm
        gm = sdpa_pattern(gm, [x]).gm
        gm = decompose_attention_projections(gm, [x]).gm
        gm = memory_plan(gm, [x]).gm

        qkv_addmm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and _aten_op_name(n.target) == "addmm"
            and list(n.meta.get("shape", [])) == [32, 384]
        ]
        qkv_items = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is operator.getitem
            and isinstance(n.args[0], torch.fx.Node)
            and _aten_op_name(n.args[0].target) == "unbind"
        ]

        self.assertEqual(len(qkv_addmm), 1)
        self.assertTrue(qkv_addmm[0].meta.get("disable_memory_pool"))
        self.assertIsNone(qkv_addmm[0].meta.get("memory_pool"))
        self.assertEqual(len(qkv_items), 3)
        self.assertTrue(all(n.meta.get("memory_pool") is None for n in qkv_items))

    def test_decompose_inference_decomposes_ffn_projections(self):
        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(32)
                self.gate_proj = nn.Linear(32, 64, bias=False)
                self.up_proj = nn.Linear(32, 64, bias=False)
                self.down_proj = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        torch.manual_seed(0)
        model = GatedFFN().eval()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = extract_ffn_regions(gm, [x]).gm
        gm = decompose_inference(gm, [x]).gm

        ffn_linears = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch._C._nn.linear
            and n.meta.get("region_kind") == "ffn"
        ]
        ffn_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and _aten_op_name(n.target) in {"mm", "addmm"}
            and n.meta.get("region_kind") == "ffn"
        ]
        ffn_pointwise = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "ffn"
            and _aten_op_name(n.target) in {"silu", "mul"}
        ]

        self.assertEqual(len(ffn_linears), 0)
        self.assertGreaterEqual(len(ffn_mm), 3)
        self.assertEqual(len(ffn_pointwise), 2)
        self.assertEqual(gm(x), model(x))


class TestFuse(TestCase):
    def test_pointwise_ops_fuse(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].target.n_inputs, 1)

    def test_non_pointwise_not_fused(self):
        def f(x, y):
            return torch.matmul(x, y)

        gm = trace(f, [torch.randn(4, 8), torch.randn(8, 4)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 0)

    def test_fused_kernel_has_correct_ops(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            return b

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        for n in gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.target, FusedKernel):
                op_names = [op.op_name for op in n.target.ops]
                self.assertIn("sin", op_names)
                self.assertIn("cos", op_names)
                break

    def test_fused_kernel_input_shapes_stored(self):
        # Verify that FusedKernel records input_shapes so that
        # _generate_kernel_source can emit broadcast-aware index expressions.
        def f(x, w, s):
            # x: [M, N], w: [N], s: [M, 1]  -- same broadcast pattern as RMSNorm
            return x * w * s

        M, N = 4, 8
        x = torch.randn(M, N)
        w = torch.randn(N)
        s = torch.randn(M, 1)
        gm = trace(f, [x, w, s])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 1)
        kernel = fused[0].target
        self.assertIsNotNone(kernel.input_shapes)
        self.assertEqual(len(kernel.input_shapes), kernel.n_inputs)

    @torch.no_grad()
    def test_broadcast_fused_kernel_correct_output(self):
        # The fused kernel must correctly handle inputs that broadcast over
        # the output shape — specifically [N] (row) and [M, 1] (column)
        # vectors multiplied with a 2D tensor [M, N].  This is the same
        # pattern as RMSNorm: out[i,j] = x[i,j] * rsqrt[i] * weight[j].
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import codegen
        from torch._torchlite.passes.triton import triton_lower

        def rms_scale(x, weight, rsqrt):
            return x * rsqrt * weight

        M, N = 128, 64
        x = torch.randn(M, N, device="cuda")
        weight = torch.randn(N, device="cuda")
        rsqrt = torch.randn(M, 1, device="cuda")

        gm = trace(rms_scale, [x, weight, rsqrt])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = memory_plan(gm, []).gm
        gm = triton_lower(gm, []).gm
        fn = codegen(gm, inference_codegen=True, example_inputs=[x, weight, rsqrt])

        expected = rms_scale(x, weight, rsqrt)
        actual1 = fn(x, weight, rsqrt)
        actual2 = fn(x, weight, rsqrt)

        self.assertEqual(actual1, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(actual2, expected, atol=1e-4, rtol=1e-4)

    @torch.no_grad()
    def test_broadcast_fused_kernel_correct_output_3d(self):
        # 3D output [B, S, D]: the column-vector [B, S, 1] and the row-vector
        # [D] must use offs // D and offs % D respectively, not plain offs.
        # Before the fix, n_out == 2 guard prevented the broadcast paths from
        # firing, causing the kernel to read garbage memory and produce NaN.
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import codegen
        from torch._torchlite.passes.triton import triton_lower

        def rms_scale_3d(x, weight, rsqrt):
            return x * rsqrt * weight

        B, S, D = 2, 16, 32
        x = torch.randn(B, S, D, device="cuda")
        weight = torch.randn(D, device="cuda")
        rsqrt = torch.randn(B, S, 1, device="cuda")

        gm = trace(rms_scale_3d, [x, weight, rsqrt])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = memory_plan(gm, []).gm
        gm = triton_lower(gm, []).gm
        fn = codegen(gm, inference_codegen=True, example_inputs=[x, weight, rsqrt])

        expected = rms_scale_3d(x, weight, rsqrt)
        actual = fn(x, weight, rsqrt)
        self.assertFalse(actual.isnan().any(), "output should not contain NaN")
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)

    @torch.no_grad()
    def test_pow_scalar_fused_kernel_correct_output(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import codegen
        from torch._torchlite.passes.triton import triton_lower

        def f(x):
            return torch.pow(x, 1.0) * torch.pow(x + 1.0, 2.0)

        x = torch.randn(128, 64, device="cuda")

        gm = trace(f, [x])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertTrue(
            any(op.op_name == "pow" for n in fused for op in n.target.ops),
            "expected fused pointwise kernel to include pow",
        )

        gm = memory_plan(gm, []).gm
        gm = triton_lower(gm, []).gm
        fn = codegen(gm, inference_codegen=True, example_inputs=[x])

        expected = f(x)
        actual1 = fn(x)
        actual2 = fn(x)

        self.assertEqual(actual1, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(actual2, expected, atol=1e-4, rtol=1e-4)


class TestMatmulEpilogue(TestCase):
    def test_matmul_epilogue_fuses_functional_relu_chain(self):
        class DeepNarrow(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = F.relu(layer(x), inplace=False)
                return x

        x = torch.randn(8, 32)
        gm = trace(DeepNarrow(), [x])
        gm = decompose_inference(gm, [x]).gm
        gm = canonicalize_pointwise_kwargs(gm, [x]).gm
        gm = matmul_epilogue(gm, [x]).gm

        raw_addmm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.addmm.default
        ]
        raw_relu = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) == "relu"
        ]
        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, MatmulEpilogueKernel)
        ]

        self.assertEqual(len(raw_addmm), 0)
        self.assertEqual(len(raw_relu), 0)
        self.assertEqual(len(fused), 4)

    def test_matmul_epilogue_fuses_functional_silu_residual_stack(self):
        class SiLUResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return x + self.fc2(F.silu(self.fc1(x), inplace=False))

        class SiLUResStack(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([SiLUResBlock() for _ in range(3)])

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        x = torch.randn(8, 32)
        gm = trace(SiLUResStack(), [x])
        gm = decompose_inference(gm, [x]).gm
        gm = canonicalize_pointwise_kwargs(gm, [x]).gm
        gm = matmul_epilogue(gm, [x]).gm

        raw_addmm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.addmm.default
        ]
        raw_silu = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and _aten_op_name(n.target) == "silu"
        ]
        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, MatmulEpilogueKernel)
        ]

        self.assertEqual(len(raw_addmm), 0)
        self.assertEqual(len(raw_silu), 0)
        self.assertEqual(len(fused), 6)

    def test_matmul_epilogue_fuses_3d_batch(self):
        # With 3D inputs, PyTorch decomposes linear as:
        #   view([B*S, K]) -> mm([B*S, N]) -> _unsafe_view([B, S, N]) -> epilogue
        # Before the fix, matmul_epilogue stopped at _unsafe_view (not pointwise)
        # and a pointwise epilogue like add or mul was left unfused.
        # Use relu as the epilogue (not a residual add, which is deliberately
        # excluded from matmul epilogue fusion to allow cuBLAS + fused pointwise).
        class Linear3DRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 32, bias=False)

            def forward(self, x):
                return torch.relu(self.proj(x))

        model = Linear3DRelu()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = decompose(gm, [x]).gm
        gm = normalize(gm, [x]).gm
        gm = matmul_epilogue(gm, [x]).gm

        fused = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, MatmulEpilogueKernel)
        ]
        self.assertGreater(len(fused), 0, "expected at least one MatmulEpilogueKernel")
        kernel = fused[0].target
        self.assertIsNotNone(kernel.out_shape)
        self.assertEqual(kernel.out_shape, [2, 8, 32])

    @torch.no_grad()
    def test_matmul_epilogue_3d_residual_add_first_call_correct(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import inference_passes

        class ResidualLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 32, bias=False)

            def forward(self, x):
                return x + self.proj(x)

        torch.manual_seed(0)
        model = ResidualLinear().cuda().eval()
        x = torch.randn(2, 8, 32, device="cuda")

        expected = model(x)
        gm = trace(model, [x])
        gm = run_passes(gm, [x], pipeline=inference_passes(gm, [x]))

        actual1 = gm(x)
        actual2 = gm(x)
        triton_mods = [
            mod for _, mod in gm.named_modules() if hasattr(mod, "_use_cublas")
        ]

        self.assertEqual(len(triton_mods), 1)
        self.assertTrue(triton_mods[0]._use_cublas)
        self.assertEqual(actual1, expected, atol=1e-3, rtol=1e-3)
        self.assertEqual(actual2, expected, atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_matmul_epilogue_3d_correct_output(self):
        # End-to-end correctness: fused matmul+silu+mul on 3D [B,S,D] input
        # must match eager output without NaN.
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.api import inference_passes

        class LlamaFFN(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.norm = nn.RMSNorm(d)
                self.gate_proj = nn.Linear(d, h, bias=False)
                self.up_proj = nn.Linear(d, h, bias=False)
                self.down_proj = nn.Linear(h, d, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return x + self.down_proj(
                    torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
                )

        torch.manual_seed(0)
        model = LlamaFFN(64, 128).cuda().eval()
        x = torch.randn(2, 16, 64, device="cuda")

        expected = model(x)
        gm = trace(model, [x])
        pipeline = inference_passes(gm, [x])
        gm = run_passes(gm, [x], pipeline=pipeline)

        ffn_linears = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch._C._nn.linear
            and n.meta.get("region_kind") == "ffn"
        ]
        packed_ffn_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and _aten_op_name(n.target) == "mm"
            and n.meta.get("region_role") == "packed_projection"
        ]

        actual = gm(x)

        self.assertEqual(len(ffn_linears), 0)
        self.assertEqual(len(packed_ffn_mm), 1)
        self.assertFalse(actual.isnan().any(), "output should not contain NaN")
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)

    def test_inference_pipeline_fuses_transformer_ffn_functional_silu(self):
        class TransformerBlock(nn.Module):
            def __init__(self, d, n_heads, ffn_mult=4):
                super().__init__()
                self.norm1 = nn.RMSNorm(d)
                self.attn = nn.Linear(d, 3 * d)
                self.out = nn.Linear(d, d)
                self.norm2 = nn.RMSNorm(d)
                self.gate = nn.Linear(d, d * ffn_mult)
                self.up = nn.Linear(d, d * ffn_mult)
                self.down = nn.Linear(d * ffn_mult, d)
                self.n_heads = n_heads
                self.head_dim = d // n_heads

            def forward(self, x):
                bsz, seqlen, dim = x.shape
                qkv = self.attn(self.norm1(x)).reshape(
                    bsz, seqlen, 3, self.n_heads, self.head_dim
                )
                q, k, v = qkv.unbind(2)
                attn = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                )
                x = x + self.out(attn.transpose(1, 2).reshape(bsz, seqlen, dim))
                h = self.norm2(x)
                x = x + self.down(F.silu(self.gate(h), inplace=False) * self.up(h))
                return x

        x = torch.randn(2, 8, 64)
        gm = trace(TransformerBlock(64, 4), [x])
        gm = run_passes(gm, [x], pipeline=inference_passes(gm, [x]))

        raw_ffn = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("region_kind") == "ffn"
            and (
                (
                    n.target == torch.ops.aten.addmm.default
                    and n.meta.get("region_role") != "packed_projection"
                )
                or _aten_op_name(n.target) in {"silu", "mul"}
            )
        ]
        fused_ffn = [
            n
            for n in gm.graph.nodes
            if n.op == "call_module"
            and hasattr(gm.get_submodule(n.target), "epilogue_ops")
        ]
        packed_ffn = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is packed_silu_mul
        ]

        self.assertEqual(raw_ffn, [])
        self.assertEqual(len(packed_ffn), 1)
        self.assertGreaterEqual(len(fused_ffn), 1)


class TestMemoryPlan(TestCase):
    def test_pool_assignment(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            c = torch.exp(b)
            return c

        gm = trace(f, [torch.randn(4, 8)])
        gm = memory_plan(gm, []).gm

        pools = set()
        for node in gm.graph.nodes:
            pool = node.meta.get("memory_pool")
            if pool is not None:
                pools.add(pool)
        self.assertGreater(len(pools), 0)

    def test_stats_computed(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            return b

        gm = trace(f, [torch.randn(4, 8)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]

        self.assertIn("naive_alloc", stats)
        self.assertIn("planned_alloc", stats)
        self.assertIn("num_tensors", stats)
        self.assertIn("num_pools", stats)
        self.assertGreater(stats["num_tensors"], 0)

    def test_empty_graph_stats(self):
        gm = trace(lambda x: x, [torch.randn(4)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]
        self.assertEqual(stats["naive_alloc"], 0)
        self.assertEqual(stats["planned_alloc"], 0)


class TestFullPipeline(TestCase):
    def test_inference_trace_and_passes(self):
        model = TwoLayerMLP(16, 32, 8)
        inp = torch.randn(4, 16)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        actual = gm(inp)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)

    def test_training_sgd_correctness(self):
        torch.manual_seed(42)
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        x = torch.randn(2, 8)
        target = torch.randn(2, 4)

        expected = train_step(x.clone(), target.clone())

        torch.manual_seed(42)
        model2 = TwoLayerMLP(8, 16, 4)
        train_step2 = TrainStep(model2)
        compiled = compile(train_step2, [x.clone(), target.clone()])
        actual = compiled(x.clone(), target.clone())

        self.assertAlmostEqual(actual.item(), expected.item(), places=4)

    def test_pass_ordering_training(self):
        from torch._torchlite import default_passes

        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]
        gm = trace(train_step, [x.clone() for x in inp])

        passes = default_passes(gm, inp)
        self.assertGreater(len(passes), 0)

        for p in passes:
            result = p(gm, inp)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.gm)
            gm = result.gm


class TestDynamicShapes(TestCase):
    def _inference_pipeline(self):
        return [verify_graph, functionalize, dynamize, memory_plan]

    def test_inference_different_batch_size(self):
        model = TwoLayerMLP(16, 32, 8)
        trace_inp = [torch.randn(4, 16)]
        gm = trace(model, trace_inp)
        gm = run_passes(
            gm,
            trace_inp,
            pipeline=self._inference_pipeline(),
            dynamic_dims={"x_0": [0]},
        )

        for batch in [1, 4, 8, 16]:
            inp = torch.randn(batch, 16)
            expected = model(inp)
            actual = gm(inp)
            self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_compile_inference_different_batch_size(self):
        def f(x):
            return torch.sin(x) * torch.cos(x) + x

        trace_inp = [torch.randn(4, 8)]
        gm = trace(f, trace_inp)
        gm = run_passes(
            gm,
            trace_inp,
            pipeline=self._inference_pipeline(),
        )

        for batch in [1, 4, 8, 16]:
            inp = torch.randn(batch, 8)
            expected = f(inp)
            actual = gm(inp)
            self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_training_same_batch_size(self):
        torch.manual_seed(0)
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        trace_x = torch.randn(4, 8)
        trace_target = torch.randn(4, 4)

        compiled = compile(train_step, [trace_x, trace_target])

        x = torch.randn(4, 8)
        target = torch.randn(4, 4)
        loss = compiled(x, target)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_dynamic_dims_explicit_none_defaults_to_batch(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        trace_inp = [torch.randn(4, 8)]
        gm = trace(f, trace_inp)
        gm = run_passes(
            gm,
            trace_inp,
            pipeline=self._inference_pipeline(),
        )

        out8 = gm(torch.randn(8, 8))
        self.assertEqual(out8.shape, (8, 8))

    def test_empty_dynamic_dims_static(self):
        def f(x):
            return torch.sin(x)

        inp4 = torch.randn(4, 8)
        gm = trace(f, [inp4])
        gm = run_passes(
            gm,
            [inp4],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        out = gm(inp4)
        self.assertEqual(out.shape, (4, 8))
        expected = torch.sin(inp4)
        self.assertEqual(out, expected, atol=1e-6, rtol=1e-6)


class TestRngFunctionalize(TestCase):
    def test_noop_without_dropout(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = rng_functionalize(gm, []).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIsNot(node.target, _save_rng_state)
                self.assertIsNot(node.target, _load_rng_state)

    def test_noop_without_rng_replay_metadata(self):
        """Dropout in forward but no rng_replay_for metadata on backward
        nodes means dispatcher-based autograd handles the mask internally.
        rng_functionalize should skip inserting save/load nodes."""

        class DropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 4) * 0.01)

            def forward(self, x):
                return torch.nn.functional.dropout(x @ self.w, p=0.5)

        model = DropoutModel()
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm

        gm = rng_functionalize(gm, inp).gm

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertIsNot(node.target, _save_rng_state)
            self.assertIsNot(node.target, _load_rng_state)

    def test_inserts_state_nodes_when_rng_replay_present(self):
        """When backward nodes have rng_replay_for metadata pointing at
        forward dropout nodes, rng_functionalize should insert save_rng_state
        before each forward dropout and load_rng_state before each backward
        replay node."""

        def f(x):
            return torch.nn.functional.dropout(x, p=0.5)

        gm = trace(f, [torch.randn(4, 8)])

        dropout_node = None
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = getattr(node.target, "__name__", "")
                if name == "dropout":
                    node.meta["phase"] = "forward"
                    dropout_node = node
                    break

        if dropout_node is None:
            self.skipTest("No dropout node found in traced graph")

        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node

        gm.graph.inserting_before(output_node)
        fake_bwd = gm.graph.call_function(torch.neg, (dropout_node,))
        fake_bwd.meta["phase"] = "backward"
        fake_bwd.meta["rng_replay_for"] = dropout_node

        gm.graph.lint()
        gm.recompile()

        gm = rng_functionalize(gm, []).gm

        has_save = any(
            n.op == "call_function" and n.target is _save_rng_state
            for n in gm.graph.nodes
        )
        has_load = any(
            n.op == "call_function" and n.target is _load_rng_state
            for n in gm.graph.nodes
        )
        self.assertTrue(has_save)
        self.assertTrue(has_load)


class TestOptimizer(TestCase):
    def test_sgd_inserts_sgd_step(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = optimizer(gm, inp, lr=0.01, optimizer_type="sgd").gm

        update_count = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is sgd_step
        )
        self.assertGreater(update_count, 0)
        optimizer_mul_sub = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.meta.get("phase") == "optimizer"
            and _aten_op_name(n.target) in {"mul", "sub"}
        ]
        self.assertEqual(optimizer_mul_sub, [])

        has_optimizer_phase = any(
            n.meta.get("phase") == "optimizer" for n in gm.graph.nodes
        )
        self.assertTrue(has_optimizer_phase)

    def test_adamw_inserts_adamw_step(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = optimizer(gm, inp, lr=0.001, optimizer_type="adamw").gm

        adam_count = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is adamw_step
        )
        self.assertGreater(adam_count, 0)

    def test_noop_without_params(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm = optimizer(gm, [], lr=0.01).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIsNot(node.target, param_update)
                self.assertIsNot(node.target, sgd_step)
                self.assertIsNot(node.target, adamw_step)


class TestCudagraphPartition(TestCase):
    def test_all_capturable_single_segment(self):
        def f(x):
            return torch.sin(torch.cos(x)) + x

        gm = trace(f, [torch.randn(4, 8)])
        cudagraph_partition(gm, [])

        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0]["num_nodes"], 0)

    def test_segment_boundary_on_rng_ops(self):
        """Nodes targeting _save_rng_state or _load_rng_state are
        non-capturable and should create segment boundaries."""

        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])

        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node

        gm.graph.inserting_before(output_node)
        rng_node = gm.graph.call_function(_save_rng_state, ())
        rng_node.name = gm.graph._graph_namespace.create_name("rng_state", None)

        sin_node = None
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "__name__", "") == "sin"
            ):
                sin_node = node
                break

        gm.graph.inserting_before(output_node)
        post_node = gm.graph.call_function(torch.cos, (sin_node,))
        post_node.meta["shape"] = sin_node.meta.get("shape", [4])
        post_node.meta["dtype"] = torch.float32

        gm.graph.lint()
        gm.recompile()

        cudagraph_partition(gm, [])
        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertGreaterEqual(len(segments), 2)

    def test_rejects_dynamic_dims(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = dynamize(gm, [torch.randn(4, 8)]).gm

        has_dynamic = any(n.meta.get("dynamic_dims") for n in gm.graph.nodes)
        if has_dynamic:
            with self.assertRaises(RuntimeError):
                cudagraph_partition(gm, [])

    def test_noop_empty_graph(self):
        gm = trace(lambda x: x, [torch.randn(4)])
        cudagraph_partition(gm, [])
        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertEqual(len(segments), 0)


class TestTritonCodegen(TestCase):
    def test_generates_triton_code_for_fused_kernels(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("@triton.jit", triton_code)
        self.assertIn("tl.load", triton_code)
        self.assertIn("tl.store", triton_code)

    def test_triton_code_contains_kernel_name(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        kernel_name = None
        for n in gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.target, FusedKernel):
                kernel_name = n.target.name
                break

        self.assertIsNotNone(kernel_name)

        gm = triton_codegen(gm, []).gm
        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn(kernel_name, triton_code)

    def test_noop_without_fused_kernels(self):
        def f(x, y):
            return torch.matmul(x, y)

        gm = trace(f, [torch.randn(4, 8), torch.randn(8, 4)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("No fused kernels found", triton_code)

    def test_triton_code_has_sin_cos_ops(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("sin", triton_code)
        self.assertIn("cos", triton_code)


class MatmulOnly(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)

    def forward(self, x):
        return x @ self.w


class MatmulChain(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        return (x @ self.w1) @ self.w2


class TestPrecompileSaveLoad(TestCase):
    def _inference_pipeline(self):
        return [verify_graph, functionalize, dynamize, memory_plan]

    def test_matmul_round_trip(self):
        torch.manual_seed(42)
        model = MatmulOnly(16, 8)
        inp = torch.randn(4, 16)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm,
            [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)
            loaded = precompile_load(artifact_dir, trust_remote_code=True)
            actual = loaded.forward(inp)
            self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_matmul_chain_round_trip(self):
        torch.manual_seed(42)
        model = MatmulChain(8)
        inp = torch.randn(4, 8)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm,
            [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)
            loaded = precompile_load(artifact_dir, trust_remote_code=True)
            actual = loaded.forward(inp)
            self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_artifact_directory_structure(self):
        model = MatmulOnly(8, 4)
        inp = torch.randn(2, 8)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm,
            [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)

            self.assertTrue(
                os.path.exists(os.path.join(artifact_dir, "compiled_module.py"))
            )
            self.assertTrue(os.path.exists(os.path.join(artifact_dir, "state_dict.pt")))
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_trust_remote_code_required(self):
        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            with self.assertRaises(ValueError):
                precompile_load(artifact_dir)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_state_dict_preserved(self):
        torch.manual_seed(42)
        model = MatmulChain(8)
        inp = torch.randn(2, 8)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm,
            [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)

            saved_sd = torch.load(
                os.path.join(artifact_dir, "state_dict.pt"),
                weights_only=True,
            )
            for key in model.state_dict():
                self.assertIn(key, saved_sd)
                self.assertEqual(model.state_dict()[key], saved_sd[key])
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)


class TestDynamizeAmbiguousBatch(TestCase):
    def test_dynamize_ambiguous_batch_dim(self):
        """When input is [4, 4] and reshape target is (4, 4), only the
        batch dimension (dim 0) should become dynamic, not dim 1 which
        happens to have the same concrete value."""

        def f(x):
            return x.reshape(4, 4)

        inp = torch.randn(4, 4)
        gm = trace(f, [inp])
        gm = dynamize(gm, [inp], dynamic_dims={"x_0": [0]}).gm

        out8 = gm(torch.randn(8, 4))
        self.assertEqual(out8.shape, (8, 4))


class TestCollectiveFallback(TestCase):
    def test_multi_rank_error_propagates(self):
        """When a real collective raises RuntimeError, it should
        propagate through _try_real_collective rather than being
        silently swallowed."""
        from unittest.mock import MagicMock, patch

        from torch._torchlite.collectives import _try_real_collective

        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("torch.distributed not available")

        mock_funcol = MagicMock()
        mock_funcol.all_reduce.side_effect = RuntimeError("test error")

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.functional_collectives", mock_funcol, create=True),
            patch.dict(
                "sys.modules", {"torch.distributed.functional_collectives": mock_funcol}
            ),
        ):
            with self.assertRaises(RuntimeError):
                _try_real_collective("allreduce", torch.randn(4))


class TestPhaseSafeFusion(TestCase):
    def _get_training_graph(self):
        model = TwoLayerMLP(10, 20, 5)
        train_step = TrainStep(model)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        gm = trace(train_step, [x, target])
        gm = functionalize(gm, [x, target]).gm
        gm = normalize(gm, [x, target]).gm
        gm = autograd_per_op(gm, [x, target]).gm
        gm = rng_functionalize(gm, [x, target]).gm
        gm = save_activations(gm, [x, target]).gm
        gm = activation_checkpoint(gm, [x, target]).gm
        gm = decompose(gm, [x, target]).gm
        return gm, [x, target]

    def test_no_cross_phase_fusion(self):
        gm, inputs = self._get_training_graph()
        gm = fuse(gm, inputs).gm
        for node in gm.graph.nodes:
            if node.op == "call_function" and isinstance(node.target, FusedKernel):
                fused_phase = node.meta.get("phase", "forward")
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        arg_phase = arg.meta.get("phase", "forward")
                        if arg_phase != fused_phase:
                            self.assertTrue(
                                arg.op in ("get_attr", "placeholder")
                                or not isinstance(arg.target, FusedKernel),
                                f"Cross-phase fusion detected: "
                                f"fused node {node.name} (phase={fused_phase}) "
                                f"has arg {arg.name} (phase={arg_phase})",
                            )

    def test_backward_has_pointwise_ops(self):
        gm, inputs = self._get_training_graph()
        from torch._torchlite.passes.common import _aten_op_name

        backward_ops = set()
        for node in gm.graph.nodes:
            if node.meta.get("phase") == "backward" and node.op == "call_function":
                backward_ops.add(_aten_op_name(node.target))
        self.assertIn("mul", backward_ops)
        self.assertIn("div", backward_ops)

    def test_backward_pointwise_chains_fuse(self):
        gm, inputs = self._get_training_graph()
        gm = fuse(gm, inputs).gm
        backward_fused = [
            node
            for node in gm.graph.nodes
            if (
                node.op == "call_function"
                and isinstance(node.target, FusedKernel)
                and node.meta.get("phase") == "backward"
            )
        ]
        self.assertGreater(
            len(backward_fused),
            0,
            "Expected at least one fused kernel in backward phase",
        )


if __name__ == "__main__":
    run_tests()
