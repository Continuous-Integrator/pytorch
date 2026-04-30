# Owner(s): ["module: higher-order-ops"]

"""Tests for validate_subgraph_args_types handling of FX Proxy objects.

When a GraphModule containing higher-order ops (e.g. flex_attention) is
serialized via pickle, deserialization reconstructs the graph by retracing
the forward code through KeepModules().trace(). During this retrace, tensor
arguments become torch.fx.Proxy objects. The HOP's __call__ method runs
validate_subgraph_args_types *before* dispatching to the tracer, so Proxy
objects must be accepted — otherwise deserialization crashes.

This is the exact code path hit when torchtitan's graph_trainer serializes
compiled FlexAttention artifacts via GraphPickler → AOTCompiledArtifact →
pickle.dumps/loads → reduce_graph_module → KeepModules().trace().
"""

import pickle
import unittest

import torch
import torch.fx
from torch._higher_order_ops.utils import validate_subgraph_args_types


class TestValidateSubgraphArgsTypes(unittest.TestCase):
    def test_accepts_tensor(self):
        validate_subgraph_args_types((torch.tensor(1.0),))

    def test_accepts_int(self):
        validate_subgraph_args_types((42,))

    def test_accepts_symint(self):
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        sym = shape_env.create_symintnode(
            shape_env.create_symbol(8, source=ConstantSource("s0"), positive=True),
            hint=8,
        )
        validate_subgraph_args_types((sym,))

    def test_accepts_mixed(self):
        validate_subgraph_args_types((torch.tensor(1.0), 42))

    def test_accepts_empty(self):
        validate_subgraph_args_types(())

    def test_rejects_string(self):
        with self.assertRaises(AssertionError):
            validate_subgraph_args_types(("bad",))

    def test_rejects_float(self):
        with self.assertRaises(AssertionError):
            validate_subgraph_args_types((3.14,))

    def test_rejects_none(self):
        with self.assertRaises(AssertionError):
            validate_subgraph_args_types((None,))

    def test_accepts_proxy(self):
        """Proxy objects appear during FX tracing (e.g. GraphModule deserialization).

        validate_subgraph_args_types must accept them so that higher-order ops
        like flex_attention can be retraced during pickle.loads.
        """
        tracer = torch.fx.Tracer()
        graph = torch.fx.Graph()
        proxy = torch.fx.Proxy(graph.placeholder("x"), tracer)
        validate_subgraph_args_types((proxy,))

    def test_accepts_proxy_mixed_with_tensor(self):
        tracer = torch.fx.Tracer()
        graph = torch.fx.Graph()
        proxy = torch.fx.Proxy(graph.placeholder("x"), tracer)
        validate_subgraph_args_types((proxy, torch.tensor(1.0), 42))


class TestGraphModulePickleWithHOP(unittest.TestCase):
    """GraphModule pickle roundtrip with higher-order ops that carry lifted buffers.

    When a HOP like flex_attention has non-empty mask_mod_other_buffers, the
    GraphModule's generated forward code passes those buffers to the HOP call.
    During pickle.loads, KeepModules().trace() retraces this forward code,
    producing Proxy objects for those buffer arguments. The HOP's __call__
    validates argument types before dispatch, so it must tolerate Proxy objects.
    """

    def _build_gm_with_hop_buffers(self):
        """Build a GraphModule that calls flex_attention with a mask_mod buffer.

        Creates an FX graph where flex_attention receives a non-empty
        mask_mod_other_buffers tuple — the pattern that triggers the
        Proxy validation issue during deserialization.
        """
        score_mod = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        score_mod_graph = score_mod.graph
        for name in ["score", "b", "h", "m", "n"]:
            score_mod_graph.placeholder(name)
        score_node = next(iter(score_mod_graph.nodes))
        score_mod_graph.output(score_node)
        score_mod.recompile()

        mask_mod = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        mask_mod_graph = mask_mod.graph
        for name in ["b", "h", "m", "n"]:
            mask_mod_graph.placeholder(name)
        mask_mod_graph.placeholder("mask_buf")
        b_node = next(iter(mask_mod_graph.nodes))
        mask_mod_graph.output(b_node)
        mask_mod.recompile()

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.score_mod_0 = score_mod
        gm.mask_mod_0 = mask_mod

        g = gm.graph
        gm_q = g.placeholder("q")
        gm_k = g.placeholder("k")
        gm_v = g.placeholder("v")
        mask_buf_ph = g.placeholder("mask_buf")

        score_attr = g.get_attr("score_mod_0")
        g.get_attr("mask_mod_0")

        block_mask = (
            128,
            128,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            128,
            128,
        )

        hop_node = g.call_function(
            torch.ops.higher_order.flex_attention,
            args=(
                gm_q,
                gm_k,
                gm_v,
                score_attr,
                block_mask,
                0.125,
                {},
                (),
                (mask_buf_ph,),
            ),
        )

        g.output(hop_node)
        gm.recompile()
        return gm

    def test_pickle_roundtrip_flex_attention_with_buffers(self):
        """GraphModule with flex_attention + mask_mod buffers survives pickle roundtrip.

        This is the exact failure mode from torchtitan's precompile path:
        the AOTCompiledArtifact containing a compiled flex_attention region
        is serialized/deserialized via pickle. During deserialization,
        KeepModules().trace() retraces the forward, which calls
        flex_attention.__call__ → validate_subgraph_args_types on the
        mask_mod_other_buffers. Without the Proxy fix, this crashes with:

            AssertionError: (Proxy(mask_buf),) can only be of
            (<class 'torch.Tensor'>, <class 'int'>, <class 'torch.SymInt'>)
            but got (<class 'torch.fx.proxy.Proxy'>,)
        """
        gm = self._build_gm_with_hop_buffers()

        data = pickle.dumps(gm)
        restored = pickle.loads(data)

        self.assertIsInstance(restored, torch.fx.GraphModule)

        hop_nodes = [
            n
            for n in restored.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.higher_order.flex_attention
        ]
        self.assertEqual(len(hop_nodes), 1)


if __name__ == "__main__":
    unittest.main()
