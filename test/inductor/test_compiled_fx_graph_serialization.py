# Owner(s): ["module: inductor"]

"""Tests that CompiledFxGraph.prepare_for_serialization clears _original_gm.

_original_gm is a deepcopy of the pre-compiled FX graph, stored only when
wrap_inductor_compiled_regions is enabled. It exists solely for FakeTensor
shape inference (inductor_compiled_code_fake). Serializing it is unnecessary
for runtime execution and causes crashes when the graph contains higher-order
ops with non-picklable lifted buffers (e.g. flex_attention with BlockMask
mask_mod_other_buffers — deserialization retraces via KeepModules().trace()
and validate_subgraph_args_types rejects the resulting Proxy objects).
"""

import torch
from torch._inductor.output_code import CompiledFxGraph
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCompiledFxGraphSerialization(TestCase):
    def test_original_gm_cleared_on_prepare_for_serialization(self):
        """_original_gm must be None after prepare_for_serialization.

        This prevents pickle.loads from retracing GraphModules that contain
        higher-order ops with non-serializable lifted buffers.
        """
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.graph.placeholder("x")
        gm.graph.output(next(iter(gm.graph.nodes)))
        gm.recompile()

        cfg = CompiledFxGraph.__new__(CompiledFxGraph)
        cfg._original_gm = gm
        cfg.current_callable = None
        cfg.recursively_apply_fns = None
        cfg.compiled_fn_runner = None

        self.assertIsNotNone(cfg._original_gm)
        cfg.prepare_for_serialization()
        self.assertIsNone(cfg._original_gm)

    def test_original_gm_none_survives_pickle(self):
        """A CompiledFxGraph with _original_gm=None can be pickled."""
        cfg = CompiledFxGraph.__new__(CompiledFxGraph)
        cfg._original_gm = None
        cfg.current_callable = None
        cfg.recursively_apply_fns = None
        cfg.compiled_fn_runner = None

        cfg.prepare_for_serialization()
        self.assertIsNone(cfg._original_gm)


if __name__ == "__main__":
    run_tests()
