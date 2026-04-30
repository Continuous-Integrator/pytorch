# Owner(s): ["module: dynamo"]

import copy
import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch._guards import tracing, TracingContext
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import _share_torchbind_on_deepcopy
from torch.fx.passes.regional_inductor import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
    regional_inductor,
)
from torch.testing._internal.common_utils import run_tests, TestCase


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _f(t):
    t = t.clone()
    dist.all_reduce(t)
    return t + 1


def _make_fx_with_allreduce():
    return make_fx(_f)(torch.ones(4))


class TestRegionalInductorCollectives(TestCase):
    def setUp(self):
        super().setUp()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29516")
        with tempfile.NamedTemporaryFile(
            prefix="regional_inductor_store_", delete=False
        ) as fd:
            self._store_path = fd.name
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            store=dist.FileStore(self._store_path, 1),
        )

    def tearDown(self):
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        if os.path.exists(self._store_path):
            os.unlink(self._store_path)
        super().tearDown()

    def test_functionalize_inplace_allreduce(self):
        gm = _make_fx_with_allreduce()
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    _torchbind_obj1 = self._torchbind_obj1
    allreduce_ = torch.ops.c10d.allreduce_.default([clone], _torchbind_obj0, _torchbind_obj1, None, False);  clone = _torchbind_obj0 = _torchbind_obj1 = None
    getitem = allreduce_[0]
    getitem_1 = getitem[0];  getitem = None
    getitem_2 = allreduce_[1];  allreduce_ = getitem_2 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
    return add""",
        )

        _functionalize_inplace_collectives(gm)

        # ReduceOp is baked as ``'sum'`` (its int value is read at rewrite
        # time) and its now-dead ``_torchbind_obj1`` attr is stripped; the
        # ProcessGroup ``get_attr`` flows through unchanged for pass 2.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )

    def test_post_pass_gm_deepcopy(self):
        # After pass 1 + pass 2, ``_torchbind_obj0`` is a Python
        # ``dist.ProcessGroup`` — still not pickleable, but
        # ``_share_torchbind_on_deepcopy()`` makes the gm deepcopy-safe by
        # sharing the PG by reference. This is what
        # ``standalone_compile``'s deepcopy of the regional submod relies
        # on.
        gm = _make_fx_with_allreduce()
        _functionalize_inplace_collectives(gm)
        _unbox_process_group_torchbinds(gm)
        self.assertIsInstance(gm._torchbind_obj0, dist.ProcessGroup)

        with _share_torchbind_on_deepcopy():
            gm2 = copy.deepcopy(gm)
        self.assertIs(gm._torchbind_obj0, gm2._torchbind_obj0)
        self.assertExpectedInline(
            gm2.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )

    def test_unbox_process_group_torchbinds(self):
        # Pass 2 unboxes the torchbind ProcessGroup attr in-place: the FX
        # graph still references ``_torchbind_obj0`` by name, but its value
        # flips from ``torch.ScriptObject`` to a Python ``dist.ProcessGroup``
        # — the form Inductor's collective lowering and runtime ops accept.
        gm = _make_fx_with_allreduce()
        _functionalize_inplace_collectives(gm)
        self.assertIsInstance(gm._torchbind_obj0, torch.ScriptObject)

        _unbox_process_group_torchbinds(gm)

        self.assertNotIsInstance(gm._torchbind_obj0, torch.ScriptObject)
        self.assertIsInstance(gm._torchbind_obj0, dist.ProcessGroup)
        # FX graph code is unchanged — only the underlying attr was swapped.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )

        # Numerics: the gm with the unboxed PG must still match eager.
        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(gm(x), _f(x))

    def test_regional_inductor_with_dist_all_reduce(self):
        gm = _make_fx_with_allreduce()
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                    "inductor_configs": {}
                }
        fake_mode = next(
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
        )

        with tracing(TracingContext(fake_mode)):
            compiled_gm = regional_inductor(gm)

        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(compiled_gm([x]), _f(x))


if __name__ == "__main__":
    run_tests()
