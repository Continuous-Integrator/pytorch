# Owner(s): ["module: dynamo"]

import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.regional_inductor import _functionalize_inplace_collectives
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
        # time) and its now-dead ``_torchbind_obj1`` get_attr / module attr
        # is stripped; the ProcessGroup ``get_attr`` is still referenced by
        # the new functional call, so it stays.
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


if __name__ == "__main__":
    run_tests()
