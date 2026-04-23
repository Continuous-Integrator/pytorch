# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


def _annotate_params_with_spmd_types(model, mesh):
    """Annotate all model parameters with spmd_types R (Replicate) on all dims."""
    from spmd_types._mesh_axis import MeshAxis
    from spmd_types._type_attr import set_local_type
    from spmd_types.types import R

    for param in model.parameters():
        local_type = {}
        for name in mesh.mesh_dim_names:
            axis = MeshAxis.of(mesh.get_group(name))
            local_type[axis] = R
        set_local_type(param, local_type)


class TestFullyShardSpmdTypes(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    def _run_train_parity(self, model, ref_model, dp_pg, mesh, num_iters=5, mlp_dim=16):
        from spmd_types._checker import typecheck
        from spmd_types._mesh import set_current_mesh
        from spmd_types._mesh_axis import MeshAxis
        from spmd_types._type_attr import get_local_type
        from spmd_types.runtime import assert_type, has_local_type
        from spmd_types.types import S, V

        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))

        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)

        # Verify params are plain tensors with spmd_types during compute
        def check_spmd_types_hook(module, args):
            for name, param in module.named_parameters(recurse=False):
                self.assertNotIsInstance(
                    param.data,
                    DTensor,
                    f"{name} should be a plain tensor during forward, not DTensor",
                )
                self.assertTrue(
                    has_local_type(param),
                    f"Missing spmd_types on {name} during forward",
                )

        handles = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_pre_hook(check_spmd_types_hook))

        # Wrap typecheck around the MLP's own forward (after FSDP's pre-forward
        # hook unshards params, before FSDP's post-forward hook reshards them).
        tc_ctx = [None]

        def enter_typecheck(module, args):
            tc_ctx[0] = typecheck(strict_mode="strict")
            tc_ctx[0].__enter__()
            assert_type(args[0], {fsdp_axis: S(0)})

        def exit_typecheck(module, args, output):
            tc_ctx[0].__exit__(None, None, None)
            tc_ctx[0] = None

        handles.append(model.register_forward_pre_hook(enter_typecheck, prepend=False))
        handles.append(model.register_forward_hook(exit_typecheck))

        torch.manual_seed(42 + dp_pg.rank() + 1)
        with set_current_mesh(mesh):
            for i in range(num_iters):
                inp = torch.randn((2, mlp_dim), device=device_type)
                ref_optim.zero_grad(set_to_none=(i % 2 == 0))
                ref_loss = ref_model(inp).sum()
                ref_loss.backward()
                ref_optim.step()

                optim.zero_grad(set_to_none=(i % 2 == 0))
                output = model(inp)

                # Output inherits S(0) on fsdp axis through matmuls and relus.
                local_type = get_local_type(output)
                self.assertIn(fsdp_axis, local_type)
                self.assertEqual(local_type[fsdp_axis], V)
                assert_type(output, {fsdp_axis: S(0)})

                loss = output.sum()
                loss.backward()
                optim.step()

                self.assertEqual(ref_loss, loss)

        for h in handles:
            h.remove()

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters(), strict=True
        ):
            if isinstance(p2, DTensor):
                p2_full = p2.full_tensor()
            elif has_local_type(p2):
                # Gather sharded spmd_types param for comparison
                gathered = [torch.empty_like(p2) for _ in range(dp_pg.size())]
                dist.all_gather(gathered, p2, group=dp_pg)
                p2_full = torch.cat(gathered, dim=0)
            else:
                p2_full = p2
            self.assertEqual(p1, p2_full, msg=f"Param mismatch: {n1} vs {n2}")

    @skip_if_lt_x_gpu(2)
    def test_fsdp_1d_spmd_types_train_parity(self):
        """Train parity: spmd_types-annotated params with 1D FSDP."""
        from torch.spmd_types import _reset

        _reset()
        mlp_dim = 16
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        torch.manual_seed(42)
        model = MLP(mlp_dim, device=device_type)
        ref_model = copy.deepcopy(model)

        _annotate_params_with_spmd_types(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        from torch.distributed._composable import replicate

        replicate(
            ref_model,
            device_ids=[self.rank] if device_type.type != "cpu" else None,
        )
        self._run_train_parity(
            model, ref_model, dist.group.WORLD, mesh=mesh, mlp_dim=mlp_dim
        )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_1d_spmd_types_sharded_param_correctness(self):
        """Verify sharded params are plain tensors with spmd_types S annotations."""
        from spmd_types._mesh_axis import MeshAxis
        from spmd_types._type_attr import get_local_type
        from spmd_types.types import Shard as SpmdShard

        from torch.spmd_types import _reset

        _reset()
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))

        model = MLP(16, device=device_type)
        _annotate_params_with_spmd_types(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )
        for param in model.parameters():
            self.assertNotIsInstance(param, DTensor)
            local_type = get_local_type(param)
            self.assertIn(fsdp_axis, local_type)
            self.assertIsInstance(local_type[fsdp_axis], SpmdShard)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_spmd_types_sharded_param_correctness(self):
        """Verify sharded param types for FSDP+TP with spmd_types."""
        from spmd_types._mesh_axis import MeshAxis
        from spmd_types._type_attr import get_local_type
        from spmd_types.types import Shard as SpmdShard

        from torch.spmd_types import _reset

        _reset()
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = MeshAxis.of(mesh.get_group("tp"))

        model = MLP(16, device=device_type)

        # Annotate params with TP sharding info + R on FSDP dim
        from spmd_types._type_attr import set_local_type
        from spmd_types.types import R, S

        for name, param in model.named_parameters():
            local_type = {fsdp_axis: R}
            if "weight" in name:
                if "in_proj" in name:
                    local_type[tp_axis] = S(0)
                else:
                    local_type[tp_axis] = S(1)
            else:
                local_type[tp_axis] = R
            set_local_type(param, local_type)

        # For TP-sharded params, we need to actually shard the data
        # Use distribute_tensor to create the TP shards, then convert back
        # to plain tensors with spmd_types
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            for param_name, param in list(module.named_parameters(recurse=False)):
                saved_type = get_local_type(param)
                if param_name == "weight":
                    parent_name = next(
                        n for n, m in model.named_modules() if m is module
                    )
                    if "in_proj" in parent_name:
                        tp_placements = [Replicate(), Shard(0)]
                    else:
                        tp_placements = [Replicate(), Shard(1)]
                else:
                    tp_placements = [Replicate(), Replicate()]
                dt = distribute_tensor(param.data, mesh, tp_placements)
                new_param = nn.Parameter(
                    dt._local_tensor.clone(), requires_grad=param.requires_grad
                )
                set_local_type(new_param, saved_type)
                module.register_parameter(param_name, new_param)

        def shard_fn(param):
            lt = get_local_type(param)
            tp_type = lt.get(tp_axis)
            if isinstance(tp_type, S) and tp_type.dim == 0:
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        for name, param in model.named_parameters():
            self.assertNotIsInstance(param, DTensor)
            local_type = get_local_type(param)
            self.assertIn(fsdp_axis, local_type)
            self.assertIsInstance(local_type[fsdp_axis], SpmdShard)
            self.assertIn(tp_axis, local_type)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_1d_optimizer_state_spmd_types(self):
        """Optimizer state should be plain tensors matching sharded param shape."""
        from spmd_types._mesh_axis import MeshAxis
        from spmd_types._type_attr import get_local_type
        from spmd_types.runtime import has_local_type
        from spmd_types.types import Shard as SpmdShard

        from torch.spmd_types import _reset

        _reset()
        mlp_dim = 16
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))

        torch.manual_seed(42)
        model = MLP(mlp_dim, device=device_type)
        _annotate_params_with_spmd_types(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # Run one forward/backward/step to populate optimizer state
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device=device_type)
        model(inp).sum().backward()
        optim.step()

        for param in model.parameters():
            # Param should still be spmd_types annotated after step
            self.assertNotIsInstance(param, DTensor)
            self.assertTrue(has_local_type(param))
            local_type = get_local_type(param)
            self.assertIsInstance(local_type[fsdp_axis], SpmdShard)

            # Optimizer state should be plain tensors with same shape as param
            state = optim.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertIn(key, state)
                self.assertNotIsInstance(state[key], DTensor)
                self.assertEqual(state[key].shape, param.shape)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_optimizer_state_spmd_types(self):
        """Optimizer state for FSDP+TP should be plain tensors."""
        from spmd_types import all_reduce, convert
        from spmd_types._mesh_axis import MeshAxis
        from spmd_types._type_attr import get_local_type, set_local_type
        from spmd_types.runtime import has_local_type
        from spmd_types.types import I, R, S, Shard as SpmdShard

        from torch.spmd_types import _reset

        _reset()
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = MeshAxis.of(mesh.get_group("tp"))
        tp_pg = mesh.get_group("tp")

        # MLP with explicit spmd_types TP collectives (Megatron style):
        #   convert(I→R) before colwise in_proj, all_reduce(P→I) after rowwise out_proj
        class SpmdTypesTPMLP(nn.Module):
            def __init__(self, dim, tp_size, device=None):
                super().__init__()
                self.in_proj = nn.Linear(
                    dim, dim * 4 // tp_size, bias=False, device=device
                )
                self.out_proj = nn.Linear(
                    dim * 4 // tp_size, dim, bias=False, device=device
                )

            def forward(self, x):
                x = convert(x, tp_pg, src=I, dst=R)
                z = self.in_proj(x)
                z = torch.nn.functional.relu(z)
                z = self.out_proj(z)
                z = all_reduce(z, tp_pg, dst=I)
                z = torch.nn.functional.relu(z)
                return z

        torch.manual_seed(42)
        model = SpmdTypesTPMLP(16, tp_size, device=device_type)

        # Annotate params: R on FSDP, S on TP
        for name, param in model.named_parameters():
            local_type = {fsdp_axis: R}
            if "in_proj" in name:
                local_type[tp_axis] = S(0)
            else:
                local_type[tp_axis] = S(1)
            set_local_type(param, local_type)

        def shard_fn(param):
            lt = get_local_type(param)
            tp_type = lt.get(tp_axis)
            if isinstance(tp_type, S) and tp_type.dim == 0:
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 16), device=device_type)
        model(inp).sum().backward()
        optim.step()

        for param in model.parameters():
            self.assertNotIsInstance(param, DTensor)
            self.assertTrue(has_local_type(param))
            local_type = get_local_type(param)
            self.assertIsInstance(local_type[fsdp_axis], SpmdShard)
            self.assertIn(tp_axis, local_type)

            state = optim.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertIn(key, state)
                self.assertNotIsInstance(state[key], DTensor)
                self.assertEqual(state[key].shape, param.shape)


if __name__ == "__main__":
    run_tests()
