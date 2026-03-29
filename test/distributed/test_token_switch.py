# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed import TokenSwitchNCCL
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


def requires_nccl_ep():
    return skip_but_pass_in_sandcastle_if(
        not torch.cuda.is_available() or not hasattr(torch._C._distributed_c10d, "_NcclEpGroup"),
        "Test requires USE_NCCL_EP build",
    )


NUM_TOKENS = 16
TOP_K = 1
HIDDEN = 64
TOKEN_SIZE_BYTES = HIDDEN * 2


def _make_routing(rank, world_size, num_tokens, top_k, device):
    remote_expert = (rank + 1) % world_size
    topk_idx = torch.full(
        (num_tokens, top_k), remote_expert, dtype=torch.int64, device=device
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return topk_idx, topk_weights


@requires_nccl_ep()
class TokenSwitchNCCLTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _pg(self):
        return dist.distributed_c10d._get_default_group()

    def _init(self):
        torch.cuda.set_device(self.device)
        dist.all_reduce(torch.tensor(1.0, device=self.device))
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    def test_dispatch_without_bind_raises(self):
        self._init()
        ts = TokenSwitchNCCL(
            self._pg(), self.world_size, NUM_TOKENS, TOKEN_SIZE_BYTES
        )
        tokens = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        topk_weights = torch.zeros(
            (NUM_TOKENS, TOP_K), dtype=torch.float32, device=self.device
        )
        n = self.world_size * NUM_TOKENS
        out_tokens = torch.zeros((n, HIDDEN), dtype=torch.bfloat16, device=self.device)
        out_topk_weights = torch.zeros((n, TOP_K), dtype=torch.float32, device=self.device)
        out_topk_idx = torch.zeros((n, TOP_K), dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "bind_routing"):
            ts.dispatch(
                tokens,
                topk_weights,
                out_tokens,
                out_topk_weights,
                out_topk_idx,
            )

    @skip_if_lt_x_gpu(2)
    def test_bind_routing(self):
        self._init()
        num_experts = self.world_size
        num_local_experts = num_experts // self.world_size
        ts = TokenSwitchNCCL(self._pg(), num_experts, NUM_TOKENS, TOKEN_SIZE_BYTES)
        topk_idx, _topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            num_local_experts, dtype=torch.int32, device=self.device
        )
        ts.bind_routing(topk_idx, per_expert_counts)
        torch.cuda.synchronize()
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        self._init()
        num_experts = self.world_size
        num_recv_tokens = self.world_size * NUM_TOKENS

        ts = TokenSwitchNCCL(self._pg(), num_experts, NUM_TOKENS, TOKEN_SIZE_BYTES)
        topk_idx, topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        ts.bind_routing(topk_idx)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx)
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float().cpu()
        self.assertTrue(
            received.eq(expected_val).all(),
            f"rank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip(self):
        self._init()
        num_experts = self.world_size
        num_recv_tokens = self.world_size * NUM_TOKENS

        ts = TokenSwitchNCCL(self._pg(), num_experts, NUM_TOKENS, TOKEN_SIZE_BYTES)
        topk_idx, topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        ts.bind_routing(topk_idx)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx)
        torch.cuda.synchronize()

        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(expert_tokens, combined)
        torch.cuda.synchronize()

        expected = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16
        )
        self.assertEqual(combined.cpu(), expected)


if __name__ == "__main__":
    run_tests()
