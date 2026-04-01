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
        not torch.cuda.is_available()
        or not hasattr(torch._C._distributed_c10d, "_NcclEpGroup"),
        "Test requires USE_NCCL_EP build",
    )


NUM_TOKENS = 16
TOP_K = 1
HIDDEN = 64
TOKEN_SIZE_BYTES = HIDDEN * 2
NUM_MULTI_ROUND_DISPATCH_COMBINE = 2


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
    _cached_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    @classmethod
    def get_token_switch(cls) -> TokenSwitchNCCL:
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            rank = dist.get_rank(pg)
            world_size = dist.get_world_size(pg)
            print(f"rank {rank} creating token switch")
            dist.barrier(pg)
            cls._cached_token_switch = TokenSwitchNCCL(
                pg, world_size, NUM_TOKENS, TOKEN_SIZE_BYTES
            )
        else:
            # Make sure we clear switch state between tests
            cls._cached_token_switch._handle = None
            cls._cached_token_switch._topk_idx = None
        return cls._cached_token_switch

    def _init(self):
        torch.cuda.set_device(self.device)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_dispatch_without_bind_raises(self):
        self._init()
        ts = self.get_token_switch()
        tokens = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        topk_weights = torch.zeros(
            (NUM_TOKENS, TOP_K), dtype=torch.float32, device=self.device
        )
        n = self.world_size * NUM_TOKENS
        out_tokens = torch.zeros((n, HIDDEN), dtype=torch.bfloat16, device=self.device)
        out_topk_weights = torch.zeros(
            (n, TOP_K), dtype=torch.float32, device=self.device
        )
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
        ts = self.get_token_switch()
        num_experts = self.world_size
        num_local_experts = num_experts // self.world_size
        topk_idx, _topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            num_local_experts, dtype=torch.int32, device=self.device
        )
        ts.bind_routing(topk_idx, per_expert_counts)
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)

        topk_idx, _topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        ts.bind_routing(topk_idx)
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

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
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            f"rank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

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

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_multiple_rounds(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _make_routing(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        ts.bind_routing(topk_idx)

        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )

        for r in range(NUM_MULTI_ROUND_DISPATCH_COMBINE):
            token_val = float(self.rank + 1 + r)
            tokens = torch.full(
                (NUM_TOKENS, HIDDEN),
                token_val,
                dtype=torch.bfloat16,
                device=self.device,
            )
            ts.dispatch(
                tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
            )
            torch.cuda.synchronize()

            expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
            ts.combine(expert_tokens, combined)
            torch.cuda.synchronize()

            expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
            self.assertEqual(combined.cpu(), expected)


if __name__ == "__main__":
    run_tests()
