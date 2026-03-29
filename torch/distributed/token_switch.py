# mypy: allow-untyped-defs
from __future__ import annotations

import abc

import torch
from torch.distributed.distributed_c10d import ProcessGroup


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`bind_routing`, then :meth:`dispatch` / :meth:`combine`.
    """

    @abc.abstractmethod
    def bind_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> None:
        """Attach expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dispatch(
        self,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        """Route tokens to experts; writes ``out_*`` tensors.

        Uses ``topk_idx`` from the prior :meth:`bind_routing` call.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def combine(self, expert_tokens: torch.Tensor, out_tokens: torch.Tensor) -> None:
        """Gather expert outputs back to token order; writes ``out_tokens``."""
        raise NotImplementedError


class TokenSwitchNCCL(TokenSwitch):
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_tokens_per_rank: int,
        token_size_bytes: int,
    ) -> None:
        c10d = torch._C._distributed_c10d
        if not hasattr(c10d, "_NcclEpGroup"):
            raise RuntimeError(
                "TokenSwitchNCCL requires a build with NCCL EP (USE_NCCL_EP)."
            )
        # NCCL_EP_AUTO (0) for qp count and channel count
        self._group = c10d._NcclEpGroup.create(
            process_group,
            num_experts,
            max_tokens_per_rank,
            token_size_bytes,
            0,
            0,
        )
        self._handle = None
        self._topk_idx: torch.Tensor | None = None

    def bind_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> None:
        """Attach expert routing for this phase; call before :meth:`dispatch` / :meth:`combine`."""
        c10d = torch._C._distributed_c10d
        self._topk_idx = topk_idx
        self._handle = c10d._NcclEpHandle.create(
            self._group, topk_idx, per_expert_token_counts
        )

    def dispatch(
        self,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        if self._handle is None or self._topk_idx is None:
            raise RuntimeError("Call bind_routing before dispatch().")
        torch._C._distributed_c10d._nccl_ep_dispatch(
            self._handle,
            tokens,
            topk_weights,
            self._topk_idx,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def combine(self, expert_tokens: torch.Tensor, out_tokens: torch.Tensor) -> None:
        if self._handle is None:
            raise RuntimeError("Call bind_routing before combine().")
        torch._C._distributed_c10d._nccl_ep_combine(
            self._handle,
            expert_tokens,
            out_tokens,
        )
