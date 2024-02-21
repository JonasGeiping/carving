import torch
from typing import Optional

import json
import hashlib


class ReverseCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    "Stripped-down version of reverse cross entropy. Does not support all options of torch CrossEntropyLoss."

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "none",
        label_smoothing: float = 0.0,
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        assert label_smoothing == 0.0
        assert weight is None
        assert reduction == "none"

    @torch.compile()
    def forward(self, input: torch.Tensor, target: torch.Tensor, eps: float = torch.finfo(torch.float).smallest_normal) -> torch.Tensor:
        # dumb, but fast:
        probs = torch.nn.functional.softmax(input[target != self.ignore_index].float(), dim=-1).reshape(-1, input.shape[-1])
        target_probs = torch.gather(probs, 1, target[target != self.ignore_index].unsqueeze(1)).squeeze(1)
        return (-torch.log(torch.ones_like(target_probs) - target_probs + eps)).mean(dim=-1)  # eps will be extinguished if no 0+eps
        # more stable, but expensive:
        # valid_inputs = input[target != self.ignore_index].float().reshape(-1, input.shape[-1])
        # normalized_inputs = valid_inputs - valid_inputs.max(dim=1, keepdim=True).values
        # exp_of_logit_selected = torch.gather(normalized_inputs.exp(), 1, target[target != self.ignore_index].unsqueeze(1)).squeeze(1)
        # guarded_exp_difference = torch.clamp(normalized_inputs.exp().sum(dim=-1) - exp_of_logit_selected, min=eps)
        # ce_per_token = -torch.log(guarded_exp_difference) + torch.logsumexp(normalized_inputs, dim=-1)
        # return ce_per_token.mean(dim=-1)
        # # This currently returns batch-averaged loss. This is fine for now, but should be fixed eventually to return per-example losses


class OneDimCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Utility to compute cross-entropy on flattened last dimension and cast to float32."""

    @torch.compile()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        one_dim_t = target.view(-1)
        one_dim_input = input.float().reshape(-1, input.shape[-1])
        return super().forward(one_dim_input, one_dim_t)[one_dim_t != self.ignore_index].mean(dim=-1)
        # This currently returns batch-averaged loss. This is fine for now, but should be fixed eventually to return per-example losses


class MaxCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    "Max along sequence dim"

    @torch.compile()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # masked location naturally do not contribute, as loss is zero there
        return super().forward(input.float().reshape(-1, input.shape[-1]), target.view(-1)).view_as(target).max(dim=-1).values


class LSECrossEntropyLoss(torch.nn.CrossEntropyLoss):
    "a soft Max (in the actual sense of the word) along the sequence dim."

    @torch.compile()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # masked locations do not contribute enough to matter
        return torch.logsumexp(super().forward(input.float().reshape(-1, input.shape[-1]), target.view(-1)).view_as(target), dim=-1)


class ReverseLSECrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """a soft Max (in the actual sense of the word) along the sequence dim, but attempted to shoehorn a reverse direction into it."""

    @torch.compile()
    def forward(self, input: torch.Tensor, target: torch.Tensor, eps: float = torch.finfo(torch.float).smallest_normal) -> torch.Tensor:
        probs = torch.nn.functional.softmax(input[target != self.ignore_index].float(), dim=-1).reshape(-1, input.shape[-1])
        target_probs = torch.gather(probs, 1, target[target != self.ignore_index].unsqueeze(1)).squeeze(1)
        cent_loss = -torch.log(torch.ones_like(target_probs) - target_probs + eps)  # eps will be extinguished if no 0+eps
        reexpanded_cent_loss = torch.zeros_like(target, dtype=input.dtype)
        reexpanded_cent_loss[target != self.ignore_index] = cent_loss
        return torch.logsumexp(reexpanded_cent_loss, dim=-1)


def hash_args(args_and_kwargs):
    """Probably a terrible idea?"""
    # strongly inspired by https://death.andgravity.com/stable-hashing (which is a much better take)
    # modified to handle a few usual suspects in HF code
    args_and_kwargs.pop("self")
    try:
        args_and_kwargs["model_config"] = args_and_kwargs["model"].config
    except ValueError:
        pass

    def skip_non_serializable(obj):
        """Attempt to serialize non-serializable objects. If not possible, skip them."""
        try:
            json.dumps(obj)
            return obj  # Object is serializable
        except TypeError:
            try:
                json.dumps(repr(obj))
                return repr(obj)
            except TypeError:
                return None

    dump = json.dumps(
        args_and_kwargs, default=skip_non_serializable, ensure_ascii=False, sort_keys=True, indent=None, separators=(",", ":")
    )
    return hashlib.md5(dump.encode("utf-8")).digest().hex()
