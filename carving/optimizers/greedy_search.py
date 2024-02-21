""" Simplest optimizer."""
import torch

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)


class GreedyOptimizer(_GenericOptimizer):
    def __init__(self, *args, setup=_default_setup, save_checkpoint=False, **kwargs):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)

    def solve(self, sigil, dryrun=False, **kwargs):
        """"""
        prompt = torch.as_tensor([sigil.tokenizer.pad_token_id for idx in range(len(sigil))], device=self.setup["device"])[None]
        for prompt_idx in range(len(sigil)):
            possible_token_at_this_index = sigil.constraint.set_at_idx(prompt_idx)

            val_tensor = torch.zeros(len(possible_token_at_this_index), **self.setup)
            with torch.inference_mode():
                for list_idx, token_idx in enumerate(possible_token_at_this_index):
                    input_ids = prompt.clone()
                    input_ids[:, prompt_idx] = token_idx
                    val_tensor[list_idx] = sigil.objective(input_ids=input_ids).mean()
                    if dryrun:
                        break
                prompt[:, prompt_idx] = possible_token_at_this_index[val_tensor.argmin().item()]
            self.callback(sigil, prompt[0], None, val_tensor[list_idx], prompt_idx)
        return torch.as_tensor(prompt, device=self.setup["device"])
