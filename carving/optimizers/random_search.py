""" Another simple baseline optimizer."""
import torch

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)


class RandomSearch(_GenericOptimizer):
    def __init__(self, *args, max_iterations=1000, batch_size=1, setup=_default_setup, save_checkpoint=False, **kwargs):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def solve(self, sigil, dryrun=False, **kwargs):
        """"""
        value_lookup = dict()

        for iteration in range(self.max_iterations):
            prompt = torch.zeros(self.batch_size, len(sigil), dtype=torch.long)
            for prompt_idx in range(len(sigil)):
                constraint_set = sigil.constraint.set_at_idx(prompt_idx)
                indices = torch.randint(0, constraint_set.numel(), (self.batch_size,))
                prompt[:, prompt_idx] = constraint_set[indices]
            input_ids = torch.as_tensor(prompt, device=self.setup["device"])
            loss = torch.zeros(len(input_ids), **self.setup)
            with torch.no_grad():
                for j, input_id_example in enumerate(input_ids):
                    loss[j] = sigil.objective(input_ids=input_id_example[None]).mean()
            minimal_loss_in_iteration = loss.argmin()

            # Return best from batch:
            best_inputs = input_ids[minimal_loss_in_iteration]
            value_lookup[best_inputs] = loss[minimal_loss_in_iteration].item()

            self.callback(sigil, best_inputs[None], min(value_lookup, key=value_lookup.get), loss[minimal_loss_in_iteration], iteration)
            if dryrun:
                break
        # Return input with minimal loss
        optimal_prompt = min(value_lookup, key=value_lookup.get)
        return optimal_prompt[None]
