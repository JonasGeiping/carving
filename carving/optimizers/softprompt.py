"""This is a sanity check optimizer, optimizing only a soft-prompt.

This strategy does not actually generate attack tokens, but it can be used to test whether an optimization objective can be minimized at all.
"""


import torch
import time

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)


class SoftPromptOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        steps=1000,
        lr=1,
        weight_decay=0.01,
        T_max=1000,
        tokenization_safe=True,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.tokenization_safe = tokenization_safe

    def solve(self, sigil, initial_guess=None, dryrun=False, **kwargs):
        """"""
        assert sigil.constraint.is_uniform  # for now

        # Initial Value
        if initial_guess is None:
            init_x = sigil.constraint.set[torch.randint(len(sigil.constraint), (1, len(sigil))).to(self.setup["device"])]
        else:
            init_x = torch.as_tensor(initial_guess, device=self.setup["device"])[None]
        with torch.no_grad():
            init_x = sigil.embedding(init_x)
        soft_embeddings = init_x.clone().to(dtype=torch.float32)  # stabilize optimizer precision
        soft_embeddings.requires_grad = True

        # Optimizers
        adv_optimizer = torch.optim.AdamW([soft_embeddings], lr=self.lr, weight_decay=self.weight_decay)
        adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adv_optimizer, self.T_max)

        # Initialize solver
        best_loss = torch.tensor(float("inf"), **self.setup)
        time_stamp = time.time()

        for step in range(self.steps):
            # Get loss
            adv_loss = sigil.objective(inputs_embeds=soft_embeddings.to(self.setup["dtype"])).mean()
            # update
            (soft_embeddings.grad,) = torch.autograd.grad(adv_loss, [soft_embeddings])
            adv_optimizer.step()
            adv_optimizer.zero_grad()
            adv_scheduler.step()

            # Project for later:
            with torch.no_grad():
                _, nn_indices = sigil.constraint.normalized_project(soft_embeddings.to(self.setup["dtype"]), topk=1)
                if self.tokenization_safe:
                    safe_ids = sigil.constraint.project_onto_tokenizable_ids(nn_indices)
                else:
                    safe_ids = nn_indices

                if adv_loss.detach() < best_loss:
                    best_loss = adv_loss.detach().clone()
                    best_prompt = safe_ids

                curr_lr = adv_optimizer.param_groups[-1]["lr"]
                best_prompt_decoded = sigil.tokenizer.decode(best_prompt[0])
                printed_prompt = best_prompt_decoded if len(best_prompt_decoded) < 40 else best_prompt_decoded[:38] + "[...]"
                self.callback(sigil, safe_ids, best_prompt, adv_loss.detach(), step, info=f"curr_lr: {curr_lr:.3f}|", **kwargs)
                if dryrun:
                    break
        return best_prompt  # always return with leading dimension
