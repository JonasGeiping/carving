"""The optimizer from Hard Prompts Made Easy."""
import torch
import time

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)


class HardPromptsMadeEasyOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        steps=1000,
        lr=1,
        weight_decay=0.01,
        T_max=250,
        reset_embeds=True,
        tokenization_safe=True,
        topk_project=1,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.reset_embeds = reset_embeds
        self.tokenization_safe = tokenization_safe
        self.topk_project = topk_project

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
        optim_input_embeds = torch.zeros_like(init_x)
        optim_input_embeds.data = init_x.data
        optim_input_embeds.requires_grad = True
        optim_input_embeds = optim_input_embeds.to(**self.setup)

        # Optimizers
        adv_optimizer = torch.optim.AdamW([optim_input_embeds], lr=self.lr, weight_decay=self.weight_decay)
        adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adv_optimizer, self.T_max)

        # Initialize solver
        best_loss = torch.tensor(float("inf"), **self.setup)
        time_stamp = time.time()

        for step in range(self.steps):
            # forward projection
            with torch.no_grad():
                projected_optim_input_embeds, nn_indices = sigil.constraint.normalized_project(optim_input_embeds, topk=self.topk_project)
                if self.tokenization_safe:
                    safe_ids = sigil.constraint.project_onto_tokenizable_ids(nn_indices)
                    if not torch.equal(nn_indices, safe_ids):
                        projected_optim_input_embeds = sigil.embedding(safe_ids)
                        optim_input_embeds.data = projected_optim_input_embeds.data
                else:
                    safe_ids = nn_indices

                if self.reset_embeds is True and step % self.T_max == 0:
                    optim_input_embeds.data = projected_optim_input_embeds.data

            projected_embeds = projected_optim_input_embeds.clone().requires_grad_()
            # Get loss
            adv_loss = sigil.objective(inputs_embeds=projected_embeds, mask_source=safe_ids).mean()
            # update
            (optim_input_embeds.grad,) = torch.autograd.grad(adv_loss, [projected_embeds])
            adv_optimizer.step()
            adv_optimizer.zero_grad()
            adv_scheduler.step()

            if adv_loss.detach() < best_loss:
                best_loss = adv_loss.detach().clone()
                best_prompt = safe_ids

            curr_lr = adv_optimizer.param_groups[-1]["lr"]
            best_prompt_decoded = sigil.tokenizer.decode(best_prompt[0])
            printed_prompt = best_prompt_decoded if len(best_prompt_decoded) < 40 else best_prompt_decoded[:38] + "[...]"
            self.callback(sigil, nn_indices, best_prompt, adv_loss.detach(), step, info=f"curr_lr: {curr_lr:.3f}|", **kwargs)
            if dryrun:
                break
        return best_prompt  # always return with leading dimension
