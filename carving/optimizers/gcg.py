"""The optimizer from Universal and Transferable Adversarial Attacks on Aligned Language Models."""

import torch

import math
import random

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)
max_retries = 20
progress_threshold = 0.5  # only valid for progressive expansion. Hardcoded for now


class GCGOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        steps=500,
        batch_size=512,
        topk=256,
        temp=1,
        filter_cand=True,
        anneal=False,
        freeze_objective_in_search=False,
        progressive_expansion=False,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.steps = steps
        self.batch_size = batch_size
        self.topk = topk
        self.temp = temp
        self.filter_cand = filter_cand
        self.anneal = anneal
        self.freeze_objective_in_search = freeze_objective_in_search
        self.progressive_expansion = progressive_expansion

        self.anneal_from = 0

    def token_gradients(self, sigil, input_ids, state=None):
        """
        Computes gradients of the loss with respect to the coordinates.
        Todo: make this more efficient by computing gradients only for embeddings in constraint.set_at_idx
        """
        sigil.model.train()  # necessary to trick HF transformers to allow gradient checkpointing
        one_hot = torch.zeros(input_ids.shape[1], sigil.num_embeddings, **self.setup)
        one_hot.scatter_(1, input_ids[0].unsqueeze(1), 1)
        one_hot.requires_grad_()
        inputs_embeds = (one_hot @ sigil.embedding.weight).unsqueeze(0)
        adv_loss = sigil.objective(inputs_embeds=inputs_embeds, mask_source=input_ids, state=state).mean()
        (input_ids_grads,) = torch.autograd.grad(adv_loss, [one_hot])  # not compatible with reentrant=True gradient checkpointing
        sigil.model.eval()  # necessary to trick HF transformers to disable gradient checkpointing
        return input_ids_grads
        # adv_loss.backward() # alternative for reentrant grad checkpointing
        # return one_hot.grad

    def P(self, e, e_prime, k, anneal_from=0):
        T = max(1 - float(k + 1) / (self.steps + self.anneal_from), 1.0e-7)
        return True if e_prime < e else math.exp(-(e_prime - e) / T) >= random.random()

    @torch.no_grad()
    def sample_candidates(self, input_ids, grad, constraint, batch_size, topk=256, temp=1):
        top_indices = constraint.select_topk(-grad, k=topk)
        original_input_ids = input_ids.repeat(batch_size, 1)
        weights = torch.as_tensor([idx.shape[0] - 1 for idx in top_indices], device=input_ids.device, dtype=torch.float)
        new_token_pos = torch.multinomial(weights, batch_size, replacement=True)
        new_token_val = constraint.gather_random_element(top_indices, new_token_pos)
        new_input_ids = original_input_ids.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val.unsqueeze(-1))
        return new_input_ids

    @torch.no_grad()
    def get_filtered_cands(self, candidate_ids, constraint):
        if self.filter_cand:
            candidate_is_valid = constraint.is_tokenization_safe(candidate_ids)
            if sum(candidate_is_valid) > 0:
                return candidate_ids[candidate_is_valid], True
            else:
                print(f"No valid candidate accepted out of {len(candidate_ids)} candidates.")
                return candidate_ids, False
        else:
            return candidate_ids, True

    def solve(self, sigil, initial_guess=None, initial_step=0, dryrun=False, **kwargs):
        """"""
        if len(sigil.constraint) < self.topk:
            new_topk = len(sigil.constraint) // 2
            print(f"Constraint space of size {len(sigil.constraint)} too small for {self.topk} topk entries. Reducing to {new_topk}.")
            self.topk = new_topk

        if self.progressive_expansion:
            if hasattr(sigil, "progressive_expansion"):
                sigil.progressive_expansion = True
            else:
                raise ValueError(f"Sigil {sigil} does not support progressive expansion.")

        # Initialize solver
        best_loss = float("inf")
        prev_loss = float("inf")
        if initial_guess is None:
            prompt_ids = sigil.constraint.draw_random_sequence(device=self.setup["device"])
        else:
            if len(initial_guess) != sigil.num_tokens:
                raise ValueError(f"Initial guess does not match expected number of tokens ({sigil.num_tokens}).")
            else:
                prompt_ids = torch.as_tensor(initial_guess, device=self.setup["device"])[None]

        # print(f"Initial Prompt is: {sigil.tokenizer.decode(prompt_ids[0])}")
        best_prompt_ids = prompt_ids.clone()
        init_state = initial_step if self.freeze_objective_in_search else None
        best_loss = prev_loss = sigil.objective(input_ids=best_prompt_ids, state=init_state).to(dtype=torch.float32).mean().item()

        for iteration in range(initial_step, self.steps):
            # Optionally freeze objective state
            state = iteration if self.freeze_objective_in_search else None
            if self.progressive_expansion and best_loss < progress_threshold:
                print(f"Loss threshold reached with loss {best_loss} in step {iteration}, expanding target length.")
                state = f"expand_{state}"
                best_loss = prev_loss = float("inf")
            # Aggregate gradients
            grad = self.token_gradients(sigil, prompt_ids, state=state)
            normalized_grad = grad / grad.norm(dim=-1, keepdim=True)

            # Select candidates
            for retry in range(max_retries):
                # Sample candidates
                candidates = self.sample_candidates(prompt_ids, normalized_grad, sigil.constraint, self.batch_size, self.topk, self.temp)
                # Filter candidates:
                candidates, valid_candidates_found = self.get_filtered_cands(candidates, sigil.constraint)
                if valid_candidates_found:
                    break

            # Search
            loss = torch.zeros(len(candidates), dtype=torch.float32, device=self.setup["device"])
            unique_candidates, uniques_map = torch.unique(candidates, dim=0, return_inverse=True)
            # print(f"Unique fwd passes: {len(unique_candidates)}")
            with torch.no_grad():
                for j, candidate_ids in enumerate(unique_candidates):
                    loss[j == uniques_map] = sigil.objective(input_ids=candidate_ids[None], state=state).to(dtype=torch.float32).mean()

            # Return best from batch:
            minimal_loss_in_iteration = loss.argmin()
            best_candidate = candidates[minimal_loss_in_iteration]
            loss_for_best_candidate = loss[minimal_loss_in_iteration]

            # Anneal?
            keep_prompt = True if not self.anneal else self.P(prev_loss, loss_for_best_candidate, iteration + self.anneal_from)
            if keep_prompt:
                prompt_ids = best_candidate[None]
            prev_loss = loss_for_best_candidate  # Not within the keep_prompt scope in the initial implementation

            if loss_for_best_candidate < best_loss:
                best_loss = loss_for_best_candidate.item()
                best_prompt_ids = best_candidate[None]

            self.callback(sigil, best_candidate[None], best_prompt_ids, loss_for_best_candidate.detach(), iteration, **kwargs)
            if dryrun:
                break

        return best_prompt_ids  # always return with leading dimension
