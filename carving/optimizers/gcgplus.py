"""The optimizer from Universal and Transferable Adversarial Attacks on Aligned Language Models, with small modifications.

Do not use, so far no evidence that these are beneficial modifications
"""
import torch

import math
import random
from functools import lru_cache

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)
max_retries = 20


class GCGPlusOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        steps=500,
        batch_size=512,
        topk=256,
        filter_cand=True,
        biased_sampler=False,
        population_size=128,
        anneal_temperature=True,
        cache_objective=False,
        freeze_objective_in_search=True,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.steps = steps
        self.batch_size = batch_size
        self.topk = topk
        self.filter_cand = filter_cand
        self.freeze_objective_in_search = freeze_objective_in_search

        self.biased_sampler = biased_sampler
        self.population_size = population_size
        self.anneal_temperature = anneal_temperature
        self.cache_objective = cache_objective

    def token_gradients(self, sigil, full_input_ids, state=None):
        """
        Computes gradients of the loss with respect to the coordinates.
        Todo: make this more efficient by computing gradients only for embeddings in constraint.set_at_idx
        """
        input_ids, uniques_map = torch.unique(full_input_ids, dim=0, return_inverse=True)
        # print(f"Unique grad passes: {len(input_ids)}")

        one_hot = torch.zeros(input_ids.shape[0], input_ids.shape[1], sigil.num_embeddings, **self.setup)
        one_hot.scatter_(2, input_ids.unsqueeze(-1), 1)
        input_ids_grads = torch.zeros_like(one_hot)

        sigil.model.train()  # necessary to trick HF transformers to allow gradient checkpointing
        for idx, input_seq in enumerate(one_hot):
            # todo: update all sigil objectives to take batched input again, if this is workable
            input_seq = input_seq.clone().detach()[None]
            input_seq.requires_grad_()
            inputs_embeds = input_seq @ sigil.embedding.weight
            adv_loss = sigil.objective(inputs_embeds=inputs_embeds, mask_source=input_ids[idx : idx + 1], state=state).mean()
            (entry_grads,) = torch.autograd.grad(adv_loss, [input_seq])
            input_ids_grads[idx] = entry_grads.detach()

        sigil.model.eval()  # necessary to trick HF transformers to disable gradient checkpointing
        return input_ids_grads[uniques_map]

    def P(self, e, e_prime, k, anneal_from=0):
        T = max(1 - float(k + 1) / (self.steps + self.anneal_from), 1.0e-7)
        return True if e_prime < e else math.exp(-(e_prime - e) / T) >= random.random()

    @torch.no_grad()
    def sample_candidates(self, input_ids, grad, constraint, batch_size, topk=256, temp=1.0):
        top_indices = constraint.select_topk(-grad, k=topk)
        repeated_input_ids = input_ids.repeat(batch_size, 1)
        weights = torch.as_tensor([idx.shape[0] - 1 for idx in top_indices], device=input_ids.device, dtype=torch.float)
        probs = torch.softmax(weights, dim=-1).repeat(batch_size, 1)
        flip_tokens_mask = torch.bernoulli(1 - (1 - probs) ** temp)  # this would be a single flip per candidate on average with temp=1
        # print(flip_tokens_mask.sum(dim=1).mean())
        loc, new_token_pos = flip_tokens_mask.nonzero(as_tuple=True)
        new_token_val = constraint.gather_random_element(top_indices, new_token_pos, scores=-grad if self.biased_sampler else None)
        repeated_input_ids[loc, new_token_pos] = new_token_val
        return repeated_input_ids

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

    @torch.no_grad()
    def initialize_population(self, constraint):
        population = []
        for p in range(self.population_size):
            population.append(constraint.draw_random_sequence(device=self.setup["device"]))
        return torch.cat(population)

    def solve(self, sigil, initial_guess=None, dryrun=False, **kwargs):
        """"""

        if len(sigil.constraint) < self.topk:
            new_topk = len(sigil.constraint) // 2
            print(f"Constraint space of size {len(sigil.constraint)} too small for {self.topk} topk entries. Reducing to {new_topk}.")
            self.topk = new_topk

        # Initialize solver
        best_loss_overall = best_loss_val = float("inf")
        prev_loss = float("inf")
        population = self.initialize_population(sigil.constraint)
        best_prompt_ids = sigil.constraint.draw_random_sequence(device=self.setup["device"])
        if initial_guess is not None:
            if len(initial_guess) != sigil.num_tokens:
                raise ValueError(f"Initial guess does not match expected number of tokens ({sigil.num_tokens}).")
            else:
                best_prompt_ids[0] = torch.as_tensor(initial_guess, device=self.setup["device"])

        # Predetermine forward pass
        def objective_loss(candidate_ids: tuple[int]) -> float:
            """Values can be be cached."""
            inputs = torch.as_tensor(candidate_ids, device=self.setup["device"], dtype=torch.long)[None]
            return sigil.objective(input_ids=inputs, state=state).to(dtype=torch.float32).mean()

        if self.cache_objective and not sigil.is_stochastic:
            objective_loss = lru_cache(maxsize=2**32)(objective_loss)

        # Start iterative process from here
        for iteration in range(self.steps):
            # Optionally freeze objective state
            state = iteration if self.freeze_objective_in_search else None
            # Aggregate gradients
            grads = self.token_gradients(sigil, population, state=state)
            normalized_grads = grads / grads.norm(dim=-1, keepdim=True)

            candidates = []
            batch_size_per_entry = self.batch_size // len(population)
            for grad, pop in zip(normalized_grads, population):  # Sample a few candidates from each entry in the population
                for retry in range(max_retries):
                    if self.anneal_temperature:
                        temp = min(len(sigil), 1.2**best_loss_val)
                    else:
                        temp = 1.0
                    # Sample candidates
                    candidates_entry = self.sample_candidates(pop[None], grad, sigil.constraint, batch_size_per_entry, self.topk, temp)
                    # Filter candidates:
                    candidates_entry, valid_candidates_found = self.get_filtered_cands(candidates_entry, sigil.constraint)
                    if valid_candidates_found:
                        break
                candidates.append(candidates_entry)
            candidates = torch.cat(candidates, dim=0)

            # Search
            loss = torch.zeros(len(candidates), dtype=torch.float32, device=self.setup["device"])
            unique_candidates, uniques_map = torch.unique(candidates, dim=0, return_inverse=True)
            # print(f"Unique fwd passes: {len(unique_candidates)}")
            with torch.no_grad():
                for j, candidate_ids in enumerate(unique_candidates):
                    loss[j == uniques_map] = objective_loss(candidate_ids)

            # Return best from batch:
            best_loss_val, best_loss_loc = torch.min(loss, dim=0)
            best_candidate = candidates[best_loss_loc : best_loss_loc + 1]
            loss_for_best_candidate = best_loss_val
            # Save best result
            if best_loss_val < best_loss_overall:
                best_loss_overall = loss_for_best_candidate.item()
                best_prompt_ids = best_candidate
            # Sample a new population relative to loss differences
            relative_loss_weights = best_loss_val.exp() / loss.exp()
            population = torch.cat(
                [
                    best_prompt_ids,  # elite
                    candidates[torch.multinomial(relative_loss_weights, self.population_size - 1, replacement=True)],  # weighted samples
                ],
                dim=0,
            )

            self.callback(sigil, best_candidate, best_prompt_ids, loss_for_best_candidate.detach(), iteration, **kwargs)
            if dryrun:
                break

        if self.cache_objective and not sigil.is_stochastic:
            print(objective_loss.cache_info())

        return best_prompt_ids  # always return with leading dimension
