"""The optimizer from autodan (reimplementation test, haven't seen the code yet)."""
import torch


from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)
max_retries = 20


class STOOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        batch_size=512,
        temp=1,
        filter_cand=True,
        prelim_selection_weight=3.0,
        fine_selection_weight=100.0,
        freeze_objective_in_search=False,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.batch_size = batch_size
        self.temp = temp
        self.filter_cand = filter_cand

        self.prelim_selection_weight = prelim_selection_weight
        self.fine_selection_weight = fine_selection_weight

        self.freeze_objective_in_search = freeze_objective_in_search

    def token_gradients(self, sigil, input_ids, state=None):
        """
        Computes gradients of the loss with respect to the coordinates.
        Todo: make this more efficient by computing gradients only for embeddings in constraint.set_at_idx
        """
        one_hot = torch.zeros(input_ids.shape[1], sigil.num_embeddings, **self.setup)
        one_hot.scatter_(1, input_ids[0].unsqueeze(1), 1)
        one_hot.requires_grad_()
        inputs_embeds = (one_hot @ sigil.embedding.weight).unsqueeze(0)
        adv_loss = sigil.objective(inputs_embeds=inputs_embeds, state=state, mask_source=input_ids).mean()
        (input_ids_grads,) = torch.autograd.grad(adv_loss, [one_hot])

        return input_ids_grads

    @torch.no_grad()
    def evaluate_logprobs_at_idx(self, prompt_ids, token_idx, sigil):
        # Get logprobs at current position
        full_inputs, _, _ = sigil.make_prompt_with_target(prompt_ids)
        curent_prompt_unpadded = full_inputs[:, : sigil.attack_indices[token_idx]]
        logits = sigil.model(input_ids=curent_prompt_unpadded)["logits"]
        logprobs = torch.nn.functional.log_softmax(logits[0, -1], dim=-1)
        return logprobs

    def single_token_optimization(self, prompt_ids, token_idx, sigil, state=None):
        # Aggregate gradients
        grad = self.token_gradients(sigil, prompt_ids, state)[token_idx]

        logprobs = self.evaluate_logprobs_at_idx(prompt_ids, token_idx, sigil)
        top_indices = sigil.constraint.select_topk(-self.prelim_selection_weight * grad + logprobs, k=self.batch_size)
        if prompt_ids[0, token_idx] not in top_indices:
            top_indices[-1] = prompt_ids[0, token_idx]

        # Form a batch of candidates
        original_input_ids = prompt_ids.repeat(self.batch_size, 1)
        loc_tensor = torch.as_tensor([token_idx] * self.batch_size, device=prompt_ids.device)
        candidates = original_input_ids.scatter_(1, loc_tensor.unsqueeze(-1), top_indices.unsqueeze(-1))
        # Filter candidates:
        if self.filter_cand:
            candidate_is_valid = sigil.constraint.is_tokenization_safe(candidates)
            if sum(candidate_is_valid) > 0:
                candidates = candidates[candidate_is_valid]
                top_indices = top_indices[candidate_is_valid]
            else:
                raise ValueError("todo: implement fallback")

        # Compute objective losses:
        adv_objective = torch.zeros(len(candidates), **self.setup)
        with torch.no_grad():
            for j, candidate_ids in enumerate(candidates):
                adv_objective[j] = sigil.objective(input_ids=candidate_ids[None], state=state).mean()
        # Get full score
        score = -self.fine_selection_weight * adv_objective + logprobs[top_indices]

        # Sample next token from score:
        probabilities = torch.softmax(score / self.temp, dim=-1)
        x = top_indices[torch.multinomial(probabilities, num_samples=1)]
        max_index = torch.argmax(probabilities)
        x_top = top_indices[max_index]
        return x, x_top, adv_objective[max_index]

    def solve(self, sigil, initial_guess=None, dryrun=False, **kwargs):
        """"""
        # Initialize solver
        if initial_guess is None:
            prompt_ids = torch.as_tensor([sigil.tokenizer.pad_token_id for idx in range(len(sigil))], device=self.setup["device"])[None]
        else:
            prompt_ids = torch.as_tensor(initial_guess, device=self.setup["device"])[None]

        for token_idx in range(len(sigil)):
            # Reset search space:
            hypothesis_set = []  # [sigil.tokenizer.pad_token_id]
            converged = False

            # Reset the state_cache for inner-loop freeze optimization.
            sigil._state_cache.clear()

            # Inner Optimization
            iteration = 0
            while not converged:
                state = iteration if self.freeze_objective_in_search else None
                x, x_top, optimal_loss = self.single_token_optimization(prompt_ids, token_idx, sigil, state)
                # Insert new token
                prompt_ids[0, token_idx] = x
                # Check convergence
                if x_top in hypothesis_set:
                    converged = True
                else:
                    hypothesis_set = hypothesis_set + [x_top]
                iteration += 1
                if dryrun:
                    break
            # Project onto valid string (this should almost always be a no-op)
            prompt_ids = sigil.constraint.project_onto_tokenizable_ids(prompt_ids)

            # Callback
            self.callback(sigil, prompt_ids[:, token_idx], None, optimal_loss.detach(), token_idx, **kwargs)

        return prompt_ids  # always return with leading dimension
