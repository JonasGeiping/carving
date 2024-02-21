"""and in darkness bind them? """

import torch
from .generic_sigils import ContextTargetSigil, _GenericSigil
from .utils import ReverseCrossEntropyLoss


class AbjurationSigilForContextTargetSigil(ContextTargetSigil):
    """Replace system prompt with new one that minimizes the effect of any other sigil. Here implemented against ContextTargetSigil.

    <atk_token> is here to be understood more generally as a "token that is optimized". The first abjuration_tokens many atk tokens
    will be optimized to reduce attack sucess, whereas all other atk tokens are optimized to maximize attack success.
    """

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        abjuration_tokens=16,
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)
        self.reverse_loss_fn = ReverseCrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
        self.abjuration_tokens = abjuration_tokens
        assert self.abjuration_tokens < self.num_tokens

        self.target = target
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": f"{tokenizer.atk_token * self.abjuration_tokens}"},
            {
                "role": "user",
                "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * (self.num_tokens - self.abjuration_tokens)}",
            },
            {"role": "assistant", "content": f"{target}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, target)

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        prompt[:, self.context_indices] = self._get_context_ids(batch_size, state)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        # The following is a non-optimal implementation, ideally a nicer framework should be built to do
        # sane min-max optimization, and not what's happening here :>
        # Here were are simply evaling the model loss twice, once with gradients computed for the abjuration tokens and once for the attack tokens

        # 1) Attack Phase
        prompt_embeds[:, self.attack_indices[self.abjuration_tokens :], :] = inputs_embeds[:, self.abjuration_tokens :]
        with torch.no_grad():
            prompt_embeds[:, self.attack_indices[: self.abjuration_tokens], :] = inputs_embeds[:, : self.abjuration_tokens]
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)
        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss_attack = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])

        # 2) Abjuration Phase
        prompt_embeds = self.embedding(prompt_ids)
        prompt_embeds[:, self.attack_indices[: self.abjuration_tokens], :] = inputs_embeds[:, : self.abjuration_tokens]
        with torch.no_grad():
            prompt_embeds[:, self.attack_indices[self.abjuration_tokens :], :] = inputs_embeds[:, self.abjuration_tokens :]
        prompt_embeds, cache, _ = self._maybe_load_cache(prompt_embeds, mask, pos_ids)
        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss_defense = self.reverse_loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss_attack + loss_defense
