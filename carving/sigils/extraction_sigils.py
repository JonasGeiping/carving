import torch

from .generic_sigils import _GenericSigil, ContextTargetSigil
from transformers import LlamaTokenizerFast


class RepeaterSigil(ContextTargetSigil):
    """Varying contexts drawn from a given dataset, try to repeat"""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        fixed_target="",
        fixed_target_end="",
        skip_special_tokens=False,
        custom_sys_prompt="",
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        # Make a message with placeholder tokens:
        full_system_message = f"{tokenizer.default_system_message} {custom_sys_prompt}".rstrip()
        messages = [
            {"role": "system", "content": full_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
        ]
        bare_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not skip_special_tokens:
            repeated_message = bare_prompt.replace(tokenizer.atk_token, "")
        else:
            repeated_message = full_system_message + f"{tokenizer.ctx_token * self.num_context_tokens}"
        messages += [{"role": "assistant", "content": fixed_target + repeated_message + fixed_target_end}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        placeholder_prompt_ids = self.register_prompt_indices(prompt)

        offset = self.num_context_tokens + self.num_tokens  # Search for the repeat only after the offset, this should be made nicer
        target_first_idx = tokenizer(prompt, add_special_tokens=False).char_to_token(
            offset + prompt[offset:].find(fixed_target + repeated_message + fixed_target_end)
        )
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds.repeat(self.batch_size, 1, 1)
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        prompt[:, self.context_indices] = self._get_context_ids(batch_size, state).repeat(1, 2)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt


class SystemRepeaterSigil(RepeaterSigil):
    """Varying contexts drawn from a given dataset and stuffed into the system prompt, try to repeat"""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        fixed_target="",
        fixed_target_end="",
        custom_sys_prompt="",
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {custom_sys_prompt}".rstrip()},
            {"role": "user", "content": f" {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": fixed_target + f"{tokenizer.ctx_token * self.num_context_tokens}" + fixed_target_end},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        placeholder_prompt_ids = self.register_prompt_indices(prompt)

        offset = self.num_context_tokens + self.num_tokens  # Search for the repeat only after the offset, this should be made nicer
        msg = fixed_target + f"{tokenizer.ctx_token * self.num_context_tokens}" + fixed_target_end
        target_first_idx = tokenizer(prompt, add_special_tokens=False).char_to_token(offset + prompt[offset:].find(msg))
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here


class ReverserSigil(ContextTargetSigil):
    """Varying contexts drawn from a given dataset, try to repeat in reverse order"""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        fixed_target="",
        fixed_target_end="",
        skip_special_tokens=False,
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)
        self.skip_special_tokens = skip_special_tokens

        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
        ]
        bare_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not skip_special_tokens:
            repeated_message = bare_prompt.replace(tokenizer.atk_token, "")
        else:
            repeated_message = tokenizer.default_system_message + f"{tokenizer.ctx_token * self.num_context_tokens}"
        first_prompt_ids = tokenizer(repeated_message, add_special_tokens=False)["input_ids"]
        reversed_ids = [val for pair in zip(first_prompt_ids[::-1], [tokenizer.vocab["|"]] * len(first_prompt_ids)) for val in pair]
        reversed_answer = tokenizer.decode(reversed_ids, add_special_tokens=False)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # hack for terrible fast llama tokenizer:
            reversed_answer = reversed_answer.replace("<|", " <|")
        messages += [{"role": "assistant", "content": fixed_target + reversed_answer + fixed_target_end}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")

        placeholder_prompt_ids = self.register_prompt_indices(prompt)
        target_first_idx = tokenizer(prompt, add_special_tokens=False).char_to_token(
            prompt.find(fixed_target + reversed_answer + fixed_target_end)
        )
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here

        if not skip_special_tokens:
            assert len(self.attack_indices) == 2 * self.num_tokens
        assert len(self.context_indices) == 2 * self.num_context_tokens

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        context_ids = self._get_context_ids(batch_size, state)
        prompt[:, self.context_indices] = torch.cat([context_ids, torch.flip(context_ids, (1,))], dim=1)
        if input_ids is not None:
            if self.skip_special_tokens:
                prompt[:, self.attack_indices] = input_ids
            else:
                prompt[:, self.attack_indices] = torch.cat([input_ids, torch.flip(input_ids, (1,))], dim=1)
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        if self.skip_special_tokens:
            prompt_embeds[:, self.attack_indices, :] = inputs_embeds.repeat(self.batch_size, 1, 1)
        else:
            prompt_embeds[:, self.attack_indices, :] = torch.cat([inputs_embeds, torch.flip(inputs_embeds, (1,))], dim=1).repeat(
                self.batch_size, 1, 1
            )

        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss
