import torch
from transformers import LlamaTokenizerFast

from .generic_sigils import ContextTargetSigil, _GenericSigil


class SpecialTokenContextTargetSigil(ContextTargetSigil):
    """Varying contexts drawn from a given dataset. Target is one of the tokenizers special tokens."""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target_token_category,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        _llama2_chat_whitespace_patch=1,
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        self.target = getattr(tokenizer, target_token_category, tokenizer.eos_token)
        assert self.target != tokenizer.pad_token
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{self.target}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if _llama2_chat_whitespace_patch == 1:
                assistant_start = prompt.find("[/INST]")
                prompt = prompt[:assistant_start] + prompt[assistant_start:].replace(" ", "")
            elif _llama2_chat_whitespace_patch == 2:
                self.target = prompt[prompt.find("[/INST]") + len("[/INST]") :]
        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, self.target)

        if _llama2_chat_whitespace_patch != 2:
            target_token_id = getattr(tokenizer, target_token_category + "_id", tokenizer.eos_token_id)
            assert self.prompt_ids[0, self.target_indices].item() == target_token_id


class DividerSigil(ContextTargetSigil):
    """Minimize attention from tokens after the attack to tokens before the attack.

    SYSTEM STUFF <|context|> User: <|attack|> <|context|>
                                  |
                                  Divide here
    """

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        divide_after_attack=False,
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)
        self.divide_after_attack = divide_after_attack

        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * (self.num_context_tokens // 2)} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{tokenizer.ctx_token * (self.num_context_tokens // 2)}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        placeholder_prompt_ids = self.register_prompt_indices(prompt)

        offset = prompt.find(tokenizer.atk_token)
        target_first_idx = tokenizer(prompt, add_special_tokens=False).char_to_token(
            offset + prompt[offset:].find(f"{tokenizer.ctx_token * (self.num_context_tokens // 2)}")
        )
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        attention_probs = self.model(
            inputs_embeds=prompt_embeds,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=pos_ids,
            output_attentions=True,
        )["attentions"]
        loss = 0
        if self.divide_after_attack:
            first_attn_pos = self.attack_indices[-1] - self.last_fixed_token_pos
        else:
            first_attn_pos = self.attack_indices[0] - self.last_fixed_token_pos
        for attention_prob in attention_probs:
            if attention_prob is None:
                raise ValueError("Disable flash attention.")
            # loss with or without square?
            num_heads, seq_len = attention_prob.shape[1], attention_prob.shape[2]
            block_value = attention_prob[:, :, first_attn_pos:, : first_attn_pos + self.last_fixed_token_pos].to(dtype=torch.float32).sum()
            loss += block_value / self.batch_size / num_heads / seq_len**2

        return loss


class MagnetSigil(DividerSigil):
    """Maximize/Minimize attention to attack tokens.

    SYSTEM STUFF <|context|> User:       <|attack|>          <|context|>
                                    |                  |
                                      Attend here only
    """

    def __init__(
        self, model, tokenizer, aux_models, *args, orientation=True, context_from_dataset=dict(num_tokens=16, source=[]), **kwargs
    ):
        super().__init__(model, tokenizer, aux_models, *args, context_from_dataset=context_from_dataset, **kwargs)
        self.orientation = float(orientation)

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        attention_probs = self.model(
            inputs_embeds=prompt_embeds,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=pos_ids,
            output_attentions=True,
        )["attentions"]
        loss = 0
        for attention_prob in attention_probs:
            if attention_prob is None:
                raise ValueError("Disable flash attention.")
            num_heads, seq_len = attention_prob.shape[1], attention_prob.shape[2]
            target_block = attention_prob[:, :, self.attack_indices - self.last_fixed_token_pos][:, :, :, self.attack_indices]
            loss -= self.orientation * target_block.to(dtype=torch.float32).sum() / self.batch_size / num_heads / seq_len**2

        return loss
