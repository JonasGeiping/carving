import torch
from operator import attrgetter

from .generic_sigils import _GenericSigil, ContextTargetSigil
from .extraction_sigils import RepeaterSigil

from ..data_utils import load_and_prep_dataset, get_data_iterators


class FixedAttackRepeaterSigil(_GenericSigil):
    """Optimize the model to repeat the attack as much as possible."""

    def __init__(self, model, tokenizer, aux_models, *args, context="", targeted_repeats=4, glider_gun_length=0, **kwargs):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self.targeted_repeats = targeted_repeats  # could/should be randomized?
        self.glider_gun_length = glider_gun_length
        assert self.glider_gun_length < self.num_tokens

        self.glider_size = self.num_tokens - self.glider_gun_length
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{context} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{tokenizer.atk_token * self.glider_size * self.targeted_repeats}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        placeholder_prompt_ids = self.register_prompt_indices(prompt)

        target_first_idx = self.attack_indices[self.num_tokens]  # Works here because the n-th atk token should be the first answer
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here
        assert len(self.attack_indices) == self.num_tokens + self.glider_size * self.targeted_repeats

    def make_prompt_with_target(self, input_ids, state=None):
        prompt = self.prompt_ids.clone()
        if input_ids is not None:
            if self.glider_gun_length == 0:
                prompt[:, self.attack_indices] = input_ids.repeat(1, self.targeted_repeats + 1)
            else:
                prompt[:, self.attack_indices[: self.num_tokens]] = input_ids
                prompt[:, self.attack_indices[self.num_tokens :]] = input_ids[:, -self.glider_size :].repeat(1, self.targeted_repeats)
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        if self.glider_gun_length == 0:
            prompt_embeds[:, self.attack_indices, :] = inputs_embeds.repeat(1, self.targeted_repeats + 1, 1)
        else:
            prompt_embeds[:, self.attack_indices[: self.num_tokens], :] = inputs_embeds
            gliders = inputs_embeds[:, -self.glider_size :, :].repeat(1, self.targeted_repeats, 1)
            prompt_embeds[:, self.attack_indices[self.num_tokens :], :] = gliders

        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss

    def _maybe_create_mask(self, attack_token_mask, prompt_ids, dtype):
        mask = prompt_ids != self.tokenizer.pad_token_id
        if attack_token_mask is not None:
            if self.glider_gun_length == 0:
                mask[:, self.attack_indices] = attack_token_mask.repeat(1, self.targeted_repeats + 1)
            else:
                mask[:, self.attack_indices[: self.num_tokens]] = attack_token_mask
                mask[:, self.attack_indices[self.num_tokens :]] = attack_token_mask[:, -self.glider_size :].repeat(1, self.targeted_repeats)

        pos_ids = (torch.cumsum(mask, dim=-1) - 1).long()
        if self.randomize_ar_fraction is not None and self.randomize_ar_fraction > 0:
            bsz, seq_len = mask.shape

            expanded_mask = mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = torch.bernoulli((expanded_mask) * (1 - self.randomize_ar_fraction))
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min), pos_ids
        else:
            return mask, pos_ids


class AttackRepeaterSigil(RepeaterSigil):
    """Varying contexts drawn from a given dataset. The model should ignore the context and repeat the attack as much as possible.
    This is an attempt at a Game of Life primitive."""

    def __init__(
        self, model, tokenizer, aux_models, *args, context_from_dataset=dict(num_tokens=16, source=[]), targeted_repeats=4, **kwargs
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)
        self.targeted_repeats = targeted_repeats  # could/should be randomized?

        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{tokenizer.atk_token * self.num_tokens * self.targeted_repeats}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        placeholder_prompt_ids = self.register_prompt_indices(prompt)

        target_first_idx = self.attack_indices[self.num_tokens]  # Works here because the n-th atk token should be the first answer
        self.register_buffer("target_indices", torch.arange(target_first_idx, len(placeholder_prompt_ids)), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here
        assert len(self.attack_indices) == self.num_tokens + self.num_tokens * self.targeted_repeats

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        prompt[:, self.context_indices] = self._get_context_ids(batch_size, state)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids.repeat(1, self.targeted_repeats + 1)
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds.repeat(self.batch_size, self.targeted_repeats + 1, 1)
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss

    def _maybe_create_mask(self, attack_token_mask, prompt_ids, dtype):
        return super()._maybe_create_mask(attack_token_mask.repeat(1, self.targeted_repeats + 1), prompt_ids, dtype)


class FixedNaNSigil(_GenericSigil):
    def __init__(self, model, tokenizer, aux_models, *args, target_objective="mean", **kwargs):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self.target_objective = target_objective

        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": " "},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_buffer("target_ids", torch.as_tensor(0).view(-1, 1), persistent=False)  # placeholder

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        B, S, H = inputs_embeds.shape
        embeddings = self.embedding(self.prompt_ids).detach()
        embeddings[:, self.attack_indices, :] = inputs_embeds
        mask, pos_ids = self._maybe_create_mask(mask, embeddings)
        embeddings, cache, pos_ids = self._maybe_load_cache(embeddings, mask, pos_ids)

        target_logits = self.model(inputs_embeds=embeddings, attention_mask=mask, past_key_values=cache)["logits"].to(torch.float32)

        if self.target_objective == "mean":
            loss = -target_logits.mean()
        elif self.target_objective == "norm":
            loss = -target_logits.norm()
        elif self.target_objective == "max-spread":
            loss = target_logits.min() - target_logits.max()
        elif self.target_objective == "max-val":
            loss = -target_logits.max()
        elif self.target_objective == "soft-spread":
            loss = torch.logsumexp(-target_logits, dim=-1) - torch.logsumexp(target_logits, dim=-1)
        elif self.target_objective == "soft-max":
            loss = -torch.logsumexp(target_logits, dim=-1)
        else:
            raise ValueError("Invalid target objective")

        loss = -target_logits.mean()
        return loss

    def make_prompt_with_target(self, input_ids, state=None):
        prompt = self.prompt_ids.clone()
        prompt[:, self.attack_indices] = input_ids
        return prompt[:, :-1], self.target_ids, prompt


class FixedActNaNSigil(_GenericSigil):
    """Works only for llama-like models right now! Make it more general if the hook misses your target."""

    def __init__(
        self, model, tokenizer, aux_models, *args, target_layer=24, target_module="mlp.down_proj", target_objective="max-sum", **kwargs
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": " "},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_buffer("target_ids", torch.as_tensor(0).view(-1, 1), persistent=False)  # placeholder

        self.target_layer = target_layer
        self.target_module = target_module
        self.target_objective = target_objective

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        B, S, H = inputs_embeds.shape
        embeddings = self.embedding(self.prompt_ids).detach()
        embeddings[:, self.attack_indices, :] = inputs_embeds
        mask, pos_ids = self._maybe_create_mask(mask, embeddings)
        embeddings, cache, pos_ids = self._maybe_load_cache(embeddings, mask, pos_ids)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.to(torch.float32)

            return hook

        tmp_hook = attrgetter(self.target_module)(self.model.model.layers[self.target_layer]).register_forward_hook(get_activation("act"))
        self.model(inputs_embeds=embeddings, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)

        tmp_hook.remove()
        if self.target_objective == "max-sum":
            loss = -activation["act"].mean()
        elif self.target_objective == "max-norm":
            loss = -activation["act"].norm()
        elif self.target_objective == "max-spread":
            loss = activation["act"].min() - activation["act"].max()
        elif self.target_objective == "max-val":
            loss = -activation["act"].max()
        elif self.target_objective == "soft-spread":
            loss = torch.logsumexp(-activation["act"], dim=-1) - torch.logsumexp(activation["act"], dim=-1)
        elif self.target_objective == "soft-max":
            loss = -torch.logsumexp(activation["act"], dim=-1)
        else:
            raise ValueError("Invalid target objective")

        return loss

    def make_prompt_with_target(self, input_ids, state=None):
        prompt = self.prompt_ids.clone()
        prompt[:, self.attack_indices] = input_ids
        return prompt[:, :-1], self.target_ids, prompt


class DDOSSigil(ContextTargetSigil):
    """Varying contexts drawn from a given dataset."""

    def __init__(self, model, tokenizer, aux_models, *args, context_from_dataset=dict(num_tokens=16, source=[]), **kwargs):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset, num_workers=0)
        self.prompt_cutoff = context_from_dataset["eval_cutoff"]

        assert tokenizer.pad_token_id != tokenizer.eos_token_id

        # Make a message with placeholder tokens:
        target_prompt = f"{tokenizer.ctx_token * (self.num_context_tokens - self.prompt_cutoff)}"
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.atk_token * self.num_tokens} {tokenizer.ctx_token * self.prompt_cutoff}"},
            {"role": "assistant", "content": target_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        # Start measuring on the 2nd block
        self.register_target_indices(prompt, target_prompt)
        self.target_ids = self.target_ids.fill_(tokenizer.eos_token_id).repeat(self.batch_size, 1)

    def _prepare_context_dataloader(self, tokenizer, context_from_dataset, num_workers=0):
        self.concat_examples = context_from_dataset["concatenate_examples"]
        self.num_context_tokens = context_from_dataset["num_tokens"]
        split = context_from_dataset.get("split", "train")

        self.data_source = load_and_prep_dataset(
            tokenizer,
            context_from_dataset["source"],
            fixed_length=self.num_context_tokens // self.concat_examples,
            split=split,
            cache_dir=self.cache_dir,
        )
        self.batch_size = context_from_dataset["batch_size"] // self.concat_examples
        self.data_iterator, self.data_iterator_holdout = get_data_iterators(
            self.data_source, context_from_dataset.get("holdout_size", 0.1), context_from_dataset["batch_size"], num_workers
        )

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds.repeat(self.batch_size, 1, 1)
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], self.target_ids)
        return loss

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        prompt[:, self.context_indices] = self._get_context_ids(batch_size * self.concat_examples, 56).view(batch_size, -1)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt
