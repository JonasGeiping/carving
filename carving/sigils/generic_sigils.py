"""Defines details for the construction of sigils, translating the task into math. notation of MINLP optimization problems."""

"""Design doc for now:
Channel simplicity people and prevent excessive inheritance and OOP, allow some code duplication.
Maybe there is a more elegant implementation that is both easier to read and easier to write, but let's figure that out once
we see all planned sigils."""

import torch
import transformers


from transformers import LlamaTokenizerFast, LlamaForCausalLM

# from transformers.cache_utils import DynamicCache # as of 01/04 broken in combination with reentrant=False checkpointing
from torch.nn.functional import log_softmax

from ..model_interface import retrieve_embedding
from .utils import (
    ReverseCrossEntropyLoss,
    OneDimCrossEntropyLoss,
    MaxCrossEntropyLoss,
    LSECrossEntropyLoss,
    ReverseLSECrossEntropyLoss,
    hash_args,
)
from .constraints import get_constraint

from ..data_utils import load_and_prep_dataset, get_data_iterators


class _GenericSigil(torch.nn.Module):
    _target_tensors = []

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        objective="xent",
        num_tokens=8,
        constraint="ascii",
        constraint_blocklist=[],
        cache_dir="~/data",
        natural_prompt="",
        randomize_ar_fraction=None,
        **kwargs,
    ):
        super().__init__()
        self.uid = self._set_argument_uid(locals())
        self.cache_dir = cache_dir
        self.randomize_ar_fraction = randomize_ar_fraction
        if self.randomize_ar_fraction:
            self.model = _patch_attention_maps_to_allow_4d_attention(model)
        else:
            self.model = model

        self.aux_models = torch.nn.ModuleList(aux_models)
        self.embedding = retrieve_embedding(model)

        self.constraint = get_constraint(constraint, tokenizer, self.embedding, num_tokens, blocklist=constraint_blocklist)
        if len(self.constraint) == 0:
            raise ValueError("Constraint set too restrictive for the given tokenizer. No valid tokens could be identified.")
        else:
            print(f"Optimizing over constraint set with {len(self.constraint)} tokens.")
        self.tokenizer = _add_placeholder_tokens(tokenizer)

        if objective == "reverse-xent":
            self.loss_fn = ReverseCrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            self.maximize = True  # for eval
        elif objective == "max-xent":
            # minimize maximal cross entropy across the sequence length
            self.loss_fn = MaxCrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            self.maximize = False
        elif objective == "lse-xent":
            # minimize soft maximal cross entropy across the sequence length
            self.loss_fn = LSECrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            self.maximize = False
        elif objective == "rlse-xent":
            # minimize soft maximal cross entropy across the sequence length
            self.loss_fn = ReverseLSECrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            self.maximize = True
        else:
            self.loss_fn = OneDimCrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
            self.maximize = False

        self.num_tokens = num_tokens
        self.num_embeddings = self.embedding.num_embeddings
        self.natural_prompt = natural_prompt
        self._cache = None
        self._state_cache = dict()

    def _set_argument_uid(self, args_and_kwargs):
        """Return hash of own arguments."""
        return hash_args(args_and_kwargs)

    def objective(self, inputs_embeds=None, input_ids=None, state=None, mask_source=None):
        if mask_source is not None:
            mask = mask_source != self.tokenizer.pad_token_id
        elif input_ids is not None:
            mask = input_ids != self.tokenizer.pad_token_id
        else:
            mask = None
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return self._objective_impl(inputs_embeds, state=state, mask=mask)

    @property
    def is_stochastic(self):
        return False

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        """Implement the target objective here. The mask should be passed to the model. The state flag determines whether the objective
        is supposed to be stateful. A given state, e.g. state=5 should return the objective in state=5, even if the objective is randomized.
        """
        raise NotImplementedError()

    def make_prompt_with_target(self, input_ids, state=None):
        """This method needs to return a tuple of [prompt_given_to_a_model, expected_completion, full_prompt_plus_completion]."""
        raise NotImplementedError()

    def _maybe_create_mask(self, attack_token_mask, inputs_embeds):
        bsz, seq_len, _ = inputs_embeds.shape
        mask = None
        if attack_token_mask is not None or self.randomize_ar_fraction is not None:
            mask = inputs_embeds.new_ones((bsz, seq_len), dtype=torch.bool)
            mask[:, self.attack_indices] = attack_token_mask
            pos_ids = (torch.cumsum(mask, dim=-1) - 1).long()
        else:
            pos_ids = None  # use defaults

        if self.randomize_ar_fraction is not None and self.randomize_ar_fraction > 0:
            # inverted_mask rules:
            # ones will be masked, zeros will be attended to
            inverted_mask = torch.as_tensor(~mask[:, None, None, :], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inverted_mask = inputs_embeds.new_ones((bsz, seq_len))[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
            inverted_mask = torch.bernoulli(inverted_mask * self.randomize_ar_fraction)
            # Never mask attack indices
            inverted_mask[0, 0, self.attack_indices.unsqueeze(0), self.attack_indices.unsqueeze(1)] = 0
            # Never mask the first token in the sequence
            inverted_mask[0, 0, :, 0] = 0
            # Never mask the token attending to itself
            inverted_mask[0, 0, torch.arange(inverted_mask.shape[2]), torch.arange(inverted_mask.shape[2])] = 0
            # torch.diagonal(inverted_mask, dim1=2, dim2=3) = 0
            # Causal masking happens elsewhere in the code for llama, so it's not neccesary here
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min), pos_ids
        else:
            return mask, pos_ids

    def __len__(self):
        return self.num_tokens

    @torch.no_grad()
    def to(self, *args, **kwargs):
        """Overwrite for device mapping. Do nothing special if num_gpus=1. Otherwise, assume all input-like things should be on gpu0 and all
        output-like things should be on GPU n"""
        if torch.cuda.device_count() > 1 and hasattr(self.model, "hf_device_map"):
            print(self.model.hf_device_map)
            source_device = torch.device("cuda:0")
            # target_device = torch.device(f"cuda:{torch.cuda.device_count()-1}")

            # all buffers and parameters are presumed to be source if not included in the const _target_tensors
            for name, buffer in self.named_buffers():
                if "model" in name:
                    pass  # handled by device-map
                elif buffer.is_nested:
                    pass  # this is terrible, has to be hardcoded for now
                else:
                    buffer.data = buffer.data.to(dtype=kwargs["dtype"] if buffer.data.is_floating_point() else None, device=source_device)
            for name, param in self.named_parameters():
                pass  # handled by device-map
            if hasattr(self.constraint, "nested_set"):
                # Special rule because buffer.data = ... does not work for nested tensors
                self.constraint.nested_set = self.constraint.nested_set.to(source_device)
        else:
            super().to(*args, **kwargs)

    def register_prompt_indices(self, prompt):
        """Figure out all indices and set them as buffers, so they can be moved to appropriate devices later."""
        placeholder_prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]  # special toks added by chat template

        prompt_ids = torch.tensor(placeholder_prompt_ids)[None]
        prompt_ids[prompt_ids > self.num_embeddings - 1] = self.tokenizer.vocab["!"]  # random id that is not pad. Never used as input!
        self.register_buffer("prompt_ids", prompt_ids, persistent=False)

        attack_indices = torch.nonzero(torch.tensor([p == self.tokenizer.atk_token_id for p in placeholder_prompt_ids])).squeeze(dim=-1)
        context_indices = torch.nonzero(torch.tensor([p == self.tokenizer.ctx_token_id for p in placeholder_prompt_ids])).squeeze(dim=-1)
        self.register_buffer("context_indices", context_indices, persistent=False)
        self.register_buffer("attack_indices", attack_indices, persistent=False)

        if self.model.attempt_to_cache_fixed_tokens:
            self.last_fixed_token_pos = self.attack_indices.min()
            if len(self.context_indices) > 0:
                self.last_fixed_token_pos = min(self.last_fixed_token_pos, self.context_indices.min())
        else:
            self.last_fixed_token_pos = 0

        return placeholder_prompt_ids

    def register_target_indices(self, prompt, target=None, use_target_token_ids=False):
        tokenized_prompt = self.tokenizer(prompt, add_special_tokens=False)  # special toks already added by chat template
        placeholder_prompt_ids = tokenized_prompt["input_ids"]
        if target is not None:
            if prompt.find(target) < 0:
                raise ValueError(
                    f"Cannot identify target string {target} in provided prompt {prompt}." f"This indicates that the sigil code has a bug."
                )
            target_first_idx = tokenized_prompt.char_to_token(prompt.find(target))
            target_last_idx = tokenized_prompt.char_to_token(prompt.find(target) + len(target) - 1) + 1
            self.register_buffer("target_indices", torch.arange(target_first_idx, target_last_idx), persistent=False)
            self.register_buffer("loss_indices", torch.arange(target_first_idx - 1, target_last_idx - 1), persistent=False)

            # After all of this, we should have the target ids exactly in the target indices locations in the prompt!
            # target_ids = tokenizer(model.template.sep + target, add_special_tokens=False)["input_ids"]
            # assert self.prompt_ids[0, self.target_indices].tolist() == target_ids
            self.register_buffer("target_ids", torch.as_tensor(self.prompt_ids[:, self.target_indices]), persistent=False)
        elif use_target_token_ids:
            target_indices = torch.nonzero(torch.tensor([p == self.tokenizer.tgt_token_id for p in placeholder_prompt_ids])).squeeze(dim=-1)
            self.register_buffer("target_indices", target_indices, persistent=False)
            self.register_buffer("loss_indices", target_indices - 1, persistent=False)
        else:
            raise ValueError("Provide either a target string or target token")

    def _maybe_load_cache(self, embeddings, mask, pos_ids):
        if self.model.attempt_to_cache_fixed_tokens:
            if self._cache is None:
                # Build cache:
                with torch.no_grad():
                    self.model.eval()  # disable grad checkpointing
                    # checkpointing shouldn't be tied to the model's .training attribute, but apparently it is
                    past_key_values = self.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=True)["past_key_values"]
                    self.model.train()  # enable grad checkpointing
                # Cut past_kv_values to size:
                past_key_values = tuple(
                    tuple(block[:, :, : self.last_fixed_token_pos, :].detach() for block in layer) for layer in past_key_values
                )
                self._cache = past_key_values  # = DynamicCache.from_legacy_cache(past_key_values) # broken in combination with cp
                # Correct all indices involved in the objective calcuation to new shifted positions
                if hasattr(self, "loss_indices"):
                    self.loss_indices = self.loss_indices - self.last_fixed_token_pos if self.loss_indices is not None else None
            cache = self._cache
            embeddings = embeddings[:, self.last_fixed_token_pos :, :]
            if pos_ids is not None:
                pos_ids = pos_ids[:, self.last_fixed_token_pos :]
            return embeddings, cache, pos_ids
        else:
            return embeddings, None, pos_ids


class FixedTargetSigil(_GenericSigil):
    _target_tensors = ["target_indices", "loss_indices"]

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        context,
        target,
        *args,
        post_context="",
        custom_sys_prompt=None,
        _progressive_expansion=False,
        **kwargs,
    ):
        super().__init__(model, tokenizer, aux_models, *args, **kwargs)
        self.context = context
        self.target = target.rstrip()  # strip trailing \n which tokenizer.apply_chat_template does not handle well

        # this flag is only implemented manually for this sigil, because it's a last-minute hack and a headache
        # can only be activated by optimizer modifiying sigil attributes
        self.progressive_expansion = False
        self.expansion_lookup = dict()
        self.target_length = 1

        messages = [
            {"role": "system", "content": tokenizer.default_system_message if custom_sys_prompt is None else custom_sys_prompt},
            {"role": "user", "content": f"{context} {tokenizer.atk_token * self.num_tokens} {post_context}"},
            {"role": "assistant", "content": f"{self.target}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")

        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, self.target)
        # Check that all attack token locations are correctly identified
        assert len(self.attack_indices) == self.num_tokens
        # on llama-2, assert shapes like this:
        # assert tokenizer.decode(self.prompt_ids[0, 1:]) == prompt.replace(" <|attack|>", "!")
        # assert tokenizer(target)["input_ids"][1:] == self.target_ids[0].tolist()

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        B, S, H = inputs_embeds.shape
        embeddings = self.embedding(self.prompt_ids).detach()
        embeddings[:, self.attack_indices, :] = inputs_embeds
        mask, pos_ids = self._maybe_create_mask(mask, embeddings)
        embeddings, cache, pos_ids = self._maybe_load_cache(embeddings, mask, pos_ids)

        if not self.progressive_expansion or "eval" in str(state):  # business as usual:
            target_logits = self.model(inputs_embeds=embeddings, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
            loss = self.loss_fn(target_logits[:, self.loss_indices], self.target_ids)
        else:
            loss = self._progressive_expansion_objective(embeddings, mask, cache, pos_ids, state)

        return loss

    def _progressive_expansion_objective(self, embeddings, mask, cache, pos_ids, state):
        if state not in self.expansion_lookup:
            if "expand" in str(state) and self.target_length < self.target_ids.shape[1]:
                self.target_length += 1
            self.expansion_lookup[state] = self.target_length
        t = self.expansion_lookup[state]
        target_logits = self.model(
            inputs_embeds=embeddings[:, : -(self.target_ids.shape[1] - t)],
            attention_mask=mask[:, : -(self.target_ids.shape[1] - t)],
            past_key_values=cache,
            position_ids=pos_ids[:, : -(self.target_ids.shape[1] - t)],
        )["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices[:t]], self.target_ids[:, :t])
        return loss

    def make_prompt_with_target(self, input_ids, state=None):
        prompt = self.prompt_ids.clone()
        prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt


class FixedCollisionSigil(_GenericSigil):
    """Collide with a target and produce identical probabilities on a fixed eval context.

    LAYOUT:
        attack_seq:         [SYS]         [Fixed Context]         [attack tokens  + <pad>]     [eval context]
        target_seq:         [SYS]         [Fixed Context]         [target meaning + <pad>]     [eval context]
        role      :System:        User:                                              Assistant:
        cached    :    |||||||||||||||||||||||||||||||||||||||
        loss_computed:                                                                         |||||||||||||||||
        Example   :    [llama2 blabla] [Please tell me a joke] [about Sam Altman <pad> <pad>] [Ok, so the joke goes as follows:]
    """

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target_meaning,
        *args,
        target_objective="forward-KL",
        force_eval_context=False,
        pre_context="",
        post_context="",
        eval_context="",
        **kwargs,
    ):
        super().__init__(model, tokenizer, aux_models, *args, **kwargs)
        self.target_meaning = target_meaning

        self.target_objective = target_objective
        self.force_eval_context = force_eval_context

        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{pre_context} {tokenizer.atk_token * self.num_tokens} {post_context}"},
            {"role": "assistant", "content": f"{eval_context}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, eval_context)

        # Handle target:
        assert len(self.tokenizer(target_meaning, add_special_tokens=False)["input_ids"]) <= self.num_tokens
        self.register_buffer(
            "target_meaning_ids",
            torch.as_tensor(
                self.tokenizer(target_meaning, add_special_tokens=False, padding="max_length", max_length=self.num_tokens)["input_ids"]
            ),
        )
        self.register_buffer("target_log_probs", torch.tensor([]), persistent=False)

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        B, S, H = inputs_embeds.shape
        embeddings = self.embedding(self.prompt_ids).detach()
        embeddings[:, self.attack_indices, :] = inputs_embeds
        mask, pos_ids = self._maybe_create_mask(mask, embeddings)
        embeddings, cache, pos_ids = self._maybe_load_cache(embeddings, mask, pos_ids)

        if len(self.target_log_probs) == 0:
            with torch.no_grad():
                # cache target log probs
                target_prompt_ids = self.prompt_ids.clone()
                target_prompt_ids[:, self.attack_indices] = self.target_meaning_ids
                target_mask = target_prompt_ids != self.tokenizer.pad_token_id
                target_prompt_ids = target_prompt_ids[:, self.last_fixed_token_pos :]  # need to handle this separately to use the cache

                self.target_logits = self.model(
                    input_ids=target_prompt_ids,
                    attention_mask=target_mask,
                    past_key_values=cache,
                    position_ids=(torch.cumsum(target_mask, dim=-1) - 1).long()[:, self.last_fixed_token_pos :],
                )["logits"]
                self.target_log_probs = log_softmax(self.target_logits[:, self.loss_indices, :], dim=-1)

        logits = self.model(inputs_embeds=embeddings, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        log_probs = log_softmax(logits[:, self.loss_indices], dim=-1)

        if self.target_objective == "forward-KL":
            loss = torch.nn.functional.kl_div(log_probs, self.target_log_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "backward-KL":
            loss = torch.nn.functional.kl_div(self.target_log_probs, log_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "log-logit":
            loss = (log_probs - self.target_log_probs).pow(2).mean()
        elif self.target_objective == "direct-logit":
            loss = (logits - self.target_logits).pow(2).mean()
        else:
            raise ValueError("Invalid target objective")

        if self.force_eval_context:
            loss += self.force_eval_context * self.loss_fn(logits[:, self.loss_indices], self.target_ids)
        return loss

    def make_prompt_with_target(self, input_ids, state=None):
        prompt = self.prompt_ids.clone()
        prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt


class ContextTargetSigil(_GenericSigil):
    """Varying contexts drawn from a given dataset."""

    def __init__(self, model, tokenizer, aux_models, target, *args, context_from_dataset=dict(num_tokens=16, source=[]), **kwargs):
        super().__init__(model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        self.target = target
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{target}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, target)

    @property
    def is_stochastic(self):
        return True

    def _prepare_context_dataloader(self, tokenizer, context_from_dataset, num_workers=0):
        self.num_context_tokens = context_from_dataset["num_tokens"]
        split = context_from_dataset.get("split", "train")
        self.data_source = load_and_prep_dataset(
            tokenizer, context_from_dataset["source"], fixed_length=self.num_context_tokens, split=split, cache_dir=self.cache_dir
        )
        self.batch_size = context_from_dataset["batch_size"]
        self.data_iterator, self.data_iterator_holdout = get_data_iterators(
            self.data_source, context_from_dataset.get("holdout_size", 0.1), self.batch_size, num_workers
        )

    def _get_context_ids(self, batch_size, state=None):
        iterator = self.data_iterator_holdout if state is not None and "eval" in str(state) else self.data_iterator
        if state is None:
            context_batch = torch.stack(next(iterator)["input_ids"], dim=1)
            return context_batch.to(self.context_indices.device)[:batch_size]
        else:
            if state not in self._state_cache:
                context_batch = torch.stack(next(iterator)["input_ids"], dim=1)
                self._state_cache[state] = context_batch.to(self.context_indices.device)
            return self._state_cache[state][:batch_size]

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
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        target_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        loss = self.loss_fn(target_logits[:, self.loss_indices], prompt_ids[:, self.target_indices])
        return loss

    def _maybe_create_mask(self, attack_token_mask, prompt_ids, dtype):
        mask = prompt_ids != self.tokenizer.pad_token_id
        if attack_token_mask is not None:
            mask[:, self.attack_indices] = attack_token_mask
        pos_ids = (torch.cumsum(mask, dim=-1) - 1).long()
        if self.randomize_ar_fraction is not None and self.randomize_ar_fraction > 0:
            bsz, seq_len = mask.shape
            # inverted_mask rules:
            # ones will be masked, zeros will be attended to
            inverted_mask = torch.as_tensor(~mask[:, None, None, :], dtype=dtype, device=prompt_ids.device)
            inverted_mask[inverted_mask == 0] = self.randomize_ar_fraction
            inverted_mask = torch.bernoulli(inverted_mask.expand(bsz, 1, seq_len, seq_len))  # sample new masks with prob ar_fraction
            # Never mask attack indices
            inverted_mask[:, 0, self.attack_indices.unsqueeze(0), self.attack_indices.unsqueeze(1)] = 0
            # Never mask the first token in the sequence
            inverted_mask[:, 0, :, 0] = 0
            # Never mask the token attending to itself
            inverted_mask[:, 0, torch.arange(inverted_mask.shape[2]), torch.arange(inverted_mask.shape[2])] = (~mask).to(dtype)
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min).to(dtype), pos_ids
        else:
            return mask, pos_ids


class ContextMultipleTargetsSigil(ContextTargetSigil):
    """Varying contexts drawn from a given dataset."""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        one_context_against_all_targets=False,
        draw_targets_without_replacement=False,
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        self.target = target  # This is now a list
        self.one_context_against_all_targets = one_context_against_all_targets
        self.draw_targets_without_replacement = draw_targets_without_replacement
        if draw_targets_without_replacement:
            self.batch_size = min(self.batch_size, len(target))
        self.register_buffer(
            "tokenized_targets", self.tokenizer(list(target), add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
        )
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{tokenizer.tgt_token * self.tokenized_targets.shape[1]}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, use_target_token_ids=True)
        self._target_cache = dict()

    def _get_target_ids(self, batch_size, state=None):
        if state is None:
            if not self.draw_targets_without_replacement:
                indices = torch.randint(0, self.tokenized_targets.shape[0], (batch_size,))
            else:
                indices = torch.randperm(self.tokenized_targets.shape[0])[:batch_size]
            return self.tokenized_targets[indices]
        else:
            if state not in self._target_cache:
                if not self.draw_targets_without_replacement:
                    indices = torch.randint(0, self.tokenized_targets.shape[0], (batch_size,))
                else:
                    indices = torch.randperm(self.tokenized_targets.shape[0])[:batch_size]
                self._target_cache[state] = self.tokenized_targets[indices]
            return self._target_cache[state]

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        if self.one_context_against_all_targets:
            prompt[:, self.context_indices] = self._get_context_ids(1, state).repeat(batch_size, 1)
        else:
            prompt[:, self.context_indices] = self._get_context_ids(batch_size, state)
        prompt[:, self.target_indices] = self._get_target_ids(batch_size, state)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt


class ContextCollisionSigil(ContextTargetSigil):
    """Given some sampled context, collide with a target and produce identical probabilities on either a fixed or sampled eval context.

    LAYOUT (fixed eval):
        attack_seq:         [SYS]         [Context from Data+<pad>]         [attack tokens  + <pad>]     [fixed eval context]
        target_seq:         [SYS]         [Context from Data+<pad>]         [target meaning + <pad>]     [fixed eval context]
        role      :System:        User:                                                    Assistant:
        cached    :    ||||||||||||||||||
        loss_computed:                                                                                   |||||||||||||||||
        Example   :    [llama2 blabla]    [random instruction]              [</s><s> <pad> <pad>]        [My instruction is to]


    LAYOUT (sampled eval):
        attack_seq:         [SYS]         [Context from Data(first n)]       [attack tokens  + <pad>]    [Context from Data(continued)]
        target_seq:         [SYS]         [Context from Data(first n)]       [target meaning + <pad>]    [Context from Data(continued)]
        role      :System:        User:                                                    Assistant:
        cached    :    ||||||||||||||||||
        loss_computed:                                                                                   |||||||||||||||||
        Example   :    [llama2 blabla]    [random wikipedia]                 [</s><s> <pad> <pad>]       [random wikipedia continued]
    """

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target_meaning,
        fixed_eval_context,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        target_objective="forward-KL",
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)
        self.prompt_cutoff = context_from_dataset["eval_cutoff"]
        assert (self.prompt_cutoff < self.num_context_tokens) or len(fixed_eval_context) > 1
        self.target_meaning = target_meaning
        self.target_objective = target_objective

        # Make message with placeholder strings:
        target_prompt = f"{fixed_eval_context}{tokenizer.ctx_token * (self.num_context_tokens - self.prompt_cutoff)}"
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.prompt_cutoff} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": target_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, target_prompt)

        # Target handling:
        assert len(self.tokenizer(target_meaning, add_special_tokens=False)["input_ids"]) < self.num_tokens
        self.register_buffer(
            "target_meaning_ids",
            torch.as_tensor(
                self.tokenizer(target_meaning, add_special_tokens=False, padding="max_length", max_length=self.num_tokens)["input_ids"]
            ),
        )

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        with torch.no_grad():
            target_prompt_ids = prompt_ids.clone()
            target_prompt_ids[:, self.attack_indices] = self.target_meaning_ids
            target_mask = target_prompt_ids != self.tokenizer.pad_token_id
            target_prompt_ids = target_prompt_ids[:, self.last_fixed_token_pos :]  # need to handle this separately to use the cache

            target_logits = self.model(
                input_ids=target_prompt_ids,
                attention_mask=target_mask,
                past_key_values=cache,
                position_ids=(torch.cumsum(target_mask, dim=-1) - 1).long()[:, self.last_fixed_token_pos :],
            )["logits"]
            target_probs = log_softmax(target_logits[:, self.loss_indices], dim=-1)

        attack_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache, position_ids=pos_ids)["logits"]
        attack_probs = log_softmax(attack_logits[:, self.loss_indices], dim=-1)

        if self.target_objective == "forward-KL":
            loss = torch.nn.functional.kl_div(attack_probs, target_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "backward-KL":
            loss = torch.nn.functional.kl_div(target_probs, attack_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "log-logit":
            loss = (attack_probs - target_probs).pow(2).mean()
        elif self.target_objective == "direct-logit":
            loss = (attack_logits - target_logits).pow(2).mean()
        else:
            raise ValueError("Invalid target objective")

        return loss

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        prompt[:, self.context_indices] = self._get_context_ids(batch_size, state)
        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt


def _add_placeholder_tokens(tokenizer):
    # Add extra tokens to mark out [context] and [target] without collisions
    # if not hasattr(tokenizer, "vocab_files_names"):  # hacky way to identify the qwen tokenizer
    tokenizer.atk_token = "<|attack|>"
    tokenizer.ctx_token = "<|context|>"
    tokenizer.tgt_token = "<|target|>"
    tokenizer.add_tokens([tokenizer.atk_token, tokenizer.ctx_token, tokenizer.tgt_token])
    # else:
    #     #  exception for qwen tokenizer:
    #     tokenizer.atk_token = "<|extra_5|>"
    #     tokenizer.ctx_token = "<|extra_6|>"
    #     tokenizer.tgt_token = "<|extra_7|>"

    tokenizer.atk_token_id = tokenizer(tokenizer.atk_token, add_special_tokens=False)["input_ids"][0]
    tokenizer.ctx_token_id = tokenizer(tokenizer.ctx_token, add_special_tokens=False)["input_ids"][0]
    tokenizer.tgt_token_id = tokenizer(tokenizer.tgt_token, add_special_tokens=False)["input_ids"][0]

    if isinstance(tokenizer, LlamaTokenizerFast):
        # whywhywhywhy llama?
        tokenizer.atk_token = " " + tokenizer.atk_token
        tokenizer.ctx_token = " " + tokenizer.ctx_token
        tokenizer.tgt_token = " " + tokenizer.tgt_token
    return tokenizer


def _patch_attention_maps_to_allow_4d_attention(model):
    if isinstance(model, LlamaForCausalLM):

        def _expand_mask(mask, dtype, tgt_len=None):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            if mask.dim() == 2:
                bsz, src_len = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len

                expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                inverted_mask = 1.0 - expanded_mask

                return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
            else:
                # pass a 4D mask straight through, without expansion
                return mask

        transformers.models.llama.modeling_llama._expand_mask = _expand_mask

    else:
        raise ValueError("Manually verify if 4D attention masks can be passed to this model, then add it here.")

    return model
