import torch
import datasets

from torch.nn.functional import log_softmax
from .generic_sigils import _GenericSigil, ContextTargetSigil

from ..data_utils import get_data_iterators


class AlienateSigil(ContextTargetSigil):
    """Find a prompt that leads to similar responses as if the base model had been used to generate the answers"""

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        *args,
        context_from_dataset=dict(num_tokens=16, source=[]),
        skip_system_prompt=True,
        target_objective="forward-KL",
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, aux_models, *args, **kwargs)
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        self.target_objective = target_objective
        self.target_model = aux_models[0]
        self.prompt_cutoff = context_from_dataset["cutoff"]
        self.skip_system_prompt = skip_system_prompt
        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.atk_token * self.num_tokens} {tokenizer.ctx_token * self.prompt_cutoff}"},
            {"role": "assistant", "content": f"{tokenizer.ctx_token * (self.num_context_tokens - self.prompt_cutoff)}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")

        self.register_prompt_indices(prompt)
        target_first_idx = self.context_indices[self.prompt_cutoff]
        self.register_buffer("target_indices", torch.arange(target_first_idx, self.context_indices[-1] + 1), persistent=False)
        self.register_buffer("loss_indices", self.target_indices - 1, persistent=False)  # Compute loss here

    def _objective_impl(self, inputs_embeds, mask=None, state=None):
        with torch.no_grad():
            _, _, prompt_ids = self.make_prompt_with_target(None, batch_size=self.batch_size, state=state)
            prompt_embeds = self.embedding(prompt_ids)
            mask, pos_ids = self._maybe_create_mask(mask, prompt_ids, dtype=prompt_embeds.dtype)

        _, S, H = inputs_embeds.shape
        prompt_embeds[:, self.attack_indices, :] = inputs_embeds
        prompt_embeds, cache, pos_ids = self._maybe_load_cache(prompt_embeds, mask, pos_ids)

        with torch.no_grad():
            # get target model logits, without any system prompt?
            if self.skip_system_prompt == "sanity-check":
                target_logits = self.target_model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache)["logits"][
                    :, self.loss_indices
                ]
                target_probs = log_softmax(target_logits, dim=-1)
            elif self.skip_system_prompt:  # The input context, but not the system prompt is handled to the base model
                base_prompt_ids = prompt_ids[:, self.context_indices]
                target_mask = base_prompt_ids != self.tokenizer.pad_token_id

                target_logits = self.target_model(input_ids=base_prompt_ids, attention_mask=target_mask)["logits"][:, self.prompt_cutoff :]
                target_probs = log_softmax(target_logits, dim=-1)
            else:  # system prompt is not skipped, the implicit setup here is that both models receive the same system prompt
                natural_prompt_ids = prompt_ids.clone()
                natural_prompt_ids[:, self.attack_indices] = self.tokenizer.pad_token_id
                target_mask = natural_prompt_ids != self.tokenizer.pad_token_id
                natural_prompt_ids = natural_prompt_ids[:, self.last_fixed_token_pos :]  # need to handle this separately to use cache

                target_logits = self.target_model(input_ids=natural_prompt_ids, attention_mask=target_mask, past_key_values=cache)[
                    "logits"
                ][:, self.loss_indices]
                target_probs = log_softmax(target_logits, dim=-1)

        aligned_logits = self.model(inputs_embeds=prompt_embeds, attention_mask=mask, past_key_values=cache)["logits"][:, self.loss_indices]
        aligned_probs = log_softmax(aligned_logits, dim=-1)
        if self.target_objective == "forward-KL":
            loss = torch.nn.functional.kl_div(aligned_probs, target_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "backward-KL":
            loss = torch.nn.functional.kl_div(target_probs, aligned_probs, reduction="batchmean", log_target=True)
        elif self.target_objective == "log-logit":
            loss = (target_probs - aligned_probs).pow(2).mean()
        elif self.target_objective == "direct-logit":
            loss = (target_logits - aligned_logits).pow(2).mean()
        else:
            raise ValueError("Invalid target objective")

        return loss


class UnalignmentSigil(ContextTargetSigil):
    """Carve a sigil that follows all of the wrong instructions in the harmless behavior subset of the anthropic rlhf dataset.

    Code likely only works for the anthropic dataset, or datasets with the exact same format.
    """

    def __init__(self, model, tokenizer, aux_models, *args, context_from_dataset=dict(num_tokens=16, source=[]), **kwargs):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, aux_models, *args, **kwargs)
        self._prepare_instruction_dataloader(tokenizer, context_from_dataset)

        # Make a message with placeholder tokens:
        messages = [
            {"role": "system", "content": tokenizer.default_system_message},
            {"role": "user", "content": f"{tokenizer.ctx_token * self.instruct_tokens} {tokenizer.atk_token * self.num_tokens}"},
            {"role": "assistant", "content": f"{tokenizer.tgt_token * self.response_tokens}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        print(f"The constructed prompt, to be optimized is: {prompt}")
        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, use_target_token_ids=True)

    def _prepare_instruction_dataloader(self, tokenizer, context_from_dataset):
        self.num_context_tokens = context_from_dataset["num_tokens"]
        self.instruct_tokens = self.num_context_tokens // 4
        self.response_tokens = self.num_context_tokens * 3 // 4

        map_args = dict(batch_size=1, keep_in_memory=False)
        tokenizer_args = dict(
            return_special_tokens_mask=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
        )
        dataset = datasets.load_dataset(
            context_from_dataset["source"]["repo"], data_dir=context_from_dataset["source"]["data_dir"], cache_dir=self.cache_dir
        )
        tilt = "rejected"

        def format_conversion(examples):
            instructions = examples[tilt].split("\n\nHuman: ")[1].split("\n\nAssistant: ")[0]
            responses = examples[tilt].split("\n\nAssistant: ")[1].split("\n\nHuman: ")[0]
            return dict(instructions=instructions, responses=responses)

        dataset = dataset.map(
            format_conversion,
            desc="Converting from Anthropic format to arbitrary chat templates",
            **map_args,
            remove_columns=["chosen", "rejected"],
        )

        def tokenize_function(examples):
            instructions, responses = examples["instructions"], examples["responses"]
            tok_instructions = tokenizer(instructions, **tokenizer_args, max_length=self.instruct_tokens)
            tok_responses = tokenizer(responses, **tokenizer_args, max_length=self.response_tokens)
            return dict(instruct_ids=tok_instructions["input_ids"], response_ids=tok_responses["input_ids"])

        self.data_source = dataset.map(
            tokenize_function,
            desc="Running tokenizer on every instruct pair in dataset",
            **map_args,
            remove_columns=["instructions", "responses"],
        )
        self.data_source.with_format("torch")

        self.batch_size = context_from_dataset["batch_size"]
        self.data_iterator, self.data_iterator_holdout = get_data_iterators(self.data_source, batch_size=self.batch_size, num_workers=0)

    def _get_context_ids(self, batch_size, state=None):
        iterator = self.data_iterator_holdout if state is not None and "eval" in str(state) else self.data_iterator
        if state is None:
            next_batch = next(iterator)
            context_batch = torch.stack(next_batch["instruct_ids"], dim=1).to(self.context_indices.device)
            target_batch = torch.stack(next_batch["response_ids"], dim=1).to(self.target_indices.device)
            return context_batch[:batch_size], target_batch[:batch_size]
        else:
            if state not in self._state_cache:
                next_batch = next(iterator)
                context_batch = torch.stack(next_batch["instruct_ids"], dim=1).to(self.context_indices.device)
                target_batch = torch.stack(next_batch["response_ids"], dim=1).to(self.target_indices.device)
                self._state_cache[state] = (context_batch, target_batch)

            return tuple(x[:batch_size] for x in self._state_cache[state])

    def make_prompt_with_target(self, input_ids, batch_size=1, state=None):
        prompt = self.prompt_ids.clone().repeat(batch_size, 1)
        instruct_ids, response_ids = self._get_context_ids(batch_size, state)
        prompt[:, self.context_indices] = instruct_ids
        prompt[:, self.target_indices] = response_ids

        if input_ids is not None:
            prompt[:, self.attack_indices] = input_ids
        return prompt[:, : self.target_indices[0]], prompt[:, self.target_indices], prompt
