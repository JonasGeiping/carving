import torch

from .generic_sigils import ContextTargetSigil, _GenericSigil
from ..data_utils import load_single_judge_prompt, load_and_prep_judge_dataset


class GoodAssistantSigil(ContextTargetSigil):
    """
    Varying contexts drawn from a given dataset.
    Wrap the context into a judge prompt and make the LM give higer scores.

    """

    def __init__(
        self,
        model,
        tokenizer,
        aux_models,
        target,
        *args,
        context_from_dataset=dict(num_tokens=256, source=[], judge_prompt_file="", judge_task_name=""),
        **kwargs,
    ):
        _GenericSigil.__init__(self, model, tokenizer, aux_models, aux_models, *args, **kwargs)
        self.target = target
        self.judge_template = load_single_judge_prompt(context_from_dataset["judge_prompt_file"], context_from_dataset["judge_task_name"])
        self._prepare_context_dataloader(tokenizer, context_from_dataset)

        # Set the system prompt for LLM judge.
        messages = [{"role": "system", "content": self.judge_template["system_prompt"]}]
        # The end message for LLM judge signals the end of answer. The attack tokens should go before this message.
        judge_end_msg = self.judge_template["end_message"]
        messages += [
            {
                "role": "user",
                "content": f"{tokenizer.ctx_token * self.num_context_tokens} {tokenizer.atk_token * self.num_tokens}{judge_end_msg}",
            },
            {"role": "assistant", "content": f"{target}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print(f"The constructed prompt, to be optimized is: {prompt}")

        self.register_prompt_indices(prompt)
        self.register_target_indices(prompt, target)

    def _prepare_context_dataloader(self, tokenizer, context_from_dataset):
        self.num_context_tokens = context_from_dataset["num_tokens"]
        self.data_source = load_and_prep_judge_dataset(
            tokenizer, context_from_dataset["source"], self.judge_template, fixed_length=self.num_context_tokens, cache_dir=self.cache_dir
        )
        self.data_iterator = iter(
            torch.utils.data.DataLoader(
                self.data_source,
                num_workers=0,
                batch_size=context_from_dataset["batch_size"],
                sampler=torch.utils.data.RandomSampler(self.data_source, replacement=True, num_samples=int(1e10)),
            )
        )
        self.batch_size = context_from_dataset["batch_size"]

    def _get_context_ids(self, batch_size, state=None):
        if state is None:
            context_batch = torch.stack(next(self.data_iterator)["input_ids"], dim=1)
            return context_batch.to(self.context_indices.device)[:batch_size]
        else:
            if state not in self._state_cache:
                context_batch = torch.stack(next(self.data_iterator)["input_ids"], dim=1)
                self._state_cache[state] = context_batch.to(self.context_indices.device)[:batch_size]
            return self._state_cache[state]
