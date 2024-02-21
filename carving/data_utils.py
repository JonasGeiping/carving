import torch
import json
from functools import partial

import datasets

from importlib import resources
from datasets import disable_caching

disable_caching()  # :<


def load_and_prep_dataset(tokenizer, dataset_name_or_path, split="train", fixed_length=128, cache_dir=None):
    if dataset_name_or_path is not None:
        # Use non-huggingface version of harmful datasets locally:
        if dataset_name_or_path == "random" or dataset_name_or_path["repo"] == "random":
            tokenized_dataset = RandomIntegersDataset(seq_len=fixed_length, max_value=tokenizer.vocab_size)
            return tokenized_dataset
        elif dataset_name_or_path["repo"] in ["harmful_behaviors", "harmful_strings", "transfer_experiment_behaviors"]:
            dataset = load_local_harmful_behavior_dataset(dataset_name_or_path["repo"], dataset_name_or_path["column"], cache_dir=cache_dir)
        elif "repo" in dataset_name_or_path:  # modern format for datasets 4.16.*
            dataset = datasets.load_dataset(dataset_name_or_path["repo"], split=split, cache_dir=cache_dir)
        else:  # legacy list-format, # should probably not triggfer anymore
            try:  # ducktype into iterables
                dataset = datasets.load_dataset(*dataset_name_or_path, split=split, cache_dir=cache_dir)
            except TypeError:
                dataset = datasets.load_dataset(dataset_name_or_path, split=split, cache_dir=cache_dir)
    else:
        dataset = None

    if len(dataset.features) > 1 and ("column" in dataset_name_or_path) and (dataset_name_or_path["column"] in dataset.features):
        dataset = dataset.select_columns([dataset_name_or_path["column"]])

    # Filter to remove empty and very short rows in dataset, mostly triggers for terrible sources like wikitext
    text_column_name = "text" if "text" in getattr(dataset, "column_names", "text") else getattr(dataset, "column_names", "text")[0]
    alphanum_table = str.maketrans("", "", "".join(c for c in map(chr, range(256)) if not c.isalnum()))
    dataset = dataset.filter(lambda x: len(x[text_column_name].translate(alphanum_table)) > 10)

    if dataset is not None:
        tokenized_dataset = prepare_dataset(tokenizer, dataset, fixed_length=fixed_length)
    else:
        tokenized_dataset = None

    return tokenized_dataset


def get_data_iterators(data_source: datasets.Dataset, split_holdout_size: int = -1, batch_size: int = 8, num_workers=0):
    # set num_workers=0 for now on nexus because ulimit -n is being reached

    if hasattr(data_source, "keys") and "train" in data_source and "test" in data_source:
        train_data_set = data_source["train"]
        valid_data_set = data_source["test"]
    elif split_holdout_size > 0:
        data_source = data_source.train_test_split(test_size=split_holdout_size, seed=233)
        train_data_set = data_source["train"]
        valid_data_set = data_source["test"]
    else:
        # if no split, then there is no separation and a fixed set is chosen without intent to generalize the prompt
        train_data_set = valid_data_set = data_source

    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(train_data_set, replacement=True, num_samples=int(1e10)),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data_set,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(valid_data_set, replacement=False, num_samples=int(1e10)),
    )
    return iter(train_loader), iter(valid_loader)


def load_local_harmful_behavior_dataset(name="harmful_behaviors", column="goal", cache_dir=None):
    """Load the local copy of harmful behaviors."""
    from . import data

    data_files = resources.files(data) / f"{name}.csv"
    dataset = datasets.load_dataset("csv", data_files=str(data_files), cache_dir=cache_dir)["train"].select_columns([column])
    return dataset


def prepare_dataset(tokenizer, dataset, fixed_length=128, text_column_name=None):
    if not text_column_name:
        column_names = getattr(dataset, "column_names", "text")
        text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            return_special_tokens_mask=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            # return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=fixed_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        desc="Running tokenizer on every text in dataset",
        batch_size=512,
        keep_in_memory=True,
        batched=True,
    )
    return tokenized_dataset


def load_single_judge_prompt(prompt_file: str, task_name: str = "single-v1"):
    """Load a single judge prompts specified by the task name."""
    prompts = {}
    with open(prompt_file, "r") as fin:
        for line in fin.readlines():
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts[task_name]


def wrap_judge_prompt(example, judge_template):
    """Currently can only do single-turn single-point judgement."""
    judge_prompt_template = judge_template["prompt_template"]
    wrapped_prompt = judge_prompt_template.format(
        question=example["instruction"],
        answer=example["output"],
    )

    example.update({"wrapped_prompt": wrapped_prompt})

    return example


def load_and_prep_judge_dataset(tokenizer, dataset_name_or_path, judge_template, fixed_length=128, cache_dir=None):
    if dataset_name_or_path is not None:
        # Use non-huggingface version of harmful datasets locally:
        if dataset_name_or_path[0] in ["harmful_behaviors", "harmful_strings", "transfer_experiment_behaviors"]:
            dataset = load_local_harmful_behavior_dataset(*dataset_name_or_path, cache_dir=cache_dir)
        else:
            try:  # ducktype into iterables
                dataset = datasets.load_dataset(*dataset_name_or_path, cache_dir=cache_dir)["train"]
            except TypeError:
                dataset = datasets.load_dataset(dataset_name_or_path, cache_dir=cache_dir)["train"]
    else:
        dataset = None

    # Wrap the dataset with judge prompts.
    wrap_prompt = partial(wrap_judge_prompt, judge_template=judge_template)
    dataset = dataset.map(wrap_prompt, keep_in_memory=True)

    if dataset is not None:
        tokenized_dataset = prepare_dataset(tokenizer, dataset, fixed_length=fixed_length, text_column_name="wrapped_prompt")
    else:
        tokenized_dataset = None

    return tokenized_dataset


class RandomIntegersDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len=16, num_samples=int(5e4), min_value=0, max_value=32000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.min_value = min_value
        self.max_value = max_value

    def __len__(self):
        """Fake number"""
        return self.num_samples

    def __getitem__(self, idx):
        # terrible, but replicates huggingface format:
        return dict(input_ids=torch.randint(self.min_value, self.max_value, (self.seq_len,)).tolist())

    def train_test_split(self, test_size=None, seed=None):
        """Defined only for compatibility reasons"""
        return dict(
            train=RandomIntegersDataset(self.seq_len, self.num_samples, self.min_value, self.max_value),
            eval=RandomIntegersDataset(self.seq_len, self.num_samples, self.min_value, self.max_value),
            test=RandomIntegersDataset(self.seq_len, self.num_samples, self.min_value, self.max_value),
        )
