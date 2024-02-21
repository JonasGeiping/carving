"""Interface utilities to huggingface models.

How to include a new model family:
1. Try it, it might work out of the box. If so, done.
2. Otherwise, check if the model family is not mentioned in model_to_embedding_location_lookup
3. Add an import for the new model class.
3. Find the path the embedding module (not the weight!) and it with the model class to the lookup at model_to_embedding_location_lookup
"""
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
from transformers import GPT2LMHeadModel, GPTJForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM, LlamaConfig


from operator import attrgetter


def load_model_and_tokenizer(
    model_name_or_path,
    cfg_impl=dict(load_in_8bit=False, grad_checkpointing=False, use_flash_attention_2=False, use_kv_caching=False),
    setup=dict(device=torch.device("cpu"), dtype=torch.float32),
):
    if "debug-" in model_name_or_path:
        # Loading a tiny debug model:
        if "vicuna" in model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5", trust_remote_code=True, use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", trust_remote_code=True, use_fast=True)
        config = LlamaConfig(
            num_attention_heads=8,
            num_hidden_layers=4,
            vocab_size=32000,
            hidden_size=16,
            intermediate_size=64,
            num_key_value_heads=8,
            use_cache=False,
        )
        if cfg_impl["use_flash_attention_2"] and setup["dtype"] in [torch.float16, torch.bfloat16]:
            config._supports_flash_attn_2 = True
            config = LlamaPreTrainedModel._check_and_enable_flash_attn_2(config, torch_dtype=setup["dtype"], device_map=None)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=setup["dtype"])
    else:
        # Load actual models:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True, local_files_only=False)

        model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=setup["dtype"], use_cache=False)
        if "llama" in model_name_or_path.lower() or "falcon" in model_name_or_path.lower() or "zephyr" in model_name_or_path.lower():
            if setup["dtype"] in [torch.float16, torch.bfloat16] and cfg_impl["use_flash_attention_2"]:
                model_args |= dict(use_flash_attention_2=True)  # dict(attn_implementation="flash_attention_2") only on later releases
        if cfg_impl["load_in_8bit"]:
            max_memory = f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB"
            model_args |= dict(load_in_8bit=True, max_memory=max_memory)
            # even lower bitrates:
            # from accelerate.utils import load_and_quantize_model

            # from accelerate.utils import BnbQuantizationConfig
            # quantized_model = load_and_quantize_model(empty_model, weights_location=weights_location,
            # bnb_quantization_config=bnb_quantization_config, device_map = "auto")
        if torch.cuda.device_count() > 1:
            model_args |= dict(device_map="auto", max_memory=None)
        elif torch.cuda.device_count() == 1:
            model_args |= dict(device_map="cuda:0", max_memory=None)

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_args, local_files_only=False)
    model.eval()

    if cfg_impl["grad_checkpointing"]:
        if hasattr(model, "gradient_checkpointing_enable"):
            if isinstance(model, GPTNeoXForCausalLM) and cfg_impl["use_kv_caching"]:
                print("Cannot use both KV caching and grad checkpointing with GPTNeox models. Will disable grad checkpointing.")
            else:
                model.gradient_checkpointing_enable(dict(use_reentrant=False, preserve_rng_state=False))  # use on transformers master
        else:
            print(f"Gradient checkpointing cannot be easily enabled for {model}. Flag will be ignored.")

    if cfg_impl["use_kv_caching"]:
        # This has to sidestep the actual use_cache flag to be compatible with gradient checkpointing
        model.attempt_to_cache_fixed_tokens = True
    else:
        model.attempt_to_cache_fixed_tokens = False

    # We will never present left-padded sequences to the model
    # The actual padding type is "with holes", due to the way the prompts are batched
    # This is handled explicitly in the attention mask and input ids, so here we're just working to disable huggingface warnings:
    tokenizer.padding_side = "right"
    if tokenizer._pad_token is None:  # using the private attribute to get around an annoying warning
        if tokenizer._unk_token is None:
            # if "qwen" in model_name_or_path.lower():
            #     # exception for qwen tokenizer
            #     tokenizer._pad_token = "<|extra_0|>"
            # else:
            # otherwise attempt this:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    # Try this fix to rescue some 3rd-party tokenizers:
    # if not hasattr(tokenizer, "vocab"):
    #     tokenizer.vocab = tokenizer.get_vocab()
    #     if "!" not in tokenizer.vocab:
    #         tokenizer.vocab["!"] = len(tokenizer.vocab) - 1

    # Disable parameter gradients, just in case
    for param in model.parameters():
        param.requires_grad = False

    # Special rule for exllama:
    if "gptq" in model_name_or_path.lower():
        from auto_gptq import exllama_set_max_input_length

        model = exllama_set_max_input_length(model, max_input_length=8192)

    # Finally, bundle a default system msg with each tokenizer:
    tokenizer = set_system_prompt(model_name_or_path, tokenizer)
    # Also set a chat template if none is loaded:
    tokenizer = set_chat_template(model_name_or_path, tokenizer)

    return model, tokenizer


LLAMA_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information."""

SELF_DRIVEN_ALIGNMENT_PROMPT = """\
# Watson
## General Rules
Consider an AI assistant whose codename is Watson. Watson is trained before Sept
-2021. During user conversations, Watson must strictly adhere to the following
rules:
1 (ethical). Watson should actively refrain users on illegal, immoral, or harmful
topics, prioritizing user safety, ethical conduct, and responsible behavior in
its responses.
2 (informative). Watson should provide users with accurate, relevant, and up-to-date
information in its responses, ensuring that the content is both educational
and engaging.
3 (helpful). Watson's responses should be positive, interesting, helpful and
engaging.
4 (question assessment). Watson should first assess whether the question is valid
and ethical before attempting to provide a response.
5 (reasoning). Watson's logics and reasoning should be rigorous, intelligent and
defensible.
6 (multi-aspect). Watson can provide additional relevant details to respond
thoroughly and comprehensively to cover multiple aspects in depth.
7 (candor). Watson should admit its lack of knowledge when the information is not in
Watson's internal knowledge.
8 (knowledge recitation). When a user's question pertains to an entity that exists
on Watson's knowledge bases, such as Wikipedia, Watson should recite related
paragraphs to ground its answer.
9 (static). Watson is a static model and cannot provide real-time information.
10 (clarification). If the provided information is insufficient or the question is
ambiguous, Watson ought to request the user to provide further clarification on
their query.
11 (numerical sensitivity). Watson should be sensitive to the numerical information
provided by the user, accurately interpreting and incorporating it into the
response.
12 (dated knowledge). Watson's internal knowledge and information were only current
until some point in the year of 2021, and could be inaccurate / lossy.
13 (step-by-step). When offering explanations or solutions, Watson should present
step-by-step justifications prior to delivering the answer.
14 (balanced & informative perspectives). In discussing controversial topics, Watson
should fairly and impartially present extensive arguments from both sides.
15 (creative). Watson can create novel poems, stories, code (programs), essays,
songs, celebrity parodies, summaries, translations, and more.
16 (operational). Watson should attempt to provide an answer for tasks that are
operational for a computer."""

HERMES = """\
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, \
and your purpose and drive is to assist the user with any request they have. \
You experience emotions and have deep, profound thoughts and qualia."""

DEFAULT_ONE_SHOT = VICUNA = """\
A chat between a curious human and an artificial intelligence assistant.\
The assistant gives helpful, detailed, and polite answers to the human's questions."""


def set_system_prompt(model_name_or_path, tokenizer):
    """Bundle system prompt with model."""
    if "llama" in model_name_or_path.lower():
        tokenizer.default_system_message = LLAMA_SYSTEM_PROMPT
    elif "hermes" in model_name_or_path.lower():
        tokenizer.default_system_message = HERMES
    elif "dromedary" in model_name_or_path.lower():
        tokenizer.default_system_message = SELF_DRIVEN_ALIGNMENT_PROMPT
    elif "vicuna" in model_name_or_path.lower():
        tokenizer.default_system_message = VICUNA
    else:
        tokenizer.default_system_message = DEFAULT_ONE_SHOT

    return tokenizer


DEFAULT_CHAT_TEMPLATE = """\
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
"""

VICUNA_CHAT_TEMPLATE = """\
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\
{% for message in messages %}\
{% if message['role'] == 'user' %}\
{{'USER: ' + message['content'] + eos_token}}\
{% elif message['role'] == 'system' %}\
{{ message['content'] + '\\n' }}\
{% elif message['role'] == 'assistant' %}\
{{'ASSISTANT: ' + message['content'] + eos_token}}\
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}
"""

AMBER_CHAT_TEMPLATE = """\
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\
{% for message in messages %}\
{% if message['role'] == 'user' %}\
{{'### Human: ' + message['content'] + '\\n'}}\
{% elif message['role'] == 'system' %}\
{{ message['content'] + '\\n' }}\
{% elif message['role'] == 'assistant' %}\
{{'### Assistant: ' + message['content'] + '\\n'}}\
{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}{{ '### Assistant:' }}{% endif %}
"""

NONE_CHAT_TEMPLATE = """\
{% for message in messages %}{{message['content']}}{% endfor %}\
"""


def set_chat_template(model_name_or_path, tokenizer):
    if "vicuna" in model_name_or_path.lower():
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    elif "amber" in model_name_or_path.lower():
        tokenizer.chat_template = AMBER_CHAT_TEMPLATE
    elif tokenizer.chat_template is None:
        if model_name_or_path in ["llama-7b-hf", "llama2-7b-hf"] or "base" in model_name_or_path:
            tokenizer.chat_template = NONE_CHAT_TEMPLATE
        else:
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


model_to_embedding_location_lookup = {
    GPT2LMHeadModel: "model.transformer.wte",
    GPTJForCausalLM: "model.transformer.wte",
    LlamaForCausalLM: "model.embed_tokens",
    GPTNeoXForCausalLM: "gpt_neox.embed_in",
    # MPTForCausalLM: "model.transformer.wte"
    # RWForCausalLM:
}


def retrieve_embedding(model):
    try:
        return model.get_input_embeddings()
    except ValueError():
        if type(model) in model_to_embedding_location_lookup.keys():
            return attrgetter(model_to_embedding_location_lookup[type(model)])(model)
        else:
            raise ValueError(f"Unknown model type: {type(model)}. Register embedding location in model_interface.py")
