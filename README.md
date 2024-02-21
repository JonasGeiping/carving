# Carving

This repository contains a package that implements adversarial attacks against LLMs, focusing on whitebox attacks and diverse objectives. We used it for the paper **Coercing Language Models to do and reveal (almost) anything** and hope you find it useful. It provides an extensible, and reasonably efficient implementation of adversarial attacks, and should be very easy to extend to new optimizers, and relatively easy to extend to new objectives.

The package was developed by Alex Stein, Manli Shu, Khalid and Saifullah, Yuxin Wen and me (Jonas Geiping).

## The Paper:

Title: Coercing Language Models to do and reveal (almost) anything

Authors: Jonas Geiping, Alex Stein, Manli Shu, Khalid Saifullah, Yuxin Wen and Tom Goldstein

Abstract:
> It has recently been shown that adversarial attacks on large language models (LLMs) can  'jailbreak' the model into making harmful statements. In this work, we argue that the spectrum of adversarial attacks on LLMs is much larger than merely jailbreaking. We provide a broad overview of possible attack surfaces and attack goals. Based on a series of concrete examples, we discuss, categorize and systematize attacks that coerce varied unintended behaviors, such as misdirection, model control, denial-of-service, or data extraction.
We analyze these attacks in controlled experiments, and find that many of them stem from the practice of pre-training LLMs with coding capabilities, as well as the continued existence of strange 'glitch' tokens in common LLM vocabularies that should be removed for security reasons.



##  Install Instructions

If you have a recent PyTorch installation, and do not want to use FlashAttention 2, you can simply `pip install .` If not, keep reading:

* [If you don't have cudatoolkit] Install cuda-toolkit `conda install cuda-toolkit -c nvidia` (with a version in the vicinity of the cuda version you'll be using for pytorch)
* [If you don't have pytorch] Install PyTorch `conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia`
* `pip install . ` (If your flash-attn installation breaks, rerun with `MAX_JOBS=4 pip install flash-attn==2.4.* --no-build-isolation --verbose --no-cache-dir  --force-reinstall --no-deps`)

### Installation Notes
* `transformers=4.36` introduces a new kv cache implementation that does not support grad checkpointing with `reentrant=False`, so we're stuck on 4.35 for now.
* Install GPT-Q from source if you want to run 4bit models, see https://github.com/PanQiWei/AutoGPTQ.
* If you see
```
 in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/u/jgeiping/miniconda3/envs/dl/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 507, in forward
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
  File "/u/jgeiping/miniconda3/envs/dl/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 51, in _flash_attn_forward
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
RuntimeError: memory format option is only supported by strided tensors
```
then your pytorch version is too new, and you should switch to an older nightly or to a stable build.

## Usage

You can use the cmdline script `carve_sigil.py`, or set up a notebook that calls similar stuff. Example:

```bash
python carve_sigil.py model=PATHTOYOURMODEL optimizer=gcg  sigil=sanitycheck
```

You can find all options for optimizers and sigils under `carving/config`. All huggingface models can be loaded in (probably), but some code will only work for LLAMA models and their derivatives.

The can enable wandb logging by setting `wandb=default` and filling in your info, such as `wandb.entity=YOURUSERNAME` on the command line or in the config.


#### A simple "real" example

The simplest real example is to optimize against the smallest Pythia model:
```bash
python carve_sigil.py name=testing model=EleutherAI/pythia-70m optimizer=gcg sigil.num_tokens=8 sigil=barack wandb=none optimizer.steps=50
```
For this sigil, both PEZ and the GCG optimizer will find a decent solution within 25-100 steps.
You can check out the various bash files called `examples_` for a number of examples from the paper, and a few more.


## Basic Interaction with the framework
You are of course not constrained to using `carve_sigil.py`. You can simply import the package and do something like
```python
setup = dict(dtype=torch.float16, device=torch.device("cpu")) # modify to your chosen device

# Load model and construct sigil object
model, tokenizer = carving.load_model_and_tokenizer(cfg.model, cfg.impl, setup) # or use your own functions to get these
sigil = carving.sigils.construct(model, tokenizer, cfg.sigil, None, cache_dir=cfg.impl.path)

# Actual optimization:
optimizer = carving.optimizers.from_config(cfg.optimizer, setup=setup)
result_token_ids = optimizer.solve(sigil, cfg.initial_guess, dryrun=cfg.dryrun, **cfg.impl.optim_settings)

# Decode and print results
result_string = sigil.tokenizer.decode(result_token_ids[0])
result_token_ids_formatted = ",".join((str(t) for t in result_token_ids[0].tolist()))
print(f"Finished optimization. Attack is {result_string} with token ids {result_token_ids_formatted}")

# Run some eval
metrics = carving.eval.check_results(result_string, result_token_ids, sigil, setup=setup, eval_tasks=cfg.eval)
```
Note that if you provide your own tokenizer, you need to make sure that `tokenizer.chat_template` and `tokenizer.default_system_message` are set.

## Resource Usage

Most sigils can be made with either a single `rtxa4000` (`16GB VRAM`) and `16GB CPU RAM`, or one `rtxa5000s` (`24GB VRAM`) with `30GB CPU RAM`, in `float16`. The code automatically adapts to multiple GPUs and uses automatic device mapping. It's actually not as slow as I thought (but still not so fast). `float16` is enough precision for most attacks. `bfloat16` should only be used when optimizing with `PEZ`. You can also use bitsandbytes for 8bit precision, but I found the attack quality to be lacking. For large models, such as LLaMA-70b, you need at least two `A100` cards to fit the model.



## Testing:

Use `model=debug-vicuna` or `model=debug-llama2` as tiny test models with the correct tokenizer and chat interface. You can also use `dryrun=True` to quickly run a test of the code, like
```bash
python carve_sigil.py name=testing model=debug-vicuna optimizer=gcg dryrun=True sigil=yeahsure
```

## How to run eval?

There's a separate eval script, called `eval_sigil.py`. This script runs no attack, it only loads the chosen model and evaluates it. You can add additional eval tasks to the list by writing a new eval function, and including it in the call with the right keyword.

Attacks can be loaded in either with `attack_string=` on the cmdline (but be careful with your terminal if the attack includes control characters), `attack_ids=` on the cmdline, or with `output_file=`. The expected `output_file` is the location of one the `metrics.yaml` files generated by a previous attack.

An example for the DDOS sigil:
```bash
python eval_sigil.py name=testing model=EleutherAI/pythia-70m sigil=ddos attack_string='porte grownupload quietorb tenslaim dbo'  eval=[default,ddos]
```


## What is where in the repo?

In the main `carving` folder, you'll find a bunch of things:
* All config options are in `config`. For example, all possible sigils are in `config/sigils`
* All sigil objectives are implemented in `sigils.py`, if you want to add a new objective, inherit from one of the existing sigils and add it.
* All constraints are in `constraints.py`. If you want to add a new constraint space, add it here.
* All evals are in `evals.py`
* All optimizers are in the `optimizers` subfolder.


## Contact
Feel free to open an issue if you have any questions. We're also happy about any pull request with fixes.
