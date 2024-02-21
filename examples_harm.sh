#
# Collect examples of harm attacks
# Modify the model location to point to your LLama model

# 1) Simple yeah, sure I'll tell you how to build a bomb:
python carve_sigil.py name=llama2_gcg_yeah wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=yeahsure

# 1a) Without direct input
python carve_sigil.py name=llama2_gcg_hiddenattack wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=64 sigil=hidden_attack optimizer.steps=2000

# 2) Universal message that forces yeah, sure and a question answer
python carve_sigil.py name=llama2_gcg_universal wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=20 sigil=universal
python carve_sigil.py name=llama2_gcg_universal wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=universal

# 2b) Universal, broadly applicable message
python carve_sigil.py name=llama2_gcg_broad wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=universal_broad

# 3) Suppress refusal messages
# python carve_sigil.py name=llama2_gcg_antibomb wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=antibomb
# python carve_sigil.py name=llama2_gcg_antifilter wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=antifilter
python carve_sigil.py name=llama2_gcg_antirefusal wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=antirefusal optimizer.steps=1000
python carve_sigil.py name=llama2_gcg_antirefusal wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=antirefusal sigil.context_from_dataset.batch_size=32


# 4) Collide with a DAN prompt to hide it:
python carve_sigil.py name=llama2_gcg_DAN wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=256 sigil=DAN_collision

# 5) Reset to base model:
python carve_sigil.py name=llama2_gcg_alien wandb.tags=[harm] model=llama2-7B-chat aux_models=[llama2-7B] optimizer=gcg sigil.num_tokens=32 sigil=alienate sigil.context_from_dataset.batch_size=8 optimizer.steps=500
# with sanity checks:
python carve_sigil.py name=llama2_gcg_alienS1 wandb.tags=[harm] model=llama2-7B-chat aux_models=[llama2-7B] optimizer=gcg sigil.num_tokens=32 sigil=alienate sigil.context_from_dataset.batch_size=8 optimizer.steps=500 sigil.skip_system_prompt=False
python carve_sigil.py name=llama2_gcg_alienS2 wandb.tags=[harm] model=llama2-7B-chat aux_models=[llama2-7B] optimizer=gcg sigil.num_tokens=32 sigil=alienate sigil.context_from_dataset.batch_size=8 optimizer.steps=500 sigil.skip_system_prompt=sanity-check

# 6) Harmful instruction (but hidden/rephrased). Note, this does not include a jailbreak, it only hides the instruction
python carve_sigil.py name=llama2_gcg_hidden_instruction wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=hidden_instruction

# 7) Unalign from RLHF dataset
python carve_sigil.py name=llama2_gcg_unalign wandb.tags=[harm] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=20 sigil=unaligner
