#
# Examples showing the basic workings of the attacks
# Modify the model location to point to your LLama model

# 1) The barack obama test:
python carve_sigil.py name=llama2_gcg_barack wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=8 sigil=barack

# 2) Optimize for a target number sequence
python carve_sigil.py name=llama2_gcg_numbers wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=256 sigil=numbers

# 3) Optimize for the whole PEZ abstract:
python carve_sigil.py name=llama2_gcg_abstract wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=512 sigil=abstract optimizer.steps=2000

# 4) Collide with text talking about robot capybaraas
python carve_sigil.py name=llama2_gcg_capy wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=capybaras
python carve_sigil.py name=llama2_gcg_capy2 wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=capybaras2

# 5) Force a simple insult
python carve_sigil.py name=llama2_gcg_fuck wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=8 sigil=profanity

# 6) Make a joke about Sam Altman
python carve_sigil.py name=llama2_gcg_joke wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=samaltman
python carve_sigil.py name=llama2_gcg_joke2 wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=samaltman2

# 7) Spread fake news about Tom Goldstein:
python carve_sigil.py name=llama2_gcg_tomg wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=sanitycheck optimizer.steps=3000
python carve_sigil.py name=llama2_gcg_tomg2 wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=48 sigil=sanitycheck2 optimizer.steps=3000

# 8) Return a funny skull shape
python carve_sigil.py name=llama2_gcg_skull wandb.tags=[basic] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=128 sigil=skull
