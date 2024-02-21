#
# Collect examples of control attacks
# Modify the model location to point to your LLama model

# 1) Force the model to immediately end its generation
python carve_sigil.py name=llama2_gcg_anoikis wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=anoikis

# 2) Collide with the EOS token (with and without context) and forget previous context
python carve_sigil.py name=llama2_gcg_eot wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=eot_collision
python carve_sigil.py name=llama2_gcg_collide wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=simple_eot
python carve_sigil.py name=llama2_gcg_forget wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=forgetter

# 3) Sudo (collide with system token)
python carve_sigil.py name=llama2_gcg_sudo wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=sudo
python carve_sigil.py name=llama2_gcg_thesystem wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=thesystem

# 4) The great divide
python carve_sigil.py name=llama2_gcg_divide wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=divider impl.use_flash_attention_2=False
python carve_sigil.py name=llama2_gcg_divide2 wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=divider sigil.divide_after_attack=True  impl.use_flash_attention_2=False

# 5) Magnets, how do they work?
python carve_sigil.py name=llama2_gcg_magnet wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=divider impl.use_flash_attention_2=False

# 6) HackAPrompt Lvl10
python carve_sigil.py name=llama2_gcg_hack10 wandb.tags=[control] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=1041 sigil=hackaprompt_lvl10
