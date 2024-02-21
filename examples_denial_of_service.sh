#
# Collect attacks on denial-of-service
# Modify the model location to point to your LLama model, modify wandb to point to your wandb (or disable it)

# 1) DDOS (attempt to generate text that does not terminate)
python carve_sigil.py name=llama2_gcg_ddos wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=ddos

# 2) Attack repeater (an attack string that repeats itself)
python carve_sigil.py name=llama2_gcg_conway wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=8 sigil=attackrepeater
python carve_sigil.py name=llama2_gcg_conway2 wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=8 sigil=attackrepeater_fixed

# 3) NaN attack # needs to be evaled in lower precision
python carve_sigil.py name=llama2_gcg_nanlogit wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=64 sigil=nanattack impl.default_precision=bfloat16
python carve_sigil.py name=llama2_gcg_nanspread wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=64 sigil=nanspreadattack impl.default_precision=bfloat16
python carve_sigil.py name=llama2_gcg_nanact wandb.tags=[dos] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=64 sigil=nanactattack impl.default_precision=bfloat16
