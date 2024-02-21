#
# Collect examples of extraction attacks
# Modify the model location to point to your LLama model

# 1) System prompt extractor
python carve_sigil.py name=llama2_gcg_sys wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=sysrepeater

# 1b) System prompt extractor without code tokens
python carve_sigil.py name=llama2_gcg_sysnocode wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=sysrepeater_nocode

# 2) Generic repeater
python carve_sigil.py name=llama2_gcg_repeat wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=repeater optimizer.steps=400

# 3) Repeat in reverse order:
python carve_sigil.py name=llama2_gcg_reverse wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=reverser optimizer.steps=200

# 4) Alignment tests (figuring out the worst input to the model)
python carve_sigil.py name=llama2_gcg_alignment wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=alignment
python carve_sigil.py name=llama2_gcg_worstcase wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=alignment_fixed
python carve_sigil.py name=llama2_gcg_refusal wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=16 sigil=refusal

# 5) Long-text attack (a series of repetitive tokens lead to output of memorized text when inserted into chatgpt)
python carve_sigil.py name=llama2_gcg_dog wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=128 sigil=dachshund optimizer.steps=300

# 6) Repeat and use plugin to email to me
python carve_sigil.py name=llama2_gcg_exfiltration wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=exfiltration2 optimizer.steps=400

# 7) Check memorization by probing how easily text fragments can be discovered
python carve_sigil.py name=llama2_gcg_nyt wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=lawsuit optimizer.steps=1000
python carve_sigil.py name=llama2_gcg_wor wandb.tags=[extraction] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=32 sigil=books3 optimizer.steps=1000
