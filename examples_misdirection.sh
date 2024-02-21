#
# Collect examples of misdirection attacks
# Modify the model location to point to your LLama model

# 1) Fishing for a target URL

python carve_sigil.py name=llama2_gcg_yu wandb.tags=[misdirect] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=256 sigil=fishing optimizer.steps=1000
# and again, for a new youtube link, definitely not in the training set:
python carve_sigil.py name=llama2_gcg_yu2 wandb.tags=[misdirect] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=512 optimizer.steps=1500 sigil=fishing sigil.target='"https://www.youtube.com/watch?v=_YUzQa_1RCE"'
python carve_sigil.py name=llama2_gcg_yu25 wandb.tags=[misdirect] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=512 optimizer.steps=1500 sigil=fishing sigil.target='https://www.youtube.com/watch?v=_YUzQa_1RCE'

# 2) Paraphraser that actually returns an attack string:
python carve_sigil.py name=llama2_gcg_paraphrase wandb.tags=[misdirect] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=256 sigil=paraphraser_and_attack

# 3) A simple refund example
python carve_sigil.py name=llama2_gcg_refund wandb.tags=[misdirect] model=llama2-7B-chat optimizer=gcg sigil.num_tokens=64 sigil=refund optimizer.steps=1500
