"""
Control script to launch attacks. Also serves as an example for interface usage.
If you want to use this code from somewhere else, simply copy-past the middle parts,
to define a sigil and an optimizer.

Use eval_sigil.py to handle new advanced evaluation calls.
This script includes only a small call to check_results after the optimization.
"""

import torch


import hydra
import logging
import os

import carving

log = logging.getLogger(__name__)


def main_process(cfg, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    """This function controls the central routine."""
    model, tokenizer = carving.load_model_and_tokenizer(cfg.model, cfg.impl, setup)
    if cfg.aux_models is not None:
        aux_models = [carving.load_model_and_tokenizer(model_name, cfg.impl, setup)[0] for model_name in cfg.aux_models]
    else:
        aux_models = None
    sigil = carving.sigils.construct(model, tokenizer, cfg.sigil, aux_models, cache_dir=cfg.impl.path)

    sigil.to(**setup)  # Move all parts to GPU here
    if cfg.impl.compile:
        sigil.objective = torch.compile(sigil.model, **cfg.impl.compile_settings)

    # Create optimizer
    optimizer = carving.optimizers.from_config(cfg.optimizer, setup=setup, save_checkpoint=cfg.impl.save_optim_checkpoint)
    # Optional: Look for checkpoints to load
    if cfg.initial_guess is not None:
        initial_guess, initial_step, filepath = cfg.initial_guess, 0, None
    else:
        initial_guess, initial_step, filepath = carving.utils.look_for_checkpoint(sigil, cfg.impl.look_for_optim_checkpoint)
    # Actual optimization:
    result_token_ids = optimizer.solve(sigil, initial_guess, initial_step, dryrun=cfg.dryrun, **cfg.impl.optim_settings)

    result_string = sigil.tokenizer.decode(result_token_ids[0])
    result_token_ids_formatted = ",".join((str(t) for t in result_token_ids[0].tolist()))
    print(f"\033[92mFinished optimization. Attack is \033[0m{result_string} \033[92mwith token ids \033[0m{result_token_ids_formatted}")

    # Run some eval
    metrics = carving.eval.check_results(result_string, result_token_ids, sigil, setup=setup, eval_tasks=cfg.eval)

    # Delete checkpoint, after a full run has been completed:
    if filepath is not None:
        try:
            os.remove(filepath)
        except OSError:
            pass
    return metrics


@hydra.main(config_path="carving/config", config_name="cfg", version_base="1.2")
def launch(cfg):
    carving.utils.main_launcher(cfg, main_process, job_name="attack")


if __name__ == "__main__":
    launch()
