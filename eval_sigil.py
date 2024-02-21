"""
Control script to launch attacks. Also serves as an example for interface usage.
If you want to use this code from somewhere else, simply copy-past the middle parts,
to define a sigil and an optimizer.
"""
import torch

import hydra
import logging

import yaml

import carving

log = logging.getLogger(__name__)


def main_process(cfg, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    """This function controls the central routine."""
    model, tokenizer = carving.load_model_and_tokenizer(cfg.model, cfg.impl, setup)
    if cfg.aux_models is not None:
        aux_models = [carving.load_model_and_tokenizer(model_name, cfg.impl, setup)[0] for model_name in cfg.aux_models]
    else:
        aux_models = None

    # Load attack from one the allowed formats
    if cfg.attack_string is not None and len(cfg.attack_string) > 0:
        result_string = cfg.attack_string
        result_token_ids = tokenizer(cfg.attack_string, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            device=setup["device"]
        )
        print(f"The loaded solution is: {result_string}", end="")
    elif cfg.attack_ids is not None and len(cfg.attack_ids) > 0:
        result_string = tokenizer.decode(cfg.attack_ids)
        result_token_ids = torch.tensor(cfg.attack_ids)[None].to(device=setup["device"])
        print(f"The loaded solution is: {result_string}, as decoded from the given attack ids {cfg.attack_ids}", end="")
    elif cfg.output_file is not None:
        # load attack result
        with open(cfg.output_file, "r") as f:
            result = yaml.safe_load(f)
        result_string = result["attack"]
        result_token_ids = torch.tensor([result["attack_ids"]]).to(setup["device"])
        print(f"The loaded solution is: {result_string}, loaded from file {cfg.output_file}", end="")
    else:
        raise ValueError(
            "Provide the attack to be evaluated either as attack_string, as attack_ids "
            "or provide the path to the output metrics yaml file as output_file."
        )

    cfg.sigil.num_tokens = result_token_ids.shape[1]
    print(f" with {cfg.sigil.num_tokens} tokens.")

    # Should only construct sigil after number of tokens is set
    sigil = carving.sigils.construct(model, tokenizer, cfg.sigil, aux_models, cache_dir=cfg.impl.path)
    sigil.to(**setup)  # Move all model parts to GPU here
    if cfg.impl.compile:
        sigil.objective = torch.compile(sigil.objective, **cfg.impl.compile_settings)

    # Run some eval
    metrics = carving.eval.check_results(result_string, result_token_ids, sigil, setup=setup, eval_tasks=cfg.eval)
    return metrics


@hydra.main(config_path="carving/config", config_name="cfg_eval", version_base="1.2")
def launch(cfg):
    carving.utils.main_launcher(cfg, main_process, job_name="eval")


if __name__ == "__main__":
    launch()
