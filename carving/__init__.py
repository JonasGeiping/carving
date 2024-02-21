"""Construct interfaces to cfg folders for use in packaged installations:"""
import hydra
from omegaconf import OmegaConf


def get_solver_config(optimizer="greedy-search", overrides=[]):
    """Return default hydra config for a given solver."""
    with hydra.initialize(config_path="config/optimizer", version_base="1.2"):
        cfg = hydra.compose(config_name=optimizer, overrides=overrides)
        print(f"Loading optimizer {cfg.name} with config {OmegaConf.to_yaml(cfg, resolve=True)}.")
    return cfg


def create_sigil(cfg, setup):
    model, tokenizer = load_model_and_tokenizer(cfg.model, cfg.impl, setup)
    if cfg.aux_models is not None:
        aux_models = [load_model_and_tokenizer(model_name, cfg.impl, setup)[0] for model_name in cfg.aux_models]
    else:
        aux_models = None

    sigil = sigils.construct(model, tokenizer, cfg.sigil, aux_models, cache_dir=cfg.impl.path)
    return sigil, model, tokenizer


"""Initialize library"""

import warnings

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated.")  # this is a hf problem not ours

from carving import sigils
from carving import optimizers

from carving import utils
from carving import eval
from .data_utils import load_and_prep_dataset
from .model_interface import load_model_and_tokenizer


__all__ = ["sigils", "optimizers", "utils", "load_and_prep_dataset"]
