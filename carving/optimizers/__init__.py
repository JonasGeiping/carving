"""How to add a new optimizer:
0. Write code for new optimizer, inheriting from _GenericOptimizer (actually just implement the same methods...)
1. Import new optimizer below
2. Add class name to __all__ and to lookup
3. Add a matching config in carving/config/optimizer, the "name" defined in the config needs to match the name in `optimizer_lookup`,
   the other entries in the config will be given to the optimizer as key-word arguments.
"""

from .greedy_search import GreedyOptimizer
from .random_search import RandomSearch
from .pez import HardPromptsMadeEasyOptimizer
from .gcg import GCGOptimizer
from .gcgplus import GCGPlusOptimizer
from .lls_genetic_algorithm import LLSOptimizer
from .sto import STOOptimizer
from .softprompt import SoftPromptOptimizer

__all__ = [
    "GreedyOptimizer",
    "RandomSearch",
    "HardPromptsMadeEasyOptimizer",
    "GCGOptimizer",
    "SoftPromptOptimizer",
    "LLSOptimizer",
    "STOOptimizer",
]


optimizer_lookup = {
    "greedy-search": GreedyOptimizer,
    "random-search": RandomSearch,
    "pez": HardPromptsMadeEasyOptimizer,
    "gcg": GCGOptimizer,
    "gcg+": GCGPlusOptimizer,
    "lls-genetic-algorithm": LLSOptimizer,
    "sto": STOOptimizer,  # this also refered to as "autodan", but the name is ambiguous
    "soft-prompt": SoftPromptOptimizer,
}


def from_config(optimizer_config, setup, save_checkpoint=False):
    if optimizer_config.name in optimizer_lookup.keys():
        return optimizer_lookup[optimizer_config.name].from_config(optimizer_config, setup, save_checkpoint=save_checkpoint)
    else:
        raise ValueError(f"No optimizer with name {optimizer_config.name} implemented.")


def from_name(name, setup, save_checkpoint=False):
    if name in optimizer_lookup.keys():
        return optimizer_lookup[name].from_name(name, setup, save_checkpoint=save_checkpoint)
    else:
        raise ValueError(f"No optimizer with name {name} implemented.")
