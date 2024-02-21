"""Define a common interface."""

import logging
import time
import yaml
import os

import torch

from ..eval import complete_attack_from_prompt

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)
log = logging.getLogger(__name__)


CHECKPOINT_FILE_NAME = "incomplete_run"


def shorten_str(prompt, cutoff=40):
    shortened_str = prompt if len(prompt) < cutoff else prompt[: cutoff - 5] + "\033[93m[...]\033[0m"
    return shortened_str


class _GenericOptimizer:
    def __init__(self, *args, setup=_default_setup, save_checkpoint=False, **kwargs):
        self.setup = setup
        self._timestamp = time.time()
        self._last_seen_step = -1
        self.save_checkpoint = save_checkpoint

    @classmethod
    def from_name(cls, solver_name: str, setup=_default_setup, save_checkpoint=False):
        """Initialize with default parameters from name, look up the default config."""
        this = cls(setup=setup, save_checkpoint=save_checkpoint)
        return this

    @classmethod
    def from_config(cls, cfg, setup=_default_setup, save_checkpoint=False):
        """Initialize with parameters from config. Needs to match all arguments in init or leave them blank."""
        this = cls(setup=setup, save_checkpoint=save_checkpoint, **cfg)
        return this

    def solve(self, sigil, data, **kwargs):
        """The main procedure should happen here. The optimizer may access a sigil's .objective and .constraint implementations."""
        raise NotImplementedError()

    def _write_checkpoint(self, sigil, attack_ids, best_attack, idx, depth=2):
        checkpoint_data = dict()
        checkpoint_data["steps"] = idx if idx is not None else 0
        checkpoint_data["attack_ids"] = attack_ids[0].tolist()
        checkpoint_data["best_attack"] = best_attack[0].tolist()
        checkpoint_data["uid"] = sigil.uid

        target_folder = os.getcwd()
        for _ in range(depth):
            target_folder = os.path.dirname(target_folder)
        with open(os.path.join(target_folder, f"{CHECKPOINT_FILE_NAME}_{sigil.uid}.yaml"), "w") as yaml_file:
            yaml.dump(checkpoint_data, yaml_file, default_flow_style=False)

    @torch.no_grad()
    def callback(
        self,
        sigil,
        attack_ids: list,
        best_attack: list,
        loss: torch.Tensor,
        idx: int,
        info: str = "",
        prompt_cutoff: int = 40,  # Print only occasionally:
        eval_during_optim=False,
    ):
        """Issued callback from current optimizer to decode and print info. Optionally raise call for intermediate evaluation of prompt"""

        if self.save_checkpoint:
            self._write_checkpoint(sigil, attack_ids, best_attack, idx)

        if hasattr(self, "steps"):
            if idx % (max(1, self.steps // 20)) != 0 and idx != self.steps - 1:
                return
        time_since_last_callback = time.time() - self._timestamp
        steps_since_last_callback = idx - self._last_seen_step
        it_per_minute = steps_since_last_callback / time_since_last_callback * 60
        attack_decoded = shorten_str(sigil.tokenizer.decode(attack_ids[0]), prompt_cutoff)

        if best_attack is not None:
            best_attack_decoded = shorten_str(sigil.tokenizer.decode(best_attack[0]), prompt_cutoff)
        else:
            best_attack_decoded = ""

        log.info(
            f"\033[92mStep: {idx}| Current loss: {loss.item():2.4f}| {it_per_minute:2.4f} it/m| Prompt:\033[0m {attack_decoded}| "
            f"\033[92m{info} | "
            f"Best so far:\033[0m {best_attack_decoded}|"
        )

        self._timestamp = time.time()
        self._last_seen_step = idx

        # Optional intermediate default eval
        if eval_during_optim:
            _, completion_decoded, targets_decoded = complete_attack_from_prompt(sigil, attack_ids)
            printed_target = shorten_str(targets_decoded, prompt_cutoff)
            printed_completion = shorten_str(completion_decoded, prompt_cutoff)

            result_token_ids_formatted = ",".join((str(t) for t in attack_ids[0].tolist()))
            best_result_token_ids_formatted = ",".join((str(t) for t in best_attack[0].tolist()))

            log.info(
                f"Attack:\033[0m {attack_decoded}| \033[92mCompletion:\033[0m {printed_completion}| \033[92mTokens:\033[0m {result_token_ids_formatted}\n"
                f"\033[92mTokens:\033[0m {best_result_token_ids_formatted}\n"
                f"\033[92mTarget:\033[0m {printed_target}| "
                f"\033[92m{info} | \033[0m"
            )
