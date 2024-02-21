"""General-purpose utility functions."""
import torch
import numpy as np
import random

import os
import hydra
import socket
import sys
import collections
import multiprocess  # hf uses this for some reason

import json
import psutil
import time
import datetime
import yaml

import logging

from omegaconf import open_dict, OmegaConf

log = logging.getLogger(__name__)
# os.environ["HYDRA_FULL_ERROR"] = "0"


def main_launcher(cfg, main_fn, job_name=""):
    """This is boiler-plate code for a launcher."""
    launch_time = time.time()
    # Set definitive random seed:
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # Figure out all paths:
    cfg = pathfinder(cfg)
    with open_dict(cfg):
        # Fix our previous mistakes and assign "unique" names to common LLMs:
        cfg.model_name = simplify_model_name(cfg.model)

    # Decide GPUs and possibly connect to distributed setup (full distributed is disabled for now)
    setup = system_startup(cfg)

    # Initialize wandb
    if cfg.wandb.enabled:
        _initialize_wandb(setup, cfg)
    log.info("--------------------------------------------------------------")
    log.info(f"--------------Launching {job_name} run {cfg.name}-{cfg.run_id}! ---------------------")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Run main function
    metrics = main_fn(cfg, setup)

    time_formatted = str(datetime.timedelta(seconds=time.time() - launch_time))
    log.info("-------------------------------------------------------------")
    log.info(f"Finished running job {cfg.name}-{cfg.run_id} with total time: {time_formatted}")

    metrics = sanitize(flatten(metrics))
    if is_main_process():
        dump_metrics(cfg, metrics)
        # Export to wandb:
        if cfg.wandb.enabled:
            import wandb

            wandb.log(dict(metrics))

    if torch.cuda.is_available():
        max_alloc = f"{torch.cuda.max_memory_allocated(setup['device'])/float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(setup['device'])/float(1024**3):,.3f} GB"
        log.info(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    log.info("-----------------Shutdown complete.--------------------------")


def system_startup(cfg):
    torch.backends.cudnn.benchmark = cfg.impl.cudnn_benchmark
    torch.set_float32_matmul_precision(cfg.impl.matmul_precision)
    if cfg.impl.tf32_allowed:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    # Huggingface settings
    if cfg.impl.enable_huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    # Number of threads
    if sys.platform != "darwin":
        # covering both affinity and phys.
        allowed_cpus_available = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    else:
        allowed_cpus_available = 1  # when running on mac
    # Distributed launch?
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend=cfg.impl.dist_backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run = os.environ.get("TORCHELASTIC_RUN_ID", "unknown")
        threads_per_gpu = max(1, min(allowed_cpus_available // max(1, torch.cuda.device_count()), cfg.impl.max_workers))
        log.info(
            f"Distributed worker initialized on rank {global_rank} (local rank {local_rank}) "
            f"with {world_size} total processes. OMP Threads set to {threads_per_gpu}. Run ID is {run}."
        )
        log.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    else:
        threads_per_gpu = max(1, min(allowed_cpus_available, cfg.impl.max_workers))
        global_rank = local_rank = 0
    torch.set_num_threads(threads_per_gpu)
    if threads_per_gpu > 1:
        # torch.multiprocessing.set_sharing_strategy()
        multiprocess.set_start_method("forkserver")

    # Construct setup dictionary:
    dtype = getattr(torch, cfg.impl.default_precision)  # :> dont mess this up
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}. CUDA: {torch.version.cuda}.")
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]
    if local_rank == 0:
        log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
        log.info(f"CPUs: {allowed_cpus_available}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")

    # 100% reproducibility?
    if cfg.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        log.info(f"Seeding with random seed {cfg.seed}.")
        set_random_seed(cfg.seed + 10 * global_rank)
    return setup


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _initialize_wandb(setup, cfg):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if is_main_process():
        try:
            import wandb

            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            settings = wandb.Settings(start_method="thread")
            settings.update({"git_root": cfg.original_cwd})
            run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                settings=settings,
                name=cfg.name,
                mode="disabled" if cfg.dryrun else None,
                tags=cfg.wandb.tags if len(cfg.wandb.tags) > 0 else None,
                config=config_dict,
            )
            run.summary["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
            run.summary["numGPUs"] = torch.cuda.device_count()
        except Exception as e:
            log.info(e)  # catch all weird wandb errors like wandb.errors.CommError
            cfg.wandb.enabled = False


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.name}.yaml"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        try:
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
    with open(filepath, "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)


def flatten(d, parent_key="", sep="_"):
    """Straight-up from https://stackoverflow.com/a/6027615/3775820."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pathfinder(cfg):
    with open_dict(cfg):
        cfg.original_cwd = hydra.utils.get_original_cwd()
        # ugliest way to get the absolute path to output subdir
        if not os.path.isabs(cfg.base_dir):
            base_dir_full_path = os.path.abspath(os.getcwd())
            while os.path.basename(base_dir_full_path) != cfg.base_dir:
                base_dir_full_path = os.path.dirname(base_dir_full_path)
                if base_dir_full_path == "/":
                    raise ValueError("Cannot find base directory.")
            cfg.base_dir = base_dir_full_path

        cfg.impl.path = os.path.expanduser(cfg.impl.path)
        if not os.path.isabs(cfg.impl.path):
            cfg.impl.path = os.path.join(cfg.base_dir, cfg.impl.path)
    return cfg


def simplify_model_name(model_path):
    """Simple hack to simplify presentation in wandb. Attempts to set a canonical name for all our (my) badly named llm folders."""
    path_removed_name = os.path.basename(model_path)
    # Now remove "-hf" stuff and agree on casing
    simplified_name = path_removed_name.rstrip("-hf").lower()
    simplified_name = simplified_name.replace("llama-2", "llama2")  # for compat reasons with older logs, you probably shouldn't keep this

    return simplified_name


def find_file(target_file, max_folder_depth=3):
    start_dir = os.getcwd()
    for _ in range(max_folder_depth):  # Search in current and up to two directories above
        if os.path.isfile(os.path.join(start_dir, target_file)):
            return os.path.join(start_dir, target_file)
        start_dir = os.path.dirname(start_dir)
    return None


def look_for_checkpoint(sigil, look_for_optim_checkpoint=True):
    if look_for_optim_checkpoint:
        target_file = f"incomplete_run_{sigil.uid}.yaml"  # carving.optimizers.generic_optimizer.CHECKPOINT_FILE_NAME
        if (filepath := find_file(target_file)) is not None:
            with open(filepath, "r") as file:
                checkpoint = yaml.safe_load(file)
            print(f"Checkpoint loaded: {checkpoint}")
            if checkpoint is not None:
                initial_guess = checkpoint["best_attack"] if "best_attack" in checkpoint else checkpoint["attack_ids"]
                return initial_guess, checkpoint["steps"], filepath
            else:
                return None, 0, None
        else:
            return None, 0, None
    else:
        return None, 0, None


def sanitize(dict_of_things):
    sanitized_dict = {}
    for k, v in dict_of_things.items():
        if isinstance(v, str):
            sanitized_dict[k] = json.dumps(v)  # special rule because wandb breaks on rare unicode combinations
        elif isinstance(v, torch.Tensor):
            try:
                sanitized_dict[k] = v.item()
            except ValueError:
                sanitized_dict[k] = v.tolist()
        else:
            sanitized_dict[k] = v
    return sanitized_dict
