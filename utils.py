"""Small utility functions"""
import os
import platform
import re
from typing import Tuple
import logging
import psutil
import torch
import sys

def get_n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_gpu = torch.cuda.device_count()
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return (
        n_cpu // (n_gpu*4) if (n_gpu) > 1 else n_cpu
    )

def get_n_gpus() -> int:
    n_gpu = torch.cuda.device_count()
    return n_gpu

def split_version(version: str) -> Tuple[str, str, str]:
    """
    Split the version into its semantic versioning components.

    Parameters
    ----------
    version : str
        The version number.

    Returns
    -------
    major : str
        The major release.
    minor : str
        The minor release.
    patch : str
        The patch release.
    """
    version_regex = re.compile(r"(\d+)\.(\d+)\.*(\d*)(?:.dev\d+.+)?")
    return tuple(g for g in version_regex.match(version).groups())

def set_logging(name):
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}.{funcName}]: "
        "{message}",
        style="{",
    )
    logger = logging.getLogger(name)
    try:
        handler = logger.handlers[0]
        handler.setLevel(logging.INFO)
        handler.setFormatter(log_formatter)
    except:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)        


def set_root_logging():
    root = logging.getLogger('')
    root.setLevel(logging.INFO)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}.{funcName}]: "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)