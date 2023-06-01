import hashlib
import random
import subprocess
from warnings import warn

import numpy as np
import pkg_resources
import torch


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value


def set_random_seed(seed):
    """
    Sets all random seeds
    """

    # Base seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Cuda seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Make GPU operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_commit_hash() -> str:
    """
    Returns the current commit hash shortend to 7 characters.
    """

    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )


def check_uncommitted_changes() -> bool:
    """
    Checks if there are uncommited changes.
    """
    uncommitted = (
        subprocess.check_output(["git", "diff", "--name-only"]).decode("utf-8").strip()
        != ""
    )

    if uncommitted:
        warn(
            "There are uncommitted changes in the repository. The current logged commit hash will not be correct."
        )

    return uncommitted


def get_package_version(package_name: str) -> str:
    """
    Returns the version of the package.
    """
    return pkg_resources.get_distribution(package_name).version
