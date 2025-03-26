# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

"""
This module handles the configuration for the python project.
"""

import copy
import os
from collections.abc import MutableMapping
from typing import Any, Dict, Generator, Tuple

import torch
from ruamel.yaml import YAML


def get_env_variable(var_name: str) -> str:
    """
    Retrieves the value of a specified environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the specified environment variable.

    Raises:
        EnvironmentError: If the environment variable is required but not set,
                          except for "SLURM_JOB_ID", where None is returned instead.

    Note:
        If the environment variable is "SLURM_JOB_ID" and it is not set, the function returns None.
        For all other environment variables, an EnvironmentError is raised if they are not set.
    """
    value = os.getenv(var_name)
    if var_name == "SLURM_JOB_ID" and value is None:
        return None
    if value is None:
        raise EnvironmentError(
            f"The environment variable {var_name} is required but not set."
        )
    return value

def auto_device() -> torch.device:
    """
    Automatically get accelerated device if available, otherwise return cpu device

    Returns:
        torch.device: device to be used
    """
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def prepare_device(request: str) -> torch.device:
    """
    Prepares the appropriate PyTorch device based on the user's request.

    Args:
        request (str): The type of device requested. Options include "mps", "cuda", and "cpu".
                       - "mps": Metal Performance Shaders (for Apple Silicon GPUs).
                       - "cuda": NVIDIA CUDA GPU.
                       - "cpu": Central Processing Unit.

    Returns:
        torch.device: The device that will be used for tensor operations.

    Notes:
        - If "mps" is requested but not available, the function defaults to "cpu".
        - If "cuda" is requested but not available, the function defaults to "cpu".
        - If the request is neither "mps" nor "cuda", the function defaults to "cpu".

    Example:
        device = prepare_device("cuda")
    """
    if request == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("MPS requested but not available. Using CPU device")
    elif request == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("CUDA requested but not available. Using CPU device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def set_random_seeds(random_seed: int) -> None:
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        random_seed (int): The seed value to be used for random number generation.

    Notes:
        - Sets the seed for the following libraries if they are available:
          - `random`: Python's built-in random module.
          - `numpy`: NumPy for handling arrays and matrices.
          - `torch`: PyTorch for deep learning operations.
          - `scipy`: SciPy for scientific computing.
        - If a library is not imported in the global scope, the seed setting for that library will be skipped.

    Example:
        set_random_seeds(42)
    """
    if "random" in globals():
        random.seed(random_seed)  # type: ignore
    else:
        print("The 'random' package is not imported, skipping random seed.")

    if "np" in globals():
        np.random.seed(random_seed)  # type: ignore
    else:
        print("The 'numpy' package is not imported, skipping numpy seed.")

    if "torch" in globals():
        torch.manual_seed(random_seed)  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(random_seed)
    else:
        print("The 'torch' package is not imported, skipping torch seed.")
    if "scipy" in globals():
        scipy.random.seed(random_seed)  # type: ignore
    else:
        print("The 'scipy' package is not imported, skipping scipy seed.")
