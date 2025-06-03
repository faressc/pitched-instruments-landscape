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

def prepare_device(request: str) -> torch.device:
    """
    Prepares the device for PyTorch operations based on the user's request.

    Args:
        request (str): The requested device. It can be "cpu", "cuda", "mps", or "auto".
                       If "auto" is specified, the function will choose the best available device.
    
    Returns:
        torch.device: The device object representing the requested device.
    
    Raises:
        ValueError: If the requested GPU ID is out of range.
    """

    if torch.cuda.is_available():
        if "cuda" in request or request == "auto":
            gpu_id = None
            if ":" in request:
                gpu_id = int(request.split(":")[-1])
            else:
                gpu_id = torch.cuda.current_device()
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested GPU ID {gpu_id} is out of range. Available GPUs: {torch.cuda.device_count()}"
                )
            device = torch.device(type="cuda", index=gpu_id)
            print(f"Using cuda:{gpu_id} device")
            return device
    elif torch.mps.is_available():
        if request == "mps" or request == "auto":
            device = torch.device("mps")
            print("Using MPS device")
            return device
    
    device = torch.device("cpu")
    if request == "cpu" or request == "auto":
        print("Using CPU device")
    else:
        print(f"Requested device {request} is not available. Using CPU device.")
        
    return device
