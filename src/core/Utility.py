"""This module provides routines for setting the computing platform, precision etc

"""

import jax
import os


def SetPrecision(precision: str):
    """Set the computation precision

    Parameters:

    precision (str): name of the precision, say "64"

    Jax will use single precision by default
    """
    jax.config.update("jax_enable_x64", True)


def SetPlatform(platform: str):
    """Set the hardware for computation

    Parameters:

    platform (str): name of the platform, say "cpu"

    Jax will use GPU by default
    """
    if platform.upper() == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        jax.config.update("jax_platform_name", "cpu")
