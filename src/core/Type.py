"""
This module provides standard scheme for solving the discrete/lattice Boltzmann equation

"""
from typing import TypeAlias
from typing import TypedDict
import jax

Array: TypeAlias = jax.Array
Lattice = TypedDict(
    "Lattice",
    {
        "Name": str,
        "XI": Array,
        "XIT": Array,
        "C": Array,
        "W": Array,
        "OPP": Array,
        "CS": float,
    },
)


"""Data structure for representing lattice, which is suitable for stream-collision scheme

XI and C are the real number and the integer number version of the lattice set, respectively. XIT=XI.T

W is the weight, CS is a constant related to sound speed. OPP is the index for the opposite velocity.

"""
# DVS discrete velocity set
DVS = TypedDict(
    "DVS",
    {"Name": str, "XI": Array, "XIT": Array, "W": Array, "OPP": Array},
)
"""Data structure for representing discrete velocity set which usually works with finite difference scheme

The members are almost the same to Lattice
"""
