"""This module provides routines for setting up the model,
including lattice, equilibrium,collision etc

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import math
from Type import Lattice, DVS, Array
from typing import Callable
from functools import partial
import jax.numpy as jnp
sqrt3 = math.sqrt(3)


def D3Q19Lattice() -> Lattice:
    name = "D3Q19"
    t000 = 1 / 3
    t001 = 1 / 18
    t011 = 1 / 36
    w = jnp.asarray(
        [
            t000,
            t001,
            t001,
            t001,
            t001,
            t001,
            t001,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
            t011,
        ]
    )
    cx = jnp.asarray(
        [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0], dtype="int32"
    )
    cy = jnp.asarray(
        [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1], dtype="int32"
    )
    cz = jnp.asarray(
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1], dtype="int32"
    )
    c = jnp.ones((3, 19), dtype="int32")
    c = c.at[0, :].set(cx)
    c = c.at[1, :].set(cy)
    c = c.at[2, :].set(cz)
    ctranspose = c.transpose()
    opp = jnp.asarray(
        [ctranspose.tolist().index((-ctranspose[i]).tolist()) for i in range(19)]
    )
    cs = sqrt3
    return {
        "Name": name,
        "XI": c * cs,
        "XIT": ctranspose * cs,
        "C": c,
        "W": w,
        "OPP": opp,
        "CS": cs,
    }


def D3Q15Lattice() -> Lattice:
    name = "D3Q15"
    t000 = 2 / 9
    t001 = 1 / 9
    t111 = 1 / 72
    w = jnp.asarray(
        [
            t000,
            t001,
            t001,
            t001,
            t001,
            t001,
            t001,
            t111,
            t111,
            t111,
            t111,
            t111,
            t111,
            t111,
            t111,
        ]
    )
    cx = jnp.asarray([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1], dtype="int32")
    cy = jnp.asarray([0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1], dtype="int32")
    cz = jnp.asarray([0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1], dtype="int32")
    c = jnp.ones((3, 15), dtype="int32")
    c = c.at[0, :].set(cx)
    c = c.at[1, :].set(cy)
    c = c.at[2, :].set(cz)
    ctranspose = c.transpose()
    opp = jnp.asarray(
        [ctranspose.tolist().index((-ctranspose[i]).tolist()) for i in range(15)]
    )
    cs = sqrt3
    return {
        "Name": name,
        "XI": c * cs,
        "XIT": ctranspose * cs,
        "C": c,
        "W": w,
        "OPP": opp,
        "CS": cs,
    }


def D2Q9Lattice() -> Lattice:
    """Define D2Q9 lattice"""
    name = "D2Q9"
    cx = jnp.asarray([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype="int32")
    cy = jnp.asarray([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype="int32")
    c: Array = jnp.ones((2, 9), dtype="int32")
    c = c.at[0, :].set(cx)
    c = c.at[1, :].set(cy)
    ctranspose = c.transpose()
    w: Array = 1.0 / 36.0 * jnp.ones(9)
    w = w.at[jnp.asarray([jnp.linalg.norm(ci) < 1.1 for ci in ctranspose])].set(
        1.0 / 9.0
    )
    w = w.at[0].set(4.0 / 9.0)
    opp = jnp.asarray(
        [ctranspose.tolist().index((-ctranspose[i]).tolist()) for i in range(9)]
    )
    cs = sqrt3
    return {
        "Name": name,
        "XI": c * cs,
        "XIT": ctranspose * cs,
        "C": c,
        "W": w,
        "OPP": opp,
        "CS": cs,
    }


def D2Q16DVS() -> DVS:
    name = "D2Q16_DVS"
    XI = jnp.asarray(
        [
            [
                -2.334414218338977,
                -2.334414218338977,
                -2.334414218338977,
                -2.334414218338977,
                -0.7419637843027259,
                -0.7419637843027259,
                -0.7419637843027259,
                -0.7419637843027259,
                0.7419637843027259,
                0.7419637843027259,
                0.7419637843027259,
                0.7419637843027259,
                2.334414218338977,
                2.334414218338977,
                2.334414218338977,
                2.334414218338977,
            ],
            [
                -2.334414218338977,
                -0.7419637843027259,
                0.7419637843027259,
                2.334414218338977,
                -2.334414218338977,
                -0.7419637843027259,
                0.7419637843027259,
                2.334414218338977,
                -2.334414218338977,
                -0.7419637843027259,
                0.7419637843027259,
                2.334414218338977,
                -2.334414218338977,
                -0.7419637843027259,
                0.7419637843027259,
                2.334414218338977,
            ],
        ]
    )

    XIT = XI.transpose()#转置
    w = jnp.asarray(
        [
            0.0021045940507009157,
            0.020833333333333353,
            0.020833333333333356,
            0.0021045940507009157,
            0.020833333333333353,
            0.20622873928263247,
            0.20622873928263252,
            0.020833333333333353,
            0.020833333333333356,
            0.20622873928263252,
            0.20622873928263258,
            0.020833333333333356,
            0.0021045940507009157,
            0.020833333333333353,
            0.020833333333333356,
            0.0021045940507009157,
        ]
    )
    opp = jnp.asarray([XIT.tolist().index((-XIT[i]).tolist()) for i in range(16)])
    return {
        "Name": name,
        "XI": XI,
        "XIT": XIT,
        "W": w,
        "OPP": opp,
    }


def DefineLattice(name: str) -> Lattice:
    """Return a predefined lattice

    Args:
        name (str): the name for the needed lattice

    Returns:
        Lattice:
    """
    if name == "D2Q9":
        lattice = D2Q9Lattice()
    elif name == "D3Q19":
        lattice = D3Q19Lattice()
    elif name == "D3Q15":
        lattice = D3Q15Lattice()
    return lattice


def DefineDVS(name: str) -> DVS:
    """Return a predefined discrete velocity set

    Args:
        name (str): the name for the needed discrete veloct set

    Returns:
        DVS:
    """
    if name == "D2Q16":
        lattice = D2Q16DVS()
    return lattice


@jit
def BGKIsothemalEquilibrium(rho: Array, u: Array, XI: Array, W: Array) -> Array:
    cu = jnp.dot(u, XI)
    usqr = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    return rho * W * (1.0 + cu + 0.5 * (cu**2 - usqr))


@jit
def BGKIsothemalCollision(
    f: Array,
    rho: Array,
    u: Array,
    XI: Array,
    W: Array,
    tau: float,
    dt: float,
) -> Array:
    cu = jnp.dot(u, XI)
    usqr = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    feq = rho * W * (1.0 + cu + 0.5 * (cu**2 - usqr))
    dtOvertauPlusdt = dt / (tau + 0.5 * dt)
    return feq + (1 - dtOvertauPlusdt) * (f - feq)


@jit
def BGKEquilibriumSWE4th(h: Array, u: Array, XI: Array, W: Array) -> Array:
    cu: Array = jnp.dot(u, XI)
    usqr: Array = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    xi2: Array = jnp.sum(jnp.square(XI), axis=0)
    return (
        h
        * W
        * (
            1.0
            + cu
            + 0.5 * (cu**2 - usqr + (h - 1) * (xi2 - 2))
            + (cu / 6) * (cu**2 - 3 * usqr + 3 * (h - 1) * (xi2 - 4))
            + 1 / 24 * (cu**4 - 6 * cu**2 * usqr + 3 * usqr**2)
            + (h - 1) / 4 * ((xi2 - 4) * (cu**2 - usqr) - 2 * cu**2)
            + (h - 1) ** 2 / 8 * (xi2**2 - 8 * xi2 + 8)
        )
    )


@jit
def BGKSWECollision(f: Array, feq: Array, tau: Array | float, h: Array | float):
    return h * (feq - f) / tau


@jit
def CalcMacroVars(f: Array, xit: Array) -> tuple[Array, Array]:
    rho = jnp.sum(f, axis=-1, keepdims=True)
    u = jnp.dot(f, xit) / rho
    return rho, u


@partial(jit, static_argnums=(0,))
def EquilibriumTest(
    Equilibrium: Callable,
    rho: Array,
    u: Array,
    xi: Array,
    xit: Array,
    w: Array,
    error: float = 1e-14,
):
    """Test if the equilibrium function satisfies the conservation laws"""
    feq = Equilibrium(rho, u, xi, w)
    rhonew, unew = CalcMacroVars(feq, xit)
    rhoerr = jnp.sqrt(jnp.sum((rho - rhonew) ** 2) / jnp.sum(rho**2))
    uerr = jnp.sqrt(jnp.sum((u - unew) ** 2) / jnp.sum(u**2))
    return rhoerr < error, uerr < error
