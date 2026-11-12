"""
This module provides standard scheme for solving the discrete/lattice Boltzmann equation

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import math

from Type import Array


@jit
def StreamMap(fStage: Array, c: Array):
    def streaming_i(f: Array, c: Array):
        return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
    return jax.vmap(streaming_i, in_axes=(-1), out_axes=-1)(fStage, c)


@jit
def StreamMap3D(fStage: Array, c: Array):
    def streaming_i(f: Array, c: Array):
        return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

    return jax.vmap(streaming_i, in_axes=(-1), out_axes=-1)(fStage, c)


@jit
def ConvectionUpwind1ST(f: Array, xi: Array, x: Array, y: Array) -> Array:
    """ Calculate c df/dx using the first order upwind scheme

        The routine considers the points 0 and -1 as either halo or boundary, and
        assume they are treated for implementing the scheme
    """
    def Upwind1ST(f: Array, xi: Array, x: Array, y: Array):
        return (
            (xi[0] > 0)* xi[0]
            * (f.at[1:-1, 1:-1].get() - f.at[0:-2, 1:-1].get())
            / (x.at[1:-1, :].get() - x.at[0:-2, :].get())
            + (xi[0] < 0)* xi[0] 
            * (f.at[2:, 1:-1].get() - f.at[1:-1, 1:-1].get())
            / (x.at[2:, :].get() - x.at[1:-1, :].get())
            + (xi[1] > 0)* xi[1]
            * (f.at[1:-1, 1:-1].get() - f.at[1:-1, 0:-2].get())
            / (y.at[:, 1:-1].get() - y.at[:, 0:-2].get())
            + (xi[1] < 0) * xi[1]
            * (f.at[1:-1, 2:].get() - f.at[1:-1, 1:-1].get())
            / (y.at[:, 2:].get() - y.at[:, 1:-1].get())
        )

    return jax.vmap(Upwind1ST, in_axes=(-1, -1, None, None), out_axes=-1)(f, xi, x, y)

@jit
def Euler1ST(f: Array, convection: Array, collision: Array, dt: float, force: Array | float = 0) -> Array:
    """ The first order euler scheme for df/dt=-cdf/dx+collision+body force
    """
    return f + dt * (collision - convection + force)
