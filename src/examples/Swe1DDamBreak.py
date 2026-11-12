#!/usr/bin/python
"""
This is an example of simulating the 1D ideal dam break problem

The 2D shallow water equations are solved by using a discrete Boltzmann model
"""
import os
import h5py as h5
import jax
import jax.numpy as jnp
from jax import jit
import math
import Model
import Scheme
import Utility
from Type import Array, DVS
from rich import pretty
from timeit import default_timer as timer


pretty.install()
Utility.SetPrecision("64")
Utility.SetPlatform("gpu")
print("JAX local devices: ", jax.local_devices())
print("Jax device", jax.devices())


import matplotlib.pyplot as plt
from matplotlib import cm

###### Flow definition #########################################################
# Total number of time iterations.
lattice: DVS = Model.DefineDVS("D2Q16")

halo = 0
nx = 501 + 2 * halo
ny = 3 + 2 * halo
tau = 0.001
L = 1000
dx = L / (nx - 1 - 2 * halo)
dt = 0.001
# we collect for every period of 1 (in the non-dimensional system)
period=1/dt
maxIter = 20 * 10000
x = jnp.arange(0, L + dx, dx).reshape((nx, 1))
y = jnp.asarray([-1, 0, 1]).reshape(1, ny)

midPoint = (nx - 1) / 2
hInlet = jnp.ones((1, ny, 1))
hOut = 0.001
hOutlet = hOut * jnp.ones((1, ny, 1))


@jit
def TreateBoundary(f, u, hInlet, hOutlet):

    # inlet h
    f = f.at[0:1, :, :].set(
        Model.BGKEquilibriumSWE4th(
            hInlet, u.at[1:2, :, :].get(), lattice["XI"], lattice["W"]
        )
    )

    # # outlet h
    f = f.at[-1:, :, :].set(
        Model.BGKEquilibriumSWE4th(
            hOutlet, u.at[-2:-1, :, :].get(), lattice["XI"], lattice["W"]
        )
    )

    # # bottom periodic
    f = f.at[:, 0, :].set(f.at[:, 1, :].get())
    # top periodic
    f = f.at[:, -1, :].set(f.at[:, -2, :].get())
    return f


h: Array = jnp.fromfunction(
    lambda i, j, k: (i <= midPoint) + hOut * (i > midPoint), (nx, ny, 1)
)
u: Array = jnp.zeros((nx, ny, 2))

feq = Model.BGKEquilibriumSWE4th(h, u, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)


@jit
def CalcGrad(fGrad0, f):
    return fGrad0.at[1:-1, 1:-1, :].set(
        Scheme.ConvectionUpwind1ST(f, lattice["XI"], x, y)
    )


skip = 1

start = timer()

###### Main time loop ##########################################################
for time in range(maxIter):

    f = TreateBoundary(f, u, hInlet, hOutlet)
    h, u = Model.CalcMacroVars(f, lattice["XIT"])
    if (time % 10000 == 0) and (time != 0):  # Visualization
        print("time=", time)
        print("hx0=", h[0, 1, 0], "ux0=", u[0, 1, 0], "vx0=", u[0, 1, 1])
        print("hx1=", h[1, 1, 0], "ux1=", u[1, 1, 0], "vx`=", u[1, 1, 1])
        print("hx-2=", h[-2, 1, 0], "ux-2=", u[-2, 1, 0], "vx-2=", u[-2, 1, 1])
        print("hx-1=", h[-1, 1, 0], "ux-1=", u[-1, 1, 0], "vx-1`=", u[-1, 1, 1])
        plt.clf()
        Fr = u.at[:, 1, 0].get() / (jnp.sqrt(2 * h.at[:, 1, 0].get()))
        plt.plot(
            x.at[::skip, 0].get(),
            h.at[::skip, 1, 0].get(),
            # x.at[::skip, 0].get(),
            # u.at[::skip, 1, 0].get(),
            # x.at[::skip, 0].get(),
            # Fr.at[::skip].get(),
        )
        plt.savefig("h." + str(time / 10000).zfill(4) + ".png")
    fGrad = CalcGrad(fGrad0, f)
    f = Scheme.Euler1ST(
        f,
        fGrad,
        Model.BGKSWECollision(
            f, Model.BGKEquilibriumSWE4th(h, u, lattice["XI"], lattice["W"]), tau, h
        ),
        dt,
        0,
    )

end = timer()
print(end - start)  # time in seconds
