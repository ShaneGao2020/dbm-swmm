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
from matplotlib.colors import ListedColormap

pretty.install()
Utility.SetPrecision("64")
Utility.SetPlatform("cpu")
print("JAX local devices: ", jax.local_devices())
print("Jax device", jax.devices())


import matplotlib.pyplot as plt

from matplotlib import cm

###### Flow definition #########################################################
# Total number of time iterations.
lattice: DVS = Model.DefineDVS("D2Q16")


casename='2DRightAngleCornerTransitionalFlow_case1_nb=0.01'
Fr_output_dir="./output/Fr_"+casename
h_output_dir="./output/h_"+casename
if not os.path.exists(Fr_output_dir):
    os.makedirs(Fr_output_dir)
if not os.path.exists(h_output_dir):
    os.makedirs(h_output_dir)


# 物理量值(无量纲之前)

L0=5.655
B0=0.6
Q0=0.0037/0.21
# Hout0=0.001
dt0=0.001
nb=0.01
#***************************************
# 无量纲化
# 特征水深 
h0=0.03
g=9.81
Q=Q0*(2/(g*h0**5))**0.5  
L=L0/h0
B=B0/h0
# Hout=Hout0/h0
dt=(g*h0)**0.5/(2**0.5*h0)*dt0
# dt=0.001
tau=0.05
print(f"Qin={Q},L={L},dt={dt}")
#***************************************

halo = 0
nx = 566 + 2 * halo
ny = 61 + 2 * halo
dx = L / (nx - 1 - 2 * halo)
dy= B/ (ny -1 -2*halo)

period=int(1/dt0) 
print("1 s 迭代次数：",period)
maxIter = 100*period


# x = jnp.arange(0, L + 0.5*dx, dx).reshape((nx, 1))
x = jnp.arange(0, L+ dx , dx).reshape((nx, 1)) 
# print(x)
y= jnp.arange(0, B+ dy , dy).reshape((1, ny)) 


idx1=int(3.19/h0/dx)
idx2=int(3.4/h0/dx)
idy1=int(0.39/h0/dy)
idy2=int(0.21/h0/dy)
# print(idx1,idx2,idy1,idy2)

hInlet = jnp.ones((1, ny, 1))
# 出口水头边界条件
# hOutlet = Hout * jnp.ones((1, ny, 1))

# 初始化每个网格的水深
h: Array =jnp.ones((nx, ny, 1))
# 
u: Array = jnp.zeros((nx, ny, 2))
# u=u.at[0, idy1:, 0].set(Q/(0.21/h0*h.at[1,idy1:,0].get()))
# u=u.at[0, idy1:, 0].set(1.8837019173458103)

slopex=jnp.zeros((nx, ny, 1))
slopey=jnp.zeros((nx, ny, 1))
slopey=slopey.at[idx1+1:idx2,:,:].set(0.01)
slopex=slopex.at[:idx1+1,:,:].set(-0.01)
slopex=slopex.at[idx2:,:,:].set(-0.01)
# # 河床高度的数组
# zb=jnp.tile(jnp.linspace(L*0.01,0,nx).reshape(-1,1),(1,ny)).reshape(nx,ny,1)
# # print(zb.at[idx2,idy2:,0].get())
# # plt.imshow(zb.at[:,:,0].get())
# # plt.colorbar()
# # plt.savefig("河床高度.png")
# slopex=jnp.zeros((nx, ny, 16))
# slopey=jnp.zeros((nx, ny, 16))
# # slopey.at[idx1+1:idx2,1:-1,:].set(0.01)

# for d in range(16):
#     if lattice['XI'].at[0,d].get()>0: # 仅考虑x方向的格子速度
#         slopex=slopex.at[1:-1,:,[d]].set((zb.at[1:-1,:,:].get()-zb.at[:-2,:,:].get())/dx)
#     else:
#         slopex=slopex.at[1:-1,:,[d]].set((zb.at[2:,:,:].get()-zb.at[1:-1,:,:].get())/dx)
# print(slopex.at[:,:,:].get())

# NodeTypes=jnp.zeros((nx, ny, 1)) # 0为液体、2为固体、1为边界
# NodeTypes=NodeTypes.at[:idx1+1,idy1,0].set(1)
# NodeTypes=NodeTypes.at[:idx1,:idy1,0].set(2)
# NodeTypes=NodeTypes.at[idx2:,idy2,0].set(1)
# NodeTypes=NodeTypes.at[idx2+1:,idy2+1:,0].set(2)
# NodeTypes=NodeTypes.at[idx1,:idy1,0].set(1)
# NodeTypes=NodeTypes.at[idx2,idy2+1:,0].set(1)

# plt.clf()
# plt.imshow(NodeTypes.at[:,:,0].get(),cmap= ListedColormap(['blue','red','grey']))
# cbar = plt.colorbar(ticks=[0.33, 1, 1.66],)
# # cbar = plt.colorbar(ticks=[0, 1, 2],orientation='horizontal')
# cbar.set_ticklabels(['Fluid', 'Boundary', 'Solid'])
# plt.savefig("工况.png")
# # print(NodeTypes.at[:,idy2,0].get())

# GeometryTypes=jnp.ones((nx, ny, 1))*10
# Vertex_Fluid = 10
# Vertex_NoneqExtrapol = 1013
# Vertex_FreeSlipWall = 1017
# Vertex_SolidBoundary = 1019
# Vertex_ImmersedSolid = -1
# GeometryTypes=GeometryTypes.at[NodeTypes==1].set(Vertex_FreeSlipWall)
# print(GeometryTypes.at[:,idy2,0].get())

@jit
def TreateBoundary(f):

    # inlet h 入口处边墙段
    h1, u1 = Model.CalcMacroVars(f[1:2, :, :], lattice["XIT"])
    uInlet = jnp.zeros(u1.shape)
    uInlet = uInlet.at[:, :, 0:1].set(1.531801559160329)
    f = f.at[0:1, :, :].set(
        Model.BGKEquilibriumSWE4th(
            # 入口处水深，取右边一个格子的水深，速度根据入口流量及水深计算
            hInlet,
            uInlet,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[1:2, :, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 入口处下角点
    # f = f.at[0, idy1, :].set(f.at[1, idy1+1, :].get())
    h1, u1 = Model.CalcMacroVars(f[1:2, idy1+1:idy1+2, :], lattice["XIT"])
    uInlet = jnp.zeros(u1.shape)
    # hInlet_bottom_corner= jnp.ones((1,1,1))
    uInlet = uInlet.at[:, :, 0:1].set(1.531801559160329)
    f = f.at[0:1, idy1:idy1+1, :].set(
        Model.BGKEquilibriumSWE4th(
            1,
            uInlet,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[1:2, idy1+1:idy1+2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )


    # 入口处上角点
    # f = f.at[0, -1, :].set(f.at[1, -2, :].get())
    h1, u1 = Model.CalcMacroVars(f[1:2, -2:-1, :], lattice["XIT"])
    # uInlet = jnp.zeros(u1.shape)
    # # hInlet_bottom_corner= jnp.ones((1,1,1))
    # uInlet = uInlet.at[:, :, 0].set(1.8837019173458103)
    f = f.at[0:1, -1:, :].set(
        Model.BGKEquilibriumSWE4th(
            1,
            uInlet,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[1:2, -2:-1, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )

    # # outlet h 自由出流，h和u均取左侧一个网格的值
    f = f.at[-1:, 1:idy2, :].set( f.at[-2:-1, 1:idy2, :].get())

    # 出口处下角点
    f = f.at[-1, 0, :].set(f.at[-2, 1, :].get())
   
    # 出口处上角点
    # f = f.at[-1, idy2, :].set(f.at[-1, idy2-1, :].get())
    f = f.at[-1, idy2, :].set(f.at[-2, idy2-1, :].get())
    
    

    # 底部边界
    # 左半边下边界
    h1, u1 = Model.CalcMacroVars(f[1:idx1+1, idy1+1:idy1+2, :], lattice["XIT"])
    ubottom =  u1.at[:,:,1:].set(0)
    f = f.at[1:idx1+1, idy1:idy1+1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            ubottom,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[1:idx1+1, idy1+1:idy1+2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 右半段下边界
    h1, u1 = Model.CalcMacroVars(f[idx1+1:-1, 1:2, :], lattice["XIT"])
    ubottom =  u1.at[:,:,1:].set(0)
    f = f.at[idx1+1:-1, 0:1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            ubottom,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx1+1:-1, 1:2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 上边角点
    h1, u1 = Model.CalcMacroVars(f[idx1+1:idx1+2, idy1+1:idy1+2, :], lattice["XIT"])
    ubottom =   jnp.zeros(u1.shape)
    f = f.at[idx1:idx1+1, idy1:idy1+1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            ubottom,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx1+1:idx1+2, idy1+1:idy1+2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 下边角点
    h1, u1 = Model.CalcMacroVars(f[idx1+1:idx1+2, 1:2, :], lattice["XIT"])
    ubottom =   jnp.zeros(u1.shape)
    f = f.at[idx1:idx1+1, 0:1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            ubottom,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx1+1:idx1+2, 1:2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )

    # 下边的竖边墙
    # 左侧竖边墙
    h1, u1 = Model.CalcMacroVars(f[idx1+1:idx1+2, 1:idy1, :], lattice["XIT"])
    uright =   u1.at[:,:,0:1].set(0)
    f = f.at[idx1:idx1+1, 1:idy1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            uright,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx1+1:idx1+2, 1:idy1, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )

    # 右侧
    h1, u1 = Model.CalcMacroVars(f[idx2-1:idx2, idy2+1:-1, :], lattice["XIT"])
    uleft =   u1.at[:,:,0:1].set(0)
    f = f.at[idx2:idx2+1, idy2+1:-1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            uleft,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx2-1:idx2, idy2+1:-1, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )

    # f = f.at[1:idx1+1, idy1, :].set(f.at[1:idx1+1, idy1+1, :].get())
    # f = f.at[idx1+1:, 0, :].set(f.at[idx1+1:, 1, :].get())
    # f = f.at[idx1, idy1, :].set(f.at[idx1+1, idy1+1, :].get()) 
    # f = f.at[idx1, 0, :].set(f.at[idx1+1, 1, :].get()) 

    # 顶部边界
    # f = f.at[1:idx2, -1, :].set(f.at[1:idx2, -2, :].get())
    # f = f.at[idx2:, idy2, :].set(f.at[idx2:, idy2-1, :].get())
    # 底部边界
    # 左半边上边界
    h1, u1 = Model.CalcMacroVars(f[1:idx2, -2:-1, :], lattice["XIT"])
    utop=  u1.at[:,:,1:].set(0)
    f = f.at[1:idx2,-1:, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            utop,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[1:idx2, -2:-1, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 右半段上边界
    h1, u1 = Model.CalcMacroVars(f[idx2+1:-1, idy2-1:idy2, :], lattice["XIT"])
    utop =  u1.at[:,:,1:].set(0)
    f = f.at[idx2+1:-1, idy2:idy2+1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            utop,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx2+1:-1, idy2-1:idy2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 上边角点
    h1, u1 = Model.CalcMacroVars(f[idx2-1:idx2, -2:-1, :], lattice["XIT"])
    utop =   jnp.zeros(u1.shape)
    f = f.at[idx2:idx2+1, -1:, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            utop,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx2-1:idx2, -2:-1, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )
    # 下边角点
    h1, u1 = Model.CalcMacroVars(f[idx2-1:idx2, idy2-1:idy2, :], lattice["XIT"])
    utop =   jnp.zeros(u1.shape)
    f = f.at[idx2:idx2+1, idy2:idy2+1, :].set(
        Model.BGKEquilibriumSWE4th(
            h1,
            utop,
            lattice["XI"],
            lattice["W"],
        )
        + f.at[idx2-1:idx2, idy2-1:idy2, :].get()
        - Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    )






    # # 竖边墙
    # f = f.at[idx1, 1:idy1, :].set(f.at[idx1+1, 1:idy1, :].get())
    # f = f.at[idx2, idy2+1:, :].set(f.at[idx2-1, idy2+1:, :].get())

    # # 四个角点
    # f = f.at[idx1, idy1, :].set(f.at[idx1+1, idy1+1, :].get()) 
    # f = f.at[idx1, 0, :].set(f.at[idx1+1, 1, :].get()) 
    # f = f.at[idx2, -1, :].set(f.at[idx2-1, -2, :].get()) 
    # f = f.at[idx2, idy2, :].set(f.at[idx2-1,idy2-1, :].get()) 

    # 墙体
    f=f.at[:idx1,:idy1,:].set(0)
    f=f.at[idx2+1:,idy2+1:,:].set(0)

    return f



feq = Model.BGKEquilibriumSWE4th(h, u, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)

@jit
def CalcGrad(fGrad0, f):
    return fGrad0.at[1:-1, 1:-1, :].set(
        Scheme.ConvectionUpwind1ST(f, lattice["XI"], x, y)
    )

@jit
def CalForce(u,h,W,XI ):
    F1x=-2*slopex # x方向的坡度
    F1y=-2*slopey # y方向的坡度
    F2x=-g*nb*nb*u.at[:,:,0:1].get()*(u.at[:,:,0:1].get()**2+u.at[:,:,1:].get()**2)**0.5/(h0**(1/3)*(h**(4/3)))
    F2y=-g*nb*nb*u.at[:,:,1:].get()*(u.at[:,:,0:1].get()**2+u.at[:,:,1:].get()**2)**0.5/(h0**(1/3)*(h**(4/3)))
    Fx=F1x+F2x
    Fy=F1y+F2y
    return W*h*(Fx*XI.at[0,:].get()+Fy*XI.at[1,:].get())


skip = 1

start = timer()
print("初始化完成")
# ###### Main time loop ##########################################################
for time in range(maxIter):
    # 根据入口处水深、单宽流量计算入口处x方向的速度
    u=u.at[0, :, 0].set(Q/(0.21/h0*h.at[1,:,0].get()))
    uInlet=u.at[0:1,:, :].get()

    feq = Model.BGKEquilibriumSWE4th(h, u, lattice["XI"], lattice["W"])

    f = TreateBoundary(f)
    h, u = Model.CalcMacroVars(f, lattice["XIT"])

    # h=h.at[0, :, 0].set(h.at[1, :, 0].get())

    if (time % period == 0) and (time != 0):  # Visualization
        print("time=", time)
        print("hx0=", h[0, idy1+5, 0], "ux0=", u[0, idy1+5, 0], "vx0=", u[0, idy1+5, 1])
        print("hx1=", h[1,idy1+5, 0], "ux1=", u[1, idy1+5, 0], "vx`=", u[1, idy1+5, 1])
        print("hx-2=", h[-2, idy2-5, 0], "ux-2=", u[-2, idy2-5, 0], "vx-2=", u[-2, idy2-5, 1])
        print("hx-1=", h[-1, idy2-5, 0], "ux-1=", u[-1, idy2-5, 0], "vx-1`=", u[-1, idy2-5, 1])
            # if time==0:
        # plt.clf()
        # plt.imshow(u.at[:,:,0].get())
        # plt.colorbar()
        # plt.savefig(f'u{time}.png')
        plt.clf()
        plt.imshow(h.at[:,:,0].get()*h0,cmap='jet')
        plt.colorbar()
        plt.clim(0, 0.1)
        plt.savefig(f"{h_output_dir}/h." + str(time / period).zfill(4) + ".png")
        plt.clf()
        # Fr = u.at[:, 1, 0].get() / (jnp.sqrt(2 * h.at[:, 1, 0].get()))
        # plt.xlim(0,80)
        # plt.ylim(-0.5,2)
        # plt.text(1,0.05,f"({time/period})s")
        # plt.plot(
        #     x.at[:idx2:skip, 0].get(),
        #     h.at[:idx2:skip, idy1+10, 0].get(),
        #     x.at[idx2::skip, 0].get(),
        #     h.at[idx2::skip, idy2-10, 0].get(),
        #     # x.at[::skip, 0].get(),
        #     # u.at[::skip, 1, 0].get(),
        #     # x.at[::skip, 0].get(),
        #     # Fr.at[::skip].get(),
        # )
        # plt.plot(
        #     x.at[::skip, 0].get(),
        #     zb.at[::skip, 1, 0].get(),
        #     # x.at[::skip, 0].get(),
        #     # h.at[::skip, 1, 0].get(),
        # )
        # # plt.legend(["h",'bed','only h'])
        # plt.legend(["h",'bed'])
        # plt.savefig(f"{h_output_dir}/截面h." + str(time / period).zfill(4) + ".png")
        # plt.clf()
        # # plt.xlim(0,1390)
        # # plt.ylim(-1,2.5)
        # plt.text(1,0.05,f"({time/period})s")
        # plt.plot(
        #     x.at[::skip, 0].get(),
        #     Fr.at[::skip].get(),
        # )
        # plt.legend(["Fr"])
        # plt.savefig(f"{Fr_output_dir}/Fr." + str(time / period).zfill(4) + ".png")
    fGrad = CalcGrad(fGrad0, f)

    force=CalForce(u,h,lattice["W"],lattice["XI"])
    # force=0

    f = Scheme.Euler1ST(
        f,
        fGrad,
        Model.BGKSWECollision(
            f, Model.BGKEquilibriumSWE4th(h, u, lattice["XI"], lattice["W"]), tau, h
        ),
        dt,
        force,
    )

end = timer()
print(end - start)  # time in seconds
plt.clf()
plt.imshow(h.at[:,:,0].get()*h0,cmap='jet')
plt.colorbar()
plt.clim(0, 0.1)
plt.savefig(f"{h_output_dir}/h." + str(time / period).zfill(4) + ".png")
plt.clf()

import pandas as pd
pd.DataFrame(h.at[:,:,0].get()).to_excel("q=3.7水深分布.xlsx",sheet_name='h')
pd.DataFrame(u.at[:,:,0].get()).to_excel("q=3.7 u 分布.xlsx",sheet_name='u')
pd.DataFrame(u.at[:,:,1].get()).to_excel("q=3.7 v 分布.xlsx",sheet_name='v')

# import imageio.v2 as imageio
# # 生成动图
# def compose_gif(dir,var,output_name):
#     files=os.listdir(dir)
#     img_paths = [f'{dir}/{var}.0{file}.0.png' for file in range(1,10)]
#     img_paths += [f'{dir}/{var}.{file}.0.png' for file in range(10,len(files))]
#     gif_images = [imageio.imread(path) for path in img_paths]
#     imageio.mimsave(f"{output_name}.gif", gif_images, fps=5)

# # compose_gif(Fr_output_dir,'Fr',Fr_output_dir)
# compose_gif(h_output_dir,'h',h_output_dir)
# 假如我有5列数据,columns是表头
pd.DataFrame(h.at[:,:,0].get(),columns=['TIME','POINTB','POINTC','POINTD','POINTE']).to_excel("q=3.7水深分布.xlsx",sheet_name='h',index=False)
