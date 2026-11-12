import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
# 生成gif
def compose_gif(dir,output_name):
    # files=os.listdir(dir)
    # img_paths = [f'./output/h_dt=0.001_h0=0.01_nb=0.012/{file}' for file in files]
    img_paths = [f'./output/Fr_dt=0.001_h0=0.01_nb=0.012/Fr.0{file}.0.png' for file in range(1,10)]
    img_paths += [f'./output/Fr_dt=0.001_h0=0.01_nb=0.012/Fr.{file}.0.png' for file in range(10,100)]
    gif_images = [imageio.imread(path) for path in img_paths]
    print(len(gif_images))
    imageio.mimsave(f"{output_name}.gif", gif_images, fps=5)

# 判断点是否在凸四边形内部
class isPointInQuadrangle(object):

    def __int__(self):
        self.__isInQuadrangleFlag = False

    def cross_product(self, xp, yp, x1, y1, x2, y2):
        return (x2 - x1) * (yp - y1)-(y2 - y1) * (xp - x1)

    def compute_para(self, xp, yp, xa, ya, xb, yb, xc, yc, xd, yd):
        cross_product_ab = isPointInQuadrangle().cross_product(xp, yp, xa, ya, xb, yb)
        cross_product_bc = isPointInQuadrangle().cross_product(xp, yp, xb, yb, xc, yc)
        cross_product_cd = isPointInQuadrangle().cross_product(xp, yp, xc, yc, xd, yd)
        cross_product_da = isPointInQuadrangle().cross_product(xp, yp, xd, yd, xa, ya)
        return cross_product_ab,cross_product_bc,cross_product_cd,cross_product_da

    def is_in_rect(self, aa, bb, cc, dd):
        if (aa > 0 and bb > 0 and cc > 0 and dd > 0) or (aa < 0 and bb < 0 and cc < 0 and dd < 0):
            # print("This point is in the Quadrangle.")
            self.__isInQuadrangleFlag= True
        else:
            # print("This point is not in the Quadrangle.")
            self.__isInQuadrangleFlag = False

        return self.__isInQuadrangleFlag
    

import math

import matplotlib.pyplot as plt
import numpy as np


# 全象限 没用到
def BresenhamAlgorithm(x0_original,y0_original,x1_original,y1_original,dx,dy):
    x0=int(x0_original/dx)
    y0=int(y0_original/dy)
    x1=ceil(x1_original/dx)
    y1=ceil(y1_original/dy)
    # 1.process parallel situation
    if x0 == x1 and y0 == y1:
        return [x0, y0]
    elif x0 == x1:
        if y0 < y1:
            y_min = y0
            y_max = y1
        else:
            y_min = y1 + 1
            y_max = y0 + 1

        result = []
        for y in range(y_min, y_max):
            result.append([x0, y])
        return result
    elif y0 == y1:
        if x0 < x1:
            x_min = x0
            x_max = x1
        else:
            x_min = x1 + 1
            x_max = x0 + 1
        result = []
        for x in range(x_min, x_max):
            result.append([x, y0])
        return result

    origin_x = x0
    origin_y = y0
    x0 -= origin_x
    y0 -= origin_y
    x1 -= origin_x
    y1 -= origin_y

    situation = 0
    if x1 > x0 and y1 > y0:
        if (y1 - y0) < (x1 - x0):
            situation = 11
        else:
            situation = 12
    elif x1 < x0 and y1 > y0:
        if (y1 - y0) < (x0 - x1):
            situation = 24
        else:
            situation = 23
    elif x1 < x0 and y1 < y0:
        if (y0 - y1) < (x0 - x1):
            situation = 35
        else:
            situation = 36
    elif x1 > x0 and y1 < y0:
        if (y0 - y1) < (x1 - x0):
            situation = 48
        else:
            situation = 47

    # transform quadrant-2/3/4 to quadrant-1, or transform back
    def Swap(xt, yt):
        tmp = xt
        xt = yt
        yt = tmp
        return xt, yt

    def TransformQuadrant(xt, yt, pose, back=False):
        if pose == 12:
            xt, yt = Swap(xt, yt)
        elif pose == 23:
            xt = -xt
            xt, yt = Swap(xt, yt)
        elif pose == 24:
            xt = -xt
        elif pose == 35:
            xt = -xt
            yt = -yt
        elif pose == 36:
            xt = -xt
            yt = -yt
            xt, yt = Swap(xt, yt)
        elif pose == 47:
            yt = -yt
            xt, yt = Swap(xt, yt)
        elif pose == 48:
            yt = -yt

        if back:
            if pose == 23 or pose == 47:
                xt = -xt
                yt = -yt
        return xt, yt

    # 3. transform to quadrant-1_1
    # print(f"before {x1 - x0}, {y1 - y0}")
    x1, y1 = TransformQuadrant(x1, y1, situation)
    # print(f"after {x1 - x0}, {y1 - y0}")

    # 4. compute grid in line
    delta_x = x1 - x0
    delta_y = y1 - y0
    error = 0
    y = y0
    result = []
    for x in range(x0, x1):
        result.append([x, y])
        error += delta_y
        if 2 * error > delta_x:
        # if  error > delta_x:
            # result.append([x, y])
            error -= delta_x
            y += 1

    # 5. transform back to original quadrant
    for pos in result:
        pos[0], pos[1] = TransformQuadrant(pos[0], pos[1], situation, True)

    for pos in result:
        pos[0] += origin_x
        pos[1] += origin_y
        
    return result


import jax.numpy as jnp


# 输出为tecplot格式文件
def show_tecplot_2d(casename,file_name,x,y,nx,ny,var,var_name='h'):
     current_dir = os.getcwd()
     subfolder_name = var_name+'_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output_{file_name}.dat'
     full_path = os.path.join(subfolder_path, filename)
     # data = np.column_stack((jnp.repeat(x[:, 0], ny), np.tile(y[0, :], nx), var[:, :, 0].flatten()))
     data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), var.flatten('F')))
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "{var_name}"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.9f')


def show_tecplot_2d_2(casename,time,dt0,x,y,nx,ny,h,u):
     """
     输出三个量h,u,v
     """
     current_dir = os.getcwd()
     subfolder_name = 'huv_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output{time*dt0}.dat'
     full_path = os.path.join(subfolder_path, filename)

     data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), h.at[:,:,0].get().flatten('F'),u.at[:,:,0].get().flatten('F'),u.at[:,:,1].get().flatten('F')))
    #  data=data[~np.isnan(data).any(axis=1)]
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "h" "u" "v"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.9f')
    #  h=h.at[jnp.isnan(h)].set(0)
    #  u=u.at[jnp.isnan(u)].set(0)
def stop():
    import sys
    sys.exit()
    
if __name__ == '__main__':

    # casename="集水区地形"
    # time=0
    # dt0=0.01
    # h0=10
    # L0=7000
    # B0=6000
    # L=L0/h0
    # B=B0/h0
    # halo = 0
    # nx = 394+ 2 * halo
    # ny = 345 + 2 * halo
    # dx = L / (nx - 1 - 2 * halo)
    # dy = B / (ny - 1 - 2 * halo)
    # x = jnp.arange(0, L+ 0.5* dx , dx).reshape((nx, 1)) 
    # y = jnp.arange(0, B+ 0.5*dy , dy).reshape((1,ny)) 
    # zb=jnp.zeros((nx, ny, 1))
    # from utils_mesh import get_zb,draw_img,get_edge_index_of_mesh,get_index_of_mesh,generate_house_S12,get_zb_by_file
    # zb_orginal=get_zb_by_file('gis_data/haltwistle25mrectangle3.tif').transpose()[:,::-1]

    # # zb=zb.at[:,:,0].set(get_zb_by_file('./gis_data/haltwistle25mrectangle.tif').transpose()[:,::-1])
    # zb=zb.at[:,:,0].set(zb_orginal[:,:])
    # # pd.DataFrame(zb.at[:,:,0].get()).to_excel("urbanzbed.xlsx",index=False,header=False)
    # zb=zb.at[zb<0].set(79.04) # 处理空值
    # g=9.81
    # coef_u_transform=math.sqrt(2)/math.sqrt(g*h0)
    # current_dir = os.getcwd()
    # subfolder_name = 'zbed_'+casename
    # subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
    # os.makedirs(subfolder_path, exist_ok=True)
    # filename = f'output{time*dt0}_zbed.dat'
    # full_path = os.path.join(subfolder_path, filename)
    # x =x*10
    # y=y*10
    # data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), zb.at[:,:,0].get().flatten('F')))
    # #  data=data[~np.isnan(data).any(axis=1)]
    # with open(full_path, 'w') as file:
    #     file.write('TITLE="某集水区"\n')
    #     file.write(f'VARIABLES="X" "Y" "zbed"\n')
    #     file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
    #     np.savetxt(file, data, fmt='%.5f')


    # pass
    output_name="验证自动处理边界几何分类程序正确性_自动处理方块边界无滑移边界_2"
    dir=f'./output/h_{output_name}/'
    files=os.listdir(dir)
    # img_paths = [f'./output/h_dt=0.001_h0=0.01_nb=0.012/{file}' for file in files]
    img_paths = [f'./output/h_{output_name}/h.0{file}.0.png' for file in range(1,10)]
    img_paths += [f'./output/h_{output_name}/h.{file}.0.png' for file in range(10,len(files)+1)]
    gif_images = [imageio.imread(path) for path in img_paths]
    print(len(gif_images))
    imageio.mimsave(f"./output/gifs/{output_name}.gif", gif_images, fps=10)
