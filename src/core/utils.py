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
   

import math

import matplotlib.pyplot as plt
import numpy as np
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
    pass

