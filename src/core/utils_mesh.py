from math import sqrt,ceil

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

def get_index_of_mesh(Nodetype,dx,dy,h0,points_coordinates,shape_type="rectangle"):
    '''
    坐标系，左下角为(0,0)，水平为x，纵向为y
    points_coordinates:矩形的四个角,左上，右上，右下、左下
    return :处理过的所有网格的节点类型
    '''

    if shape_type=="rectangle":
        x0,y0,  x1,y1,  x2,y2,  x3,y3=[i/h0 for i in points_coordinates]
        # 缩小范围
        x_min=min(x0,x1,x2,x3)
        x_max=max(x0,x1,x2,x3)
        y_min=min(y0,y1,y2,y3)
        y_max=max(y0,y1,y2,y3)
        # 待考虑清楚
        left=int(x_min/dx) 
        right=ceil(x_max/dx)
        top=ceil(y_max/dy)
        bottom=int(y_min/dy)
        # left=ceil(x_min/dx) +1
        # right=int(x_max/dx)
        # top=int(y_max/dy)
        # bottom=ceil(y_min/dy)+1
        # print(left,right,bottom,top)
        # Nodetype[left:right,bottom:top]=2

        # 矩形实际的坐标转换成网格索引
        xa, ya, xb, yb, xc, yc, xd, yd= x0/dx,y0/dy,x1/dx,y1/dy,x2/dx,y2/dy,x3/dx,y3/dy
        # print(xa, ya, xb, yb, xc, yc, xd, yd)
        for i in range(left,right):
            for j in range(bottom,top):
                aa, bb, cc, dd = isPointInQuadrangle().compute_para(i, j,  xa, ya, xb, yb, xc, yc, xd, yd)
                flag=isPointInQuadrangle().is_in_rect(aa, bb, cc, dd)
                if flag:
                    Nodetype[i,j]=-1 # 固体
        return Nodetype

def get_edge_index_of_mesh(dx,dy,h0,points_coordinates,shape_type="rectangle"):
    '''
    仅适用于水平放置的矩形
    points_coordinates:矩形的四个角,左上，右上，右下、左下
    return :处理过的所有网格的节点类型
    '''
    if shape_type=="rectangle":
        x0,y0,  x1,y1,  x2,y2,  x3,y3=[i/h0 for i in points_coordinates]
        
        x_min=min(x0,x1,x2,x3)
        x_max=max(x0,x1,x2,x3)
        y_min=min(y0,y1,y2,y3)
        y_max=max(y0,y1,y2,y3)

        left=ceil(x_min/dx)
        right=int(x_max/dx)
        top=int(y_max/dy)
        bottom=ceil(y_min/dy)

        # x开始，x结束，y开始，y结束
        return [
            [left,left+1,top,top+1], # LEFT_TOP_O
            [left+1,right,top,top+1], # TOP_O
            [right,right+1,top,top+1], # RIGHT_TOP_O

            [right,right+1,bottom+1,top], # RIGHT_O

            [right,right+1,bottom,bottom+1], # RIGHT_BOTTOM_O
            [left+1,right,bottom,bottom+1], # BOTTOM_O
            [left,left+1,bottom,bottom+1], # LEFT_BOTTOM_O

            [left,left+1,bottom+1,top], # LEFT_O
        ]
    
def check_edge(Nodetype,edge):
    for i in range(8):
        x_start,x_end,y_start,y_end=edge[i]
        Nodetype[x_start:x_end,y_start:y_end]=1

    return Nodetype

def generate_house():
    result=[]
    # 竖放的房子
    xa, ya, xb, yb, xc, yc, xd, yd=[0.06,0.73,0.36,0.73,0.36,0.53,0.06,0.53]
    for i in range(4):
        for j in range(2):
           result.append([xa+i*0.61,ya+j*0.72,
                          xb+i*0.61,yb+j*0.72,
                          xc+i*0.61,yc+j*0.72,
                          xd+i*0.61,yd+j*0.72,
                          ]) 
    
    # xa, ya, xb, yb, xc, yc, xd, yd=[0.12,0.39,0.42,0.39,0.42,0.59,0.12,0.59]
    xa, ya, xb, yb, xc, yc, xd, yd=[0.39,0.42,0.59,0.42,0.59,0.12,0.39,0.12]
    for i in range(4):
        for j in range(3):
           result.append([xa+i*0.62,ya+j*0.73,
                          xb+i*0.62,yb+j*0.73,
                          xc+i*0.62,yc+j*0.73,
                          xd+i*0.62,yd+j*0.73,
                          ]) 

    return result

def generate_house2():
    result=[]
    # 竖放的房子
    # xa, ya, xb, yb, xc, yc, xd, yd=[0.1625,0.1,0.1625,0.3,0.4625,0.3,0.4625,0.1]
    xa, ya, xb, yb, xc, yc, xd, yd=[0.1625,0.125,0.1625,0.325,0.4625,0.325,0.4625,0.125]
    # xa, ya, xb, yb, xc, yc, xd, yd=[0.15,0.2,0.15,0.4,0.4,0.4,0.4,0.2]
    for i in range(4):
        for j in range(5):
           result.append([xa+i*0.625,ya+j*0.4,
                          xb+i*0.625,yb+j*0.4,
                          xc+i*0.625,yc+j*0.4,
                          xd+i*0.625,yd+j*0.4,
                          ]) 
    

    return result

def generate_house3():
    result=[]
    # 竖放的房子

    xa, ya, xb, yb, xc, yc, xd, yd=[0.469,0.406,0.669,0.406,0.669,0.106,0.469,0.106]
    for i in range(3):
        for j in range(4):
           result.append([xa+i*0.692,ya+j*0.497,
                          xb+i*0.692,yb+j*0.497,
                          xc+i*0.692,yc+j*0.497,
                          xd+i*0.692,yd+j*0.497,
                          ]) 
    

    return result


def generate_house_X20():
    result=[]
    # 竖放的房子

    xa, ya, xb, yb, xc, yc, xd, yd=[0.13338,0.40338,0.33338,0.40338,0.33338,0.10338,0.13338,0.10338]
    for i in range(5):
        for j in range(4):
           result.append([xa+i*0.5,ya+j*0.5,
                          xb+i*0.5,yb+j*0.5,
                          xc+i*0.5,yc+j*0.5,
                          xd+i*0.5,yd+j*0.5,
                          ]) 
    

    return result

def generate_house_S12():
    result=[]
    # 竖放的房子

    xa, ya, xb, yb, xc, yc, xd, yd=[0.20694,0.73608,0.50694,0.73608,0.50694,0.53608,0.20694,0.53608]
    for i in range(3):
        for j in range(2):
           result.append([xa+i*0.89815,ya+j*0.73431,
                          xb+i*0.89815,yb+j*0.73431,
                          xc+i*0.89815,yc+j*0.73431,
                          xd+i*0.89815,yd+j*0.73431,
                          ]) 
           
    xa, ya, xb, yb, xc, yc, xd, yd=[0.7141,0.42329,0.9141,0.42329,0.9141,0.12329,0.7141,0.12329]
    for i in range(2):
        for j in range(3):
           result.append([xa+i*0.89349,ya+j*0.73335,
                          xb+i*0.89349,yb+j*0.73335,
                          xc+i*0.89349,yc+j*0.73335,
                          xd+i*0.89349,yd+j*0.73335,
                          ]) 
    

    return result

def get_zb_by_file(file):
    import rasterio

    with rasterio.open(file,'r') as img:
        # 获取数组
        array = img.read()

    first_band = array[0]

    return first_band

def draw_img(array,output):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(array,cmap='jet',interpolation='None')
    plt.colorbar()
    plt.savefig(output)
    plt.clf()


if __name__ == '__main__':
    
    nx=251
    ny=201
    dx=0.01
    dy=0.01
    import numpy as np
    Nodetype=np.zeros((nx,ny))
    points=generate_house()
    for point in points:
        Nodetype=get_index_of_mesh(Nodetype,nx,ny,dx,dy,point,shape_type="rectangle")
        edge=get_edge_index_of_mesh(dx,dy,point,shape_type="rectangle")
        Nodetype=check_edge(Nodetype,edge)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(Nodetype,interpolation='None')
    plt.colorbar()
    plt.savefig("Topography.png")
    plt.clf()
    pass
    import pandas as pd
    pd.DataFrame(   Nodetype).to_excel('Nodetype.xlsx',header=None,index=None)



    #****************************************************************************************************
    # import rasterio
 
    # # 打开GeoTIFF文件
    # with rasterio.open('./gis_data/dem.tif') as img:
    #     # 获取数组
    #     array = img.read()
    
    # # 打印数组形状
    # print(array.shape)
    
    # # 如果GeoTIFF有多个波段，array是一个包含多个波段的数组
    # # 你可以通过指定波段索引来获取特定波段
    # # 例如，获取第一个波段
    # first_band = array[0]
    # first_band=first_band.transpose()
    # import matplotlib.pyplot as plt
    first_band=get_zb()
    plt.clf()
    plt.imshow(first_band)
    plt.colorbar()
    plt.savefig("first_band.transpose.png")

    # plt.show()
    plt.clf()
    # first_band=first_band[::-1,:]
    # # first_band[-1,:]=first_band[-2,:]

    # plt.clf()
    # plt.imshow(first_band)
    # plt.savefig("dem.png")
    # plt.clf()
    #**********************************************************************************
    # img_bounds = img.bounds  # 影像四至
    # img_rows, img_cols = img.shape  # 影像行列号
    # img_bands = img.count  # 影像波段数
    # img_indexes = img.indexes  # 影像波段
    # img_crs = img.crs  # 影像坐标系
    # img_transform = img.transform  # 影像仿射矩阵
    # upper_left = img.transform * (0, 0)  # 影像左上角像元的左上角点经纬度
    # lower_right = img.transform * (img.width, img.height)  # 影像右下角像元的右下角点经纬度
    # img_meta = img.meta  # 影像基础信息，包含driver、数据类型、坐标系、仿射矩阵等

    # new_tif_path = r'./gis_data/band1.tif'
  
    # new_img = rasterio.open(new_tif_path,
    #                     'w',
    #                     driver='GTiff',
    #                     height=img_rows,
    #                     width=img_cols,
    #                     count=1,
    #                     crs=img.crs,
    #                     transform=img.transform,
    #                     dtype=first_band.dtype,
    #                     nodata=0)
    # new_img.write(first_band, 1)  # 写入波段数据
    # new_img.close()  # 数据存入磁盘
    pass
    


