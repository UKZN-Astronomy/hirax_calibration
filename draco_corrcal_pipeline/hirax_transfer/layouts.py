'''Update this file in hirax_transfer/hirax_transfer to be able to run prod_params.yaml'''

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

from caput import config
from drift.util import util

def fetch_layout(conf):
    ltype = conf['type']
    return LAYOUT_TYPES[ltype].from_config(conf)


class SquareGridLayout_scatter(config.Reader):

    spacing = config.Property(proptype=float, default=6.)
    grid_size = config.Property(proptype=int, default=3)
    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.meshgrid(
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size),
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size))
        col_unscattered=np.column_stack((pos_u.T.flat, pos_v.T.flat))
        x=col_unscattered[:,0]
        y=col_unscattered[:,1]
        Ndish = self.grid_size**2
        x_scattered=x+np.load('/home/zahra/corrcal2/random_pt01_x_50.npy')[:Ndish] # you have to save and load arrays with x and y dish position scatter
        # otherwise the positions will change each time you load the config file in which you use the layout
        y_scattered=y+np.load('/home/zahra/corrcal2/random_pt01_y_50.npy')[:Ndish]
        col_scattered=np.column_stack((x_scattered.T.flat, y_scattered.T.flat))
        return col_scattered

class SquareGridLayout_pt1_scatter(config.Reader):

    spacing = config.Property(proptype=float, default=6.)
    grid_size = config.Property(proptype=int, default=3)
    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.meshgrid(
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size),
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size))
        col_unscattered=np.column_stack((pos_u.T.flat, pos_v.T.flat))
        x=col_unscattered[:,0]
        y=col_unscattered[:,1]
        Ndish = self.grid_size**2
        x_scattered=x+np.load('/home/zahra/corrcal2/random_pt1_x_50.npy')[:Ndish]
        y_scattered=y+np.load('/home/zahra/corrcal2/random_pt1_y_50.npy')[:Ndish]
        col_scattered=np.column_stack((x_scattered.T.flat, y_scattered.T.flat))
        return col_scattered

class SquareGridLayout(config.Reader):

    spacing = config.Property(proptype=float, default=6.)
    grid_size = config.Property(proptype=int, default=3)
    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.meshgrid(
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size),
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size))
        col_unscattered=np.column_stack((pos_u.T.flat, pos_v.T.flat))
        return col_unscattered

class Subgrid(config.Reader):
    #generates blocks of dishes
    short_spacing=config.Property(proptype=int, default=6) #space between dishes within a block
    long_spacing=config.Property(proptype=int, default=9) #space between blocks
    Nsub_x=config.Property(proptype=int, default=4) #number of blocks in x
    Nsub_y=config.Property(proptype=int, default=4) #number of blocks in y
    Nx=config.Property(proptype=int, default=32) #total number of dishes in x
    Ny=config.Property(proptype=int, default=32) #total number of dishes in y

    @util.cache_last
    def __call__(self):
        subgrid_x_arr = np.arange(0, self.short_spacing*self.Nx/self.Nsub_x, self.short_spacing)
        subgrid_y_arr = np.arange(0, self.short_spacing*self.Ny/self.Nsub_y, self.short_spacing)

        blocks_x_arr = np.arange(0, self.Nsub_x)*(self.short_spacing*(self.Nx/self.Nsub_x-1) + self.long_spacing)
        blocks_y_arr = np.arange(0, self.Nsub_y)*(self.short_spacing*(self.Ny/self.Nsub_y-1) + self.long_spacing)

        x_arr = [subgrid_x_arr+b for b in blocks_x_arr]
        y_arr = [subgrid_y_arr+b for b in blocks_y_arr]

        gridx,gridy=np.meshgrid(x_arr, y_arr)
        flatx=np.ndarray.flatten(gridx)
        colx=flatx.reshape(len(flatx),1)
        flaty=np.ndarray.flatten(gridy)
        coly=flaty.reshape(len(flaty),1)
        pos_full=np.hstack((colx,coly))
        return pos_full

class HexGridLayout(config.Reader): #compact hex grid at the moment
    spacing = config.Property(proptype=int, default=6)
    u = config.Property(proptype=int, default=3)
    v = config.Property(proptype=int, default=3) #might not work if u and/or v <3


    @util.cache_last
    def __call__(self):
        pos_even=np.array([])
        half_v_up=-(-self.v//2)
        half_v_down=self.v//2
        for i in range(0,self.u*self.spacing,self.spacing):
            for j in range(0,half_v_up*2*self.spacing,2*self.spacing):
                sing_pos_even=i,j
                pos_even=np.append(pos_even,sing_pos_even)
        pos_even=pos_even.reshape(self.u*half_v_up,2)

        pos_odd=np.array([])
        for i in range(np.int(self.spacing/2),np.int(self.spacing/2+self.u*self.spacing),self.spacing):
            for j in range(self.spacing,np.int(self.spacing+half_v_down*2*self.spacing),2*self.spacing):
                sing_pos_odd=i,j
                pos_odd=np.append(pos_odd,sing_pos_odd)
        pos_odd=pos_odd.reshape(half_v_down*self.u,2)

        row_even,col_even=pos_even.shape
        row_odd,col_odd=pos_odd.shape
        #pos_full=np.zeros((self.u*self.v,2))
        #alternateMerge(pos_even, pos_odd, row_even, row_odd, pos_full)
        pos_full=np.vstack((pos_even,pos_odd))

        #pos_full[::2]=pos_even
        #pos_full[1::2]=pos_odd
        return pos_full

class LayoutFile_small(config.Reader):

    #filename = config.Property(proptype=str)
    filename_1 = config.Property(proptype=str)
    filename_2 = config.Property(proptype=str)
    filename_3 = config.Property(proptype=str)

    @util.cache_last
    def __call__(self):
        #pos_u, pos_v = np.loadtxt(self.filename, unpack=True)
        pos_1 = np.load(self.filename_1)
        pos_2=np.load(self.filename_2)
        pos_3=np.load(self.filename_3)
        #posu=np.append(np.append(pos_u1,pos_u2),pos_u3)
        #posv=np.append(np.append(pos_v1,pos_v2),pos_v3)
        return np.vstack((pos_1, pos_2,pos_3))


class LayoutFile_large(config.Reader):

    filename_1x = config.Property(proptype=str)
    filename_1y = config.Property(proptype=str)
    filename_2x = config.Property(proptype=str)
    filename_2y = config.Property(proptype=str)
    filename_3x = config.Property(proptype=str)
    filename_3y = config.Property(proptype=str)

    @util.cache_last
    def __call__(self):
        #pos_u, pos_v = np.loadtxt(self.filename, unpack=True)
        pos_u1 = np.loadtxt(self.filename_1x, unpack=True)
        pos_v1 = np.loadtxt(self.filename_1y, unpack=True)
        pos_u2=np.loadtxt(self.filename_2x, unpack=True)
        pos_v2=np.loadtxt(self.filename_2y, unpack=True)
        pos_u3=np.loadtxt(self.filename_3x, unpack=True)
        pos_v3=np.loadtxt(self.filename_3y, unpack=True)

        posu=np.append(np.append(pos_u1,pos_u2),pos_u3)
        posv=np.append(np.append(pos_v1,pos_v2),pos_v3)
        return np.column_stack((posu, posv))


LAYOUT_TYPES = {
    'file_small': LayoutFile_small,
    'file_large': LayoutFile_large,
    'square_grid': SquareGridLayout,
    'square_grid_pt1': SquareGridLayout_pt1_scatter,
    'square_grid_scatter': SquareGridLayout_scatter,
    'hex_grid': HexGridLayout,
    'sub_grid': Subgrid,
}
