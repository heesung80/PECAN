################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################


import os
import sys
import numpy as np
import scipy as sp
import scipy.ndimage
import h5py
import csv
import torch
import h5py
import base64
import pandas as pd
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


# In[34]:


class Dataset_to_3d(Dataset):
    def __init__(self, file_name, csv_file, cfile_name="", i_feature=False, relative_size=True, size_angstrom=48, size_dim=48, atom_radius=1, atom_radii=None, sigma=1,affine_trans = False):
        super(Dataset_to_3d, self).__init__()

        self.pose_list = []
        self.vol_dim = [size_dim, size_dim, size_dim, 19 ]
        self.hdf = h5py.File(file_name, 'r')
        if len(cfile_name) > 0:
            self.chdf = h5py.File(cfile_name, 'r')
        self.i_feature = i_feature
        self.relative_size = relative_size
        self.size_angstrom = size_angstrom
        self.size_dim = size_dim
        self.atom_radius = atom_radius
        self.atom_radii= atom_radii
        self.sigma = sigma
        self.affine_trans = affine_trans
        
        with open(csv_file, 'r',encoding="ISO-8859-1") as f:        
            input_lines = f.readlines()
            for line in input_lines[1:]:
                l = line.split(",")
                pdbid = l[0]
                poseid = l[1]
                poselabel = int(l[4])
                self.pose_list.append((pdbid, poseid, poselabel))
        crystal_input_idlist =list(self.chdf.keys())
        #print(crystal_input_idlist)
        for input_cid in crystal_input_idlist:
            #print(input_cid)
            poseid = 0
            poselabel = 1
            self.pose_list.append((input_cid, poseid, poselabel))
            
        #print(self.pose_list)
        print("total:", len(self.pose_list))
        
    def __len__(self):
        count = len(self.pose_list)
        return count
    
    def __getitem__(self, idx): #, pose_name):
        pdbid, poseid, poselabel = self.pose_list[idx]
        print(pdbid)
        print(poseid)
        if int(poseid) >= 1:
            input_data = self.hdf[pdbid]['pybel']['processed']['docking'][poseid]['data'][:]
        else:
            input_data = self.chdf[pdbid]['pybel']['processed']['crystal']['data'][:]
        
        if self.affine_trans:
            angle = np.radians((np.random.rand(3)*2.0-1.0)*20) # +/- 20
            t_mat = (np.random.rand(3)*2.0-1.0)*10  # +/- 10 angstrom
            r_mat = R.from_euler('xyz', angle).as_matrix()
            actual_xyz = input_data[:,:3]
            actual_xyzt = np.matmul(actual_xyz, r_mat) + t_mat
            input_data[:,:3] = actual_xyzt
            
        input_xyz = input_data[:,0:3]
        input_feat = input_data[:,3:]
        output_3d_data = self.__get_3D__(input_xyz, input_feat, self.vol_dim,
                                         self.relative_size, self.size_angstrom, self.atom_radii,
                                         self.atom_radius, self.sigma)
        i_tensor = torch.tensor(output_3d_data)
        input_tensor = i_tensor.permute(3,0,1,2)
        #output_tensor = torch.tensor(int(poselabel))
        output_tensor = torch.tensor(np.expand_dims(int(poselabel), axis=0))
        if self.i_feature == False:
            return input_tensor, output_tensor
        else:
            if int(poseid) >= 1:
                ifeat = self.hdf[pdbid]['pybel']['processed']['docking'][poseid]['ifeat'][:]
                #lfeat = self.hdf[pdbid]['pybel']['processed']['docking'][poseid]['lfeat'][:]
            else:
                ifeat = self.chdf[pdbid]['pybel']['processed']['crystal']['ifeat'][:]
                #lfeat = self.chdf[pdbid]['pybel']['processed']['crystal']['lfeat'][:]
            #interaction_data = np.array([ifeat, lfeat], dtype=np.float32)
            input_tensor_ifeat = torch.tensor(ifeat)
            #input_tensor_lfeat = torch.tensor(lfeat)
            input_tensor_lfeat = torch.tensor(np.zeros((200,), dtype=np.float32))
            return input_tensor, output_tensor, input_tensor_ifeat, input_tensor_lfeat
        
    
    def __get_3D__(self, xyz, feat, vol_dim, relative_size, size_angstrom, atom_radii, atom_radius, sigma):

        # get 3d bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = self.__get_3D_bound__(xyz)

        # initialize volume
        vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

        if relative_size:
            # voxel size (assum voxel size is the same in all axis
            vox_size = float(zmax - zmin) / float(vol_dim[0])
        else:
            vox_size = float(size_angstrom) / float(vol_dim[0])
            xmid = (xmin + xmax) / 2.0
            ymid = (ymin + ymax) / 2.0
            zmid = (zmin + zmax) / 2.0
            xmin = xmid - (size_angstrom / 2)
            ymin = ymid - (size_angstrom / 2)
            zmin = zmid - (size_angstrom / 2)
            xmax = xmid + (size_angstrom / 2)
            ymax = ymid + (size_angstrom / 2)
            zmax = zmid + (size_angstrom / 2)
            vox_size2 = float(size_angstrom) / float(vol_dim[0])
            #print(vox_size, vox_size2)

        # assign each atom to voxels
        for ind in range(xyz.shape[0]):
            x = xyz[ind, 0]
            y = xyz[ind, 1]
            z = xyz[ind, 2]
            if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
                continue

            # compute van der Waals radius and atomic density, use 1 if not available
            if not atom_radii is None:
                vdw_radius = atom_radii[ind]
                atom_radius = 1 + vdw_radius * vox_size

            cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
            cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
            cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

            vx_from = max(0, int(cx - atom_radius))
            vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
            vy_from = max(0, int(cy - atom_radius))
            vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
            vz_from = max(0, int(cz - atom_radius))
            vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

            for vz in range(vz_from, vz_to + 1):
                for vy in range(vy_from, vy_to + 1):
                    for vx in range(vx_from, vx_to + 1):
                            vol_data[vz, vy, vx, :] += feat[ind, :]

        # gaussian filter
        if sigma > 0:
            for i in range(vol_data.shape[-1]):
                vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

        return vol_data
    
    def __get_3D_bound__(self,xyz_array):
        xmin = min(xyz_array[:, 0])
        ymin = min(xyz_array[:, 1])
        zmin = min(xyz_array[:, 2])
        xmax = max(xyz_array[:, 0])
        ymax = max(xyz_array[:, 1])
        zmax = max(xyz_array[:, 2])
        return xmin, ymin, zmin, xmax, ymax, zmax












