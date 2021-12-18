################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################

import os
import sys
import csv
import h5py
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset



class Dataset_MLHDF(Dataset):
	def __init__(self, csv_path, mlhdf_path, cmlhdf_path, max_atoms=1000, feat_dim=22, feat_int=False, affine_trans=True):
		super(Dataset_MLHDF, self).__init__()
		self.csv_path = csv_path
		self.mlhdf_path = mlhdf_path
		self.cmlhdf_path = cmlhdf_path
		self.max_atoms = max_atoms
		self.feat_dim = feat_dim
		self.feat_int = feat_int
		self.affine_trans = affine_trans

		self.data_info_list = []
		with open(self.csv_path, 'r') as fp:
			csv_reader = csv.reader(fp, delimiter=',')
			next(csv_reader)
			for row in csv_reader:
				self.data_info_list.append([row[0], row[1], float(row[3]), int(row[4])])
				
		self.mlhdf = h5py.File(self.mlhdf_path, 'r')
		self.cmlhdf = h5py.File(self.cmlhdf_path, 'r')
		for pdbid in self.cmlhdf.keys():
			self.data_info_list.append([pdbid, 0, 0, 1])

	def close(self):
		self.mlhdf.close()
		self.cmlhdf.close()

	def __len__(self):
		count = len(self.data_info_list)
		return count

	def __getitem__(self, idx):
		pdbid, poseid, affinity, poselabel = self.data_info_list[idx]

		if poseid == 0:
			mlhdf_ds = self.cmlhdf[pdbid]["pybel"]["processed"]["crystal"]
		else:
			mlhdf_ds = self.mlhdf[pdbid]["pybel"]["processed"]["docking"][poseid]
		actual_data = mlhdf_ds["data"][:]

		if self.affine_trans:
			angle = np.radians((np.random.rand(3)*2.0-1.0)*20) # +/- 20
			t_mat = (np.random.rand(3)*2.0-1.0)*10  # +/- 10 angstrom
			r_mat = R.from_euler('xyz', angle).as_matrix()
			actual_xyz = actual_data[:,:3]
			actual_xyzt = np.matmul(actual_xyz, r_mat) + t_mat
			actual_data[:,:3] = actual_xyzt

		# normalize data (no scaling needed because we need absolute scale in angstrom)
		#xyz_center = np.mean(actual_data[:,:3], axis=0)
		#actual_data[:,:3] -= xyz_center
		#xyz_center = np.mean(actual_data[:,:3], axis=0)

		data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
		data[:actual_data.shape[0],:] = actual_data
		if self.feat_int:
			ifeat = mlhdf_ds["ifeat"][:]
		else:
			ifeat = np.zeros((4,), dtype=np.float32)

		x = torch.tensor(np.expand_dims(data, axis=0))
		y = torch.tensor(np.expand_dims(poselabel, axis=0))
		x_ifeat = torch.tensor(ifeat)
		return x, x_ifeat, y

