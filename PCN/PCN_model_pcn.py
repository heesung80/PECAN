################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.stats import truncnorm # truncnorm.rvs is equivalent to tf.truncated_normal


# to read checkpoint files from any GPU configuration
# from https://discuss.pytorch.org/t/checkpoint-in-multi-gpu/97852
def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
		return

	for key in keys:
		newkey = key[len(prefix) :]
		state_dict[newkey] = state_dict.pop(key)

	try:
		metadata = state_dict._metadata
	except AttributeError:
		pass
	else:
		for key in list(metadata.keys()):
			if len(key) == 0:
				continue
			newkey = key[len(prefix) :]
			metadata[newkey] = metadata.pop(key)


class Model_Transform(nn.Module):
	def __init__(self, input_dim=[1000,3], verbose=0):
		super(Model_Transform, self).__init__()
		self.verbose = verbose

		self.conv_block1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=[1,3], stride=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
		)
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=[1,1], stride=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(128),
		)
		self.conv_block3 = nn.Sequential(
			nn.Conv2d(128, 1024, kernel_size=[1,1], stride=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(1024),
		)
		self.dense_block1 = nn.Sequential(
			nn.Linear(1024, 3*3),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(3*3),
		)

	def forward(self, x):
		# input shape
		if self.verbose != 0:
			print(x.shape)
		
		# convolutional layers
		c1 = self.conv_block1(x)
		if self.verbose != 0:
			print(c1.shape)

		c2 = self.conv_block2(c1)
		if self.verbose != 0:
			print(c2.shape)

		c3 = self.conv_block3(c2)
		if self.verbose != 0:
			print(c3.shape)

		pooled = func.max_pool2d(c3, kernel_size=[1000, 1], stride=1, padding=0)
		if self.verbose != 0:
			print(pooled.shape)

		# flatten
		flattened = torch.reshape(pooled, [-1, 1024])
		if self.verbose != 0:
			print(flattened.shape)

		# fully connected layers
		fc1 = self.dense_block1(flattened)
		if self.verbose != 0:
			print(fc1.shape)
		
		flattened = torch.reshape(fc1, [-1, 3, 3])
		if self.verbose != 0:
			print(flattened.shape)

		return flattened


class Model_PCN(nn.Module):

	def __init__(self, num_filters=2048, input_dim=[1000,22], use_feat=False, verbose=0):
		super(Model_PCN, self).__init__()

		self.filter_sizes = [1]
		self.num_filters = num_filters
		self.input_dim = input_dim
		self.use_feat = use_feat
		self.verbose = verbose

		self.encoder_param_w = nn.ParameterList()
		self.encoder_param_b = nn.ParameterList()
		for filter_size in self.filter_sizes:
			w = torch.tensor(truncnorm.rvs(-1, 1, size=[self.num_filters,1,filter_size,self.input_dim[1]])).float()
			b = (torch.zeros([self.num_filters]) + 0.1).float()
			self.encoder_param_w.append(nn.Parameter(w))
			self.encoder_param_b.append(nn.Parameter(b))

		self.conv_block2 = nn.Sequential(
			nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_filters),
		)

		self.dense_block1 = nn.Sequential(
			nn.Linear(num_filters, 512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
		)

		self.dense_block2 = nn.Sequential(
			nn.Linear(512, 64),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(64),
		)

		self.dense_block3 = nn.Sequential(
			nn.Linear(64, 16),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(16),
		)

		if self.use_feat:
			self.dense_block3_2 = nn.Sequential(
                		nn.Linear(4, 2),
            		)
			self.dense_block4 = nn.Sequential(
				nn.Linear(18, 1),
			)
		else:
			self.dense_block4 = nn.Sequential(
				nn.Linear(16, 1),
			)
		
		self.sigmoid = nn.Sigmoid()
		

	def forward(self, x, xf):
		# input shape
		if self.verbose != 0:
			print(x.shape)
		
		pooled_outputs = []
		for filter_size, w, b in zip(self.filter_sizes, self.encoder_param_w, self.encoder_param_b):
			c = func.conv2d(x, w, bias=b, stride=1, padding=0)
			h = torch.relu(c)
			pooled = func.max_pool2d(h, kernel_size=[self.input_dim[0]-filter_size+1,1], stride=1, padding=0)
			if self.verbose != 0:
				print(pooled.shape)
			pooled_outputs.append(pooled)
		
		# concatenate all kernel outputs (after max-pooling)
		pooled_all = torch.cat(pooled_outputs, 3)
		if self.verbose != 0:
			print(pooled_all.shape)
			
		pooled_all2 = self.conv_block2(pooled_all)
		if self.verbose != 0:
			print(pooled_all2.shape)

		# flatten
		flattened = torch.reshape(pooled_all2, [-1, self.num_filters])
		if self.verbose != 0:
			print(flattened.shape)

		# 4 fully connected layers
		fc1 = self.dense_block1(flattened)
		if self.verbose != 0:
			print(fc1.shape)

		fc2 = self.dense_block2(fc1)
		if self.verbose != 0:
			print(fc2.shape)

		fc3 = self.dense_block3(fc2)
		if self.verbose != 0:
			print(fc3.shape)

		if self.use_feat:
			fc3_2 = self.dense_block3_2(xf)
			if self.verbose != 0:
				print(fc3_2.shape)
			fc3 = torch.cat((fc3,fc3_2),1)
			if self.verbose != 0:
				print(fc3.shape)
		
		fc4 = self.dense_block4(fc3)
		if self.verbose != 0:
			print(fc4.shape)

		return self.sigmoid(fc4), fc4
