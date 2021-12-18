################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################


import os
import sys
sys.stdout.flush()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# to read checkpoint files from any GPU configuration
# from https://discuss.pytorch.org/t/checkpoint-in-multi-gpu/97852
def strip_prefix_if_present(state_dict, prefix):
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
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


class Model_CNN(nn.Module):
    def __init__(self, num_feat=19, num_ifeat=0, num_lfeat=0, verbose=0):
        super(Model_CNN, self).__init__()
        self.verbose = verbose
        self.num_ifeat = num_ifeat
        self.num_lfeat = num_lfeat
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(num_feat, 32, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128),
        )
        self.fc_block1 = nn.Sequential(
            nn.Linear(1024,200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(200,20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
        )
        if self.num_ifeat > 0 and self.num_lfeat > 0:
            self.fc_block3_2 = nn.Sequential(
                nn.Linear(num_ifeat,2),
            )
            self.fc_block3_3 = nn.Sequential(
                nn.Linear(num_lfeat,10),
            )
            self.fc_block3 = nn.Sequential(
                nn.Linear(32,1),
            )
        else:
            self.fc_block3 = nn.Sequential(
                nn.Linear(20,1),
            )
        
    def forward(self, x, x_ifeat, x_lfeat):
        c1 = self.conv_block1(x)
        if self.verbose == 1:
            print('conv1', c1.shape)
        p1 = F.max_pool3d(c1, 2)
        if self.verbose == 1:
            print('max_pool3d', p1.shape)
        c2 = self.conv_block2(p1)
        if self.verbose == 1:
            print('conv2', c2.shape)
        p2 = F.max_pool3d(c2, 2)
        if self.verbose == 1:
            print('max_pool3d', p2.shape)
        c3 = self.conv_block3(p2)
        if self.verbose == 1:
            print('conv3', c3.shape)
        p3 = F.max_pool3d(c3, 2)
        if self.verbose == 1:
            print('max_pool3d', p3.shape)
        v = p3.view(-1, 1024)
        if self.verbose == 1:
            print(v.shape)
        fc1= self.fc_block1(v)
        if self.verbose == 1:
            print('fc1',fc1.shape)
        fc2= self.fc_block2(fc1)
        if self.verbose == 1:
            print('fc2',fc2.shape)
        if self.num_ifeat and self.num_lfeat > 0 :
            fc3_2 = self.fc_block3_2(x_ifeat)
            if self.verbose == 1:
                print('fc3_2',fc3_2.shape)
            fc3_3 = self.fc_block3_3(x_lfeat)
            if self.verbose == 1:
                print('fc3_3',fc3_3.shape)
            cat = torch.cat((fc2,fc3_2,fc3_3),1)
            fc3 = self.fc_block3(cat)
            if self.verbose == 1:
                print('fc3',fc3.shape)
        else:
            fc3= self.fc_block3(fc2)
            if self.verbose == 1:
                print('fc3',fc3.shape)
        return torch.sigmoid(fc3)

