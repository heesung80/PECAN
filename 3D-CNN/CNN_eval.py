################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################

import os
import sys
sys.stdout.flush()
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import *
from scipy.stats import *

from CNN_DatasetReader import Dataset_to_3d
from CNN_model import Model_CNN, strip_prefix_if_present

def valid_file(a_path):
    return os.path.isfile(a_path) and os.path.getsize(a_path) > 0


# In[15]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(use_cuda, device)

# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--data-hdf", default="Data/pdbbind2019_core_docking.hdf", help="dataset directory")
parser.add_argument("--data-csv", default="Data/pdbbind2019_core.csv", help="dataset csv file")
parser.add_argument("--data-chdf", default="Data/pdbbind2019_core_crystal.hdf", help="dataset crystal hdf file")
parser.add_argument("--model-path", default="Model_Checkpoint/CNN_a.pth", help="model checkpoint file path")
parser.add_argument("--model-type", default=3, help="3: 3D-CNN, 5: 3D-CNN_i, 6: 3D-CNN_a 7: 3D-CNN_ia")
parser.add_argument("--batch-size", default=50, help="mini-batch size")
parser.add_argument("--number-gpu",default = 1, help = "number of gpu")
parser.add_argument("--output-prefix", default="", help="")
args = parser.parse_args()


# In[16]:





# In[17]:

if int(args.model_type) == 1:
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv)
elif int(args.model_type) == 3:
    print("check -modeltype3!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv,args.data_chdf )
elif int(args.model_type) == 4:
    print("check -modeltype4!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf, i_feature=True )
elif int(args.model_type) == 5:
    print("check -modeltype5!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf, i_feature=True )
elif int(args.model_type) == 6:
    print("check -modeltype6!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf)
elif int(args.model_type) == 7:
    print("check -modeltype6!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf, i_feature=True)
else:
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, i_feature=True )


# In[18]:


dataloader = DataLoader(dataset, batch_size = int(args.batch_size), shuffle=False, num_workers=0, worker_init_fn=None)
batch_count = len(dataset) // int(args.batch_size)


# In[19]:


if int(args.model_type) == 1 or int(args.model_type) == 3 or int(args.model_type) == 6:
    model = Model_CNN()
elif int(args.model_type) == 5 or int(args.model_type) == 7:
    model = Model_CNN(num_ifeat = 4, num_lfeat= 0)
else:
    model = Model_CNN(num_ifeat = 4, num_lfeat= 200)
if use_cuda:
    if int(args.number_gpu) > 1 or torch.cuda.device_count() > 1:
        print('# GPUs:', torch.cuda.device_count())
        model = nn.DataParallel(model)
    model = model.cuda()


# In[11]:
if isinstance(model, (DistributedDataParallel, DataParallel)):
    model_to_save = model.module
else:
    model_to_save = model

# load model
if not valid_file(args.model_path):
    print("checkpoint not found! %" % args.model_path)
checkpoint = torch.load(args.model_path, map_location=device)
model_state_dict = checkpoint.pop("model_state_dict")
strip_prefix_if_present(model_state_dict,"module.")
model_to_save.load_state_dict(model_state_dict,strict = False)


# In[26]:

batch_size = int(args.batch_size)
model.eval()
with torch.no_grad():
    y_true_arr = np.zeros((len(dataset)), dtype=np.float32)
    y_pred_arr = np.zeros((len(dataset)), dtype=np.float32)
    for batch_ind, batch in enumerate(dataloader):
        #print(batch[0].shape)
        #print(batch[1].shape)

        # transfer to GPU
        if int(args.model_type) == 1 or int(args.model_type) == 3 or int(args.model_type) == 6:
            batch_input, batch_label = batch
            #print(batch_label.shape)
            batch_input_gpu, batch_label_gpu = batch_input.to(device), batch_label.to(device)
            batch_pred = model(batch_input_gpu, None, None)
            #print(batch_pred.shape)
        else:
            #print("confirmed!")
            batch_input, batch_label, batch_input_ifeat, batch_input_lfeat = batch
            #print(batch_label.shape)
            batch_input_gpu = batch_input.to(device)
            batch_label_gpu = batch_label.to(device)
            batch_input_ifeat_gpu = batch_input_ifeat.to(device)
            batch_input_lfeat_gpu = batch_input_lfeat.to(device)
            batch_pred = model(batch_input_gpu,batch_input_ifeat_gpu, batch_input_lfeat_gpu )
            #print(batch_pred.shape)

        # get numpy arrays of y_true and y_label
        y_true = batch_label.cpu().float().data.numpy()
        y_pred = batch_pred.cpu().float().data.numpy()
        
        if batch_ind == batch_count:
            y_true_arr[batch_size*batch_ind : len(dataset)] = y_true[:,0]
            y_pred_arr[batch_size*batch_ind : len(dataset)] = y_pred[:,0]
        else:
            #print(y_true[:].shape)
            #print(y_true_arr[batch_size*batch_ind : batch_size*batch_ind + batch_size].shape)
            y_true_arr[batch_size*batch_ind : batch_size*batch_ind + batch_size] = y_true[:,0]
            y_pred_arr[batch_size*batch_ind : batch_size*batch_ind + batch_size] = y_pred[:,0]
        #exit()
    out_dir = os.path.dirname(args.model_path)
    np.save(os.path.join(out_dir, args.output_prefix + "_eval_label.npy"), y_true_arr)
    np.save(os.path.join(out_dir, args.output_prefix +"_eval_pred.npy"), y_pred_arr)
            
    y_pred_arr_2 = np.where(y_pred_arr > 0.5, 1, 0)
    acc = 100.0 * np.sum(y_pred_arr_2 == y_true_arr) / y_true_arr.shape[0]
    eval = classification_report(y_true_arr, y_pred_arr_2)
    print("accuracy: %.1f%%" % acc)
    print("prec/recall/f1:")
    print(eval)
    np.save(os.path.join(out_dir, args.output_prefix +"_eval_pred_2.npy"), y_pred_arr_2)
    
