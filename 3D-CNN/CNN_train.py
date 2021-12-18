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
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
from CNN_DatasetReader import Dataset_to_3d
from CNN_model import Model_CNN, strip_prefix_if_present


def valid_file(a_path):
    return os.path.isfile(a_path) and os.path.getsize(a_path) > 0

# In[7]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(use_cuda, device)


# In[8]:


parser = argparse.ArgumentParser()
parser.add_argument("--data-hdf", default="", help="dataset docking hdf file")
parser.add_argument("--data-chdf", default="", help="dataset crystal hdf file")
parser.add_argument("--data-csv", default="", help="dataset csv file (docking + crystal)")
parser.add_argument("--model-path", default="", help="model checkpoint file path")
parser.add_argument("--model-type", default=6, help="3: 3D-CNN, 5: 3D-CNN_i, 6: 3D-CNN_a 7: 3D-CNN_ia")
parser.add_argument("--epoch-count", default=100, help="number of training epochs")
parser.add_argument("--batch-size", default=50, help="mini-batch size")
parser.add_argument("--learning-rate", default=0.0002, help="initial learning rate")
parser.add_argument("--checkpoint-iter", default=100, help="checkpoint save rate")
parser.add_argument("--number-gpu",default = 1, help = "number of gpu")
args = parser.parse_args()


# In[10]:

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
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf, affine_trans = True )
elif int(args.model_type) == 7:
    print("check -modeltype7!!")
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, args.data_chdf, affine_trans = True, i_feature=True )
else:
    dataset = Dataset_to_3d(args.data_hdf, args.data_csv, i_feature=True )


# In[6]:


dataloader = DataLoader(dataset, batch_size = int(args.batch_size), shuffle=True, num_workers=0, worker_init_fn=None)
batch_count = len(dataset) // int(args.batch_size)


# In[7]:

if int(args.model_type) == 1 or int(args.model_type) == 3 or int(args.model_type) == 6 :
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

loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=float(args.learning_rate))


# In[8]:
if isinstance(model, (DistributedDataParallel, DataParallel)):
    model_to_save = model.module
else:
    model_to_save = model

epoch_start = 0
if valid_file(args.model_path):
    checkpoint = torch.load(args.model_path)
    #model.load_state_dict(checkpoint["model_state_dict"])
    model_state_dict = checkpoint.pop("model_state_dict")
    strip_prefix_if_present(model_state_dict,"module.")
    model_to_save.load_state_dict(model_state_dict,strict = False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_start = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("checkpoint loaded: %s" % args.model_path)

if not os.path.exists(os.path.dirname(args.model_path)):
    os.makedirs(os.path.dirname(args.model_path))

# In[ ]:


model.train()
step = 0
for epoch_ind in range(epoch_start, int(args.epoch_count)):
    losses = []
    for batch_ind, batch in enumerate(dataloader):
        #print(batch[0].shape)
        #print(batch[1].shape)

        optimizer.zero_grad()

        # transfer to GPU
        if int(args.model_type) == 1 or int(args.model_type) == 3 or int(args.model_type) == 6:
            batch_input, batch_label = batch
            print(batch_label.shape)
            batch_input_gpu, batch_label_gpu = batch_input.to(device), batch_label.to(device)
            batch_pred = model(batch_input_gpu, None, None)
            print(batch_pred.shape)
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
        loss = loss_fn(batch_pred.cpu().float(),batch_label.float())
        losses.append(loss.cpu().data.item())
        loss.backward()

        print("[%d/%d-%d/%d] training, loss: %.5f" % (epoch_ind+1, int(args.epoch_count), batch_ind+1, batch_count, loss.cpu().data.item()))
        if step == 0 or step % int(args.checkpoint_iter) == 0:
            checkpoint_dict = {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "step": step,
                "epoch": epoch_ind
            }
            torch.save(checkpoint_dict, args.model_path)
            print("checkpoint saved: %s" % args.model_path)

        optimizer.step()
        step += 1
        #exit()
    loss_avg = np.mean(np.asarray(losses))
    print("[%d/%d] training epoch loss: %.5f" % (epoch_ind+1, int(args.epoch_count), loss_avg))
    




