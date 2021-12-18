################################################################################
# Copyright 2021 Heesung Shim
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
################################################################################

import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import *
from scipy.stats import *

from PCN_model_pcn import Model_PCN, Model_Transform, strip_prefix_if_present
from PCN_data_reader import Dataset_MLHDF

def valid_file(a_path):
    return os.path.isfile(a_path) and os.path.getsize(a_path) > 0

# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cpu", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="../../Data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="pdbbind2019_core_docking.hdf", help="training docking ml-hdf path")
parser.add_argument("--cmlhdf-fn", default="pdbbind2019_core_crystal.hdf", help="training crystal ml-hdf path")
parser.add_argument("--csv-fn", default="pdbbind2019_core.csv", help="training csv file path")
parser.add_argument("--model-path", default="../../Model_Checkpoint/PCN_a.pth", help="model checkpoint file path")
parser.add_argument("--use-feat", default=False, action="store_true", help="use ligand and interaction feature or not")
parser.add_argument("--max-atoms", type=int, default=2000, help="maximum number of atoms")
parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")
args = parser.parse_args()


# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
if use_cuda:
	device = torch.device(args.device_name)
	torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)




def eval():

	# load dataset
	csv_path = os.path.join(args.data_dir, args.csv_fn)
	mlhdf_path = os.path.join(args.data_dir, args.mlhdf_fn)
	cmlhdf_path = os.path.join(args.data_dir, args.cmlhdf_fn)
	dataset = Dataset_MLHDF(csv_path, mlhdf_path, cmlhdf_path, feat_int=args.use_feat, affine_trans=False, max_atoms=args.max_atoms)

	# check multi-gpus
	num_workers = 0
	if args.multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_count = len(dataset) // args.batch_size
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)

	# define model
	model = Model_PCN(input_dim=[args.max_atoms,22], use_feat=args.use_feat, verbose=args.verbose)
	model_t = Model_Transform(verbose=args.verbose)
	#if use_cuda:
	#	model = model.cuda()
	if args.multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# load model
	if not valid_file(args.model_path):
		print("checkpoint not found! %s" % args.model_path)
		return
	checkpoint = torch.load(args.model_path, map_location=device)
	#checkpoint = torch.load(args.model_path)
	model_state_dict = checkpoint.pop("model_state_dict")
	strip_prefix_if_present(model_state_dict, "module.")
	epoch = checkpoint["epoch"]
	model_to_save.load_state_dict(model_state_dict, strict=False)
	output_dir = os.path.dirname(args.model_path)
	print("checkpoint loaded: %s (epoch: %d)" % (args.model_path, epoch))

	ytrue_arr = np.zeros((len(dataset),), dtype=np.float32)
	ypred_arr = np.zeros((len(dataset),), dtype=np.float32)

	model.eval()
	with torch.no_grad():
		for bind, batch in enumerate(dataloader):

			# transfer to GPU
			x_cpu, xf_cpu, y_cpu = batch
			x = x_cpu.to(device)
			y = y_cpu.to(device)
			xf = xf_cpu.to(device)
			
			# forward training
			yp, _ = model(x, xf)

			ytrue = y_cpu.float().data.numpy()[:,0]
			ypred = yp.cpu().float().data.numpy()[:,0]
			if bind == batch_count:
				ytrue_arr[bind*args.batch_size:len(dataset)] = ytrue
				ypred_arr[bind*args.batch_size:len(dataset)] = ypred
			else:
				ytrue_arr[bind*args.batch_size:(bind+1)*args.batch_size] = ytrue
				ypred_arr[bind*args.batch_size:(bind+1)*args.batch_size] = ypred

			print("[%d/%d] evaluating" % (bind+1, batch_count))

	ypred_arr2 = np.where(ypred_arr > 0.5, 1, 0)
	acc = 100.0 * np.sum(ypred_arr2 == ytrue_arr) / ytrue_arr.shape[0]
	eval = classification_report(ytrue_arr, ypred_arr2)
	print("Evaluation Summary:")
	print("accuracy: %.1f%%" % acc)
	print("prec/recall/f1:")
	print(eval)

	np.save(args.model_path[:-4] + "_" + args.mlhdf_fn + "_eval_label.npy", ytrue_arr)
	np.save(args.model_path[:-4] + "_" + args.mlhdf_fn + "_eval_pred.npy", ypred_arr)
	np.save(args.model_path[:-4] + "_" + args.mlhdf_fn + "_eval_pred2.npy", ypred_arr2)

def main():
	eval()


if __name__ == "__main__":
	main()


