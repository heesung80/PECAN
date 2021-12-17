# Pose classification using 3D atomic structure-based neural networks for ion channel-ligand docking
Identifying the correct binding pose of ligand is important in docking based virtual high-throughput screening. This code implements two convolutional neural network approaches: a 3D convolutional neural network (3D-CNN) and an attention-based point cloud network (PCN) to improves virtual high-throughput screening to identify novel molecules against each target protein. The code is written in python with Tensorflow and Pytorch.
## Prerequsite
- [PyTorch](https://pytorch.org)
- [Open Drug Discovery Tool Kit (ODDT)](https://oddt.readthedocs.io/en/latest/)
- [Open Babel](https://openbabel.org/docs/dev/Installation/install.html)
- [RDkit](https://www.rdkit.org)

## Running the application

### Data Format
Both 3D-CNN and PCN use a 3D atomic representation as input data in a Hierarchical Data Format (HDF5). 

### 3D-CNN
To train, ```3D-CNN/CNN_train.py``` To test/evaluate, ```run 3D-CNN/CNN_eval.py```  Here is an example comand to evaluate a pre-trained 3D-CNN model:
```
python CNN_eval.py --data-hdf pdbbind2019_core_docking.hdf  --data-csv pdbbind2019_core.csv  --data-chdf pdbbind2019_core_crystal.hdf  --model-path /Model_Checkpoint/3D-CNN/CNN_a.pth --model-type 6 
```
### PCN
To train, ```3D-CNN/PCN_main_train.py``` To test/evaluate, run ```3D-CNN/PCN_main_eval.py```. Here is an example comand to evaluate a pre-trained PCN model:
```
python PCN_main_eval.py --device-name cuda:1  --data-dir /Data  --dataset-type 1  --mlhdf-fn pdbbind2019_core_docking.hdf  --csv-fn pdbbind2019_core.csv  --cmlhdf-fn pdbbind2019_core_crystal.hdf  --model-path /Model_Checkpoint/PCN/PCN_a.pth
```
### Pre-trained weights
We trained all of the networks above on pdbbind 2019 datasets. Particularly, we used refined datasets for training, and evaluated the model on the core set (see Data/pdbbind2019_core_docking.hdf). (https://drive.google.com/drive/folders/17aTy-5Epvvwcpn_Kxwn7ksyUU2EGed3g?usp=sharing)


## Authors
Pose-Classifier was created by Heesung Shim ( hsshim@ucdavis.edu )
