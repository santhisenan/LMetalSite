import os
import pickle
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from time import time
import datetime, random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

#from LSTM import *
#from GraphSite import *
from model import *
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./Dataset/')
parser.add_argument("--feature_path", type=str, default='./Feature/')
parser.add_argument("--task", type=str, default='Metal') # ZN CA MG MN Metal
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_id", type=str, default=None)

args = parser.parse_args()

seed = args.seed
root = args.feature_path + 'input_protrans/'
Dataset_Path = args.dataset_path
Feature_Path = args.feature_path
run_id = args.run_id
task = args.task

Seed_everything(seed=seed)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if task == "Metal":
    model_class = MetalSite # GraphSite # LSTM
else:
    model_class = MetalSingle


train_df = pd.read_csv(Dataset_Path + task +  '_train.csv')
test_df = pd.read_csv(Dataset_Path + task + '_test.csv')


if args.train:
    ID_list = list(set(train_df['ID']) | set(test_df['ID']))
elif args.test:
    ID_list = list(set(test_df['ID']))


all_protein_data = {}
for pdb_id in ID_list:
    if task == "Metal":
        all_protein_data[pdb_id]=torch.load(root+f"{pdb_id}_node_feature.tensor"),torch.load(root+f"{pdb_id}_mask.tensor"),torch.load(root+f"{pdb_id}_label_mask.tensor"),torch.load(root+f"{pdb_id}_label.tensor")
    else:
        all_protein_data[pdb_id]=torch.load(root+f"{pdb_id}_node_feature.tensor"),torch.load(root+f"{pdb_id}_mask.tensor"),torch.load(root+f"{pdb_id}_label_mask.tensor"),torch.load(root+task+"_label/"+f"{pdb_id}_label.tensor")


train_size = {"ZN":1647, "CA":1554, "MG":1730, "MN":547, "Metal":5337}
num_samples = train_size[task] * 5 # 1个epoch等于5个

nn_config = {
    'feature_dim': 1024, # ProtTrans
    'hidden_dim': 64,
    'num_encoder_layers': 2,
    'num_heads': 4,
    'augment_eps': 0.05,
    'dropout': 0.2,
    'lr': 3e-4,
    'id_name':'ID', # column name in dataframe
    'obj_max': 1,   # optimization object: max is better
    'epochs': 30,
    'patience': 6,
    'batch_size': 32,
    'num_samples': num_samples,
    'folds': 5,
    'seed': seed,
    'remark': task + ' binding site prediction'
}


if args.train:
    NN_train_and_predict(train_df, test_df, all_protein_data, model_class, nn_config, logit = True, run_id = run_id, args=args)
elif args.test:
    NN_train_and_predict(None, test_df, all_protein_data, model_class, nn_config, logit = True, run_id = run_id, args=args)
