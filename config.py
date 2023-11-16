import os
from pathlib import Path

import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Paths
ROOT_PATH = Path(__file__).resolve().parent
PROTTRANS_PATH = "/scratch/users/ntu/santhise/models/prot_t5_xl_uniref50"
MAX_REPR_PATH = ROOT_PATH / "script/ProtTrans_repr_max.npy"
MIN_REPR_PATH = ROOT_PATH / "script/ProtTrans_repr_min.npy"
OUTPUT_PATH = ROOT_PATH / "script"
MODELS_PATH = ROOT_PATH / "model"
DATASETS_PATH = ROOT_PATH / "datasets"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
