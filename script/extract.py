import os

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datetime
import gc
from tqdm import tqdm
import pandas as pd
import re
import string

from config import (
    PROTTRANS_PATH,
    MODELS_PATH,
    MAX_REPR_PATH,
    MIN_REPR_PATH,
    OUTPUT_PATH,
    DATASETS_PATH,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
)

print(f"Device: {DEVICE}")

Max_repr = np.load(MAX_REPR_PATH)
Min_repr = np.load(MIN_REPR_PATH)


def feature_extraction(ID_list, seq_list, outpath, feat_bs, save_feat, device):
    protein_features = {}
    if save_feat:
        feat_path = outpath / "ProtTrans_repr"
        os.makedirs(feat_path, exist_ok=True)

    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(PROTTRANS_PATH, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTTRANS_PATH)
    gc.collect()

    # Load the model into CPU/GPU and switch to inference mode
    model = model.to(device)
    model = model.eval()

    # Extract feature of one batch each time
    for i in tqdm(range(0, len(ID_list), feat_bs)):
        if i + feat_bs <= len(ID_list):
            batch_ID_list = ID_list[i : i + feat_bs]
            batch_seq_list = seq_list[i : i + feat_bs]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]

        # Load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [
            re.sub(r"[UZOB]", "X", " ".join(list(sequence)))
            for sequence in batch_seq_list
        ]

        # Tokenize, encode sequences and load it into GPU if avilabile
        ids = tokenizer.batch_encode_plus(
            batch_seq_list, add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        # Extract sequence features and load it into CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][: seq_len - 1]
            if save_feat:
                np.save(feat_path / batch_ID_list[seq_num], seq_emd)
            # Normalization
            seq_emd = (seq_emd - Min_repr) / (Max_repr - Min_repr)
            protein_features[batch_ID_list[seq_num]] = seq_emd

    return protein_features


# Load data
def process_fasta(fasta_file):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            name_item = line[1:-1].split("|")
            ID = "_".join(name_item[0 : min(2, len(name_item))])
            ID = re.sub(" ", "_", ID)
            ID_list.append(ID)
        elif line[0] in string.ascii_letters:
            seq_list.append(line.strip().upper())
    return ID_list, seq_list


def load_dataloader(input):
    print("Loading data from:", input)
    result = process_fasta(input)

    if isinstance(result, int):
        if result == -1:
            print("Mismatch between IDs and sequences.")
        elif result == 1:
            print("Number of sequences exceeds MAX_INPUT_SEQ.")
    else:
        ID_list, seq_list = result
        print("success.")
        # Continue processing or using ID_list and seq_list as needed
    feat_bs = 32  # batch size
    save_feat = True  # set to True if you want to save embeddings as .npy files

    print(
        "\n######## Feature extraction begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    protein_embeddings = feature_extraction(
        ID_list, seq_list, OUTPUT_PATH, feat_bs, save_feat, DEVICE
    )
    print(
        "\n######## Feature extraction is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )


ca_dir = DATASETS_PATH / "CA_Train_1554.fa"
mg_dir = DATASETS_PATH / "MG_Train_1730.fa"
mn_dir = DATASETS_PATH / "MN_Train_547.fa"
zn_dir = DATASETS_PATH / "ZN_Train_1647.fa"
catest_dir = DATASETS_PATH / "CA_Test_183.fa"
mgtest_dir = DATASETS_PATH / "MG_Test_235.fa"
mntest_dir = DATASETS_PATH / "MN_Test_57.fa"
zntest_dir = DATASETS_PATH / "ZN_Test_211.fa"

main_dir = [
    ca_dir,
    mg_dir,
    mn_dir,
    zn_dir,
    catest_dir,
    mgtest_dir,
    mntest_dir,
    zntest_dir,
]

for index, i in enumerate(
    ["CA", "MG", "MN", "ZN", "CAtest", "MGtest", "MNtest", "ZNtest"]
):
    print("Training the following metals:", i)
    dataloader = load_dataloader(main_dir[index])
