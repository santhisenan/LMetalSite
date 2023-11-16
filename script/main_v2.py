import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from model import LMetalSite
import numpy as np
from utils import *
import datetime
import gc
from tqdm import tqdm
import pandas as pd

# Configuration
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

############ Set to your own path! ############
ProtTrans_path = "/data/s230112/LMetalSite-main/script/prot_t5_xl_uniref50"

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/model/"

Max_repr = np.load(script_path + "ProtTrans_repr_max.npy")
Min_repr = np.load(script_path + "ProtTrans_repr_min.npy")

device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # use GPU if available, else use CPU
print(device)


def feature_extraction(ID_list, seq_list, outpath, feat_bs, save_feat, device):
    protein_features = {}
    if save_feat:
        feat_path = outpath + "ProtTrans_repr"
        os.makedirs(feat_path, exist_ok=True)

    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
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
                np.save(feat_path + "/" + batch_ID_list[seq_num], seq_emd)
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


def load_dataloader(result):
    result = process_fasta("/data/s230112/LMetalSite-main/datasets/CA_Train_1554.fa")

    if isinstance(result, int):
        if result == -1:
            print("Mismatch between IDs and sequences.")
        elif result == 1:
            print("Number of sequences exceeds MAX_INPUT_SEQ.")
    else:
        ID_list, seq_list = result
        print("success.")
        # Continue processing or using ID_list and seq_list as needed

    outpath = "/data/s230112/LMetalSite-main/script"  # output directory
    feat_bs = 32  # batch size
    save_feat = True  # set to True if you want to save embeddings as .npy files

    print(
        "\n######## Feature extraction begins at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )
    protein_embeddings = feature_extraction(
        ID_list, seq_list, outpath, feat_bs, save_feat, device
    )
    print(
        "\n######## Feature extraction is done at {}. ########\n".format(
            datetime.datetime.now().strftime("%m-%d %H:%M")
        )
    )

    train_dict = {ID_col: ID_list, sequence_col: seq_list}
    train_df = pd.DataFrame(train_dict)

    for metal in metal_list:
        train_df[metal + "_prob"] = 0.0
        train_df[metal + "_pred"] = 0.0

    train_dataset = MetalDataset(train_df, protein_embeddings)
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return dataloader


ca_dir = "/data/s230112/LMetalSite-main/datasets/CA_Train_1554.fa"
mg_dir = "/data/s230112/LMetalSite-main/datasets/MG_Train_1730.fa"
mn_dir = "/data/s230112/LMetalSite-main/datasets/MN_Train_57.fa"
zn_dir = "/data/s230112/LMetalSite-main/datasets/ZN_Train_1647.fa"

main_dir = [ca_dir, mg_dir, mn_dir, zn_dir]

# Load LMetalSite Model
model = LMetalSite(
    feature_dim=NN_config["feature_dim"]
)  # replace YOUR_FEATURE_DIM with the correct dimension
model = model.to(DEVICE)
model.train()

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # or your choice of loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# loop for different metals
for index, i in enumerate(["CA", "MG", "MN", "ZN"]):
    print("Training the following metals:", i)
    dataloader = load_dataloader(main_dir[index])
    # Training loop
    for epoch in range(EPOCHS):
        for batch_data in tqdm(dataloader):
            protein_feats, protein_masks, maxlen = batch_data
            protein_feats = protein_feats.to(device)
            protein_masks = protein_masks.to(device)

            # print("Shapes:",protein_feats.shape,protein_masks.shape)
            mask_shape = list(protein_masks.shape)
            # print("mask shape",mask_shape)
            logits = model(protein_feats, protein_masks)
            # Compute the loss. You need to prepare the correct targets for your sequences.

            # reference point (logits_ZN, logits_CA, logits_MG, logits_MN)
            if i == "ZN":
                logit_output = [1.0, 0, 0, 0]
            elif i == "CA":
                logit_output = [0, 1.0, 0, 0]
            elif i == "MG":
                logit_output = [0, 0, 1.0, 0]
            elif i == "MN":
                logit_output = [0, 0, 0, 1.0]

            targets = torch.tensor(
                [logit_output * mask_shape[1] for _ in range(mask_shape[0])]
            )
            targets = targets.to(device)
            # print("Target size:",targets.shape)
            loss = criterion(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

# Save the trained model if needed
torch.save(model.state_dict(), "LMetalSite.ckpt")
exit()
