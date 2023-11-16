import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

Dataset_Path = "./Dataset/"
Feature_Path = "./Feature/"

NODE_DIM = 1024
max_len = 1000  # within train & tests
metal_list = ["ZN", "CA", "MG", "MN"]


def prepare_features(pdb_id, dataset):
    node_features = np.load(Feature_Path + f"ProtTrans/emb/{pdb_id}.npy")

    # Padding
    padded_node_features = np.zeros((max_len, NODE_DIM))
    padded_node_features[: node_features.shape[0]] = node_features
    padded_node_features = torch.tensor(padded_node_features, dtype=torch.float)

    masks = np.zeros(max_len)
    masks[: node_features.shape[0]] = 1
    masks = torch.tensor(masks, dtype=torch.long)

    labels = dataset[pdb_id][1]
    padded_labels = []  # labels for multi-task learning
    label_masks = []  # masks for multi-task learning
    for i in range(len(metal_list)):
        padded_y = np.zeros(max_len)
        label_mask = np.zeros(max_len)
        y = labels[i]
        if y:
            padded_y[: node_features.shape[0]] = y
            # labels for single-task learning
            torch.save(
                torch.tensor(padded_y, dtype=torch.float),
                Feature_Path
                + f"input_protrans/{metal_list[i]}_label/{pdb_id}_label.tensor",
            )

            label_mask[: node_features.shape[0]] = 1

        padded_labels += list(padded_y)
        label_masks += list(label_mask)
    padded_labels = torch.tensor(padded_labels, dtype=torch.float)
    label_masks = torch.tensor(label_masks, dtype=torch.float)

    # Save
    torch.save(
        padded_node_features,
        Feature_Path + f"input_protrans/{pdb_id}_node_feature.tensor",
    )
    torch.save(padded_labels, Feature_Path + f"input_protrans/{pdb_id}_label.tensor")
    torch.save(masks, Feature_Path + f"input_protrans/{pdb_id}_mask.tensor")
    torch.save(label_masks, Feature_Path + f"input_protrans/{pdb_id}_label_mask.tensor")


def pickle2csv(metal_name):
    with open(Dataset_Path + metal_name + "_train.pkl", "rb") as f:
        train = pickle.load(f)

    train_IDs, train_sequences, train_labels = [], [], []
    for ID in train:
        train_IDs.append(ID)
        item = train[ID]
        train_sequences.append(item[0])
        train_labels.append(item[1])

    train_dic = {"ID": train_IDs, "sequence": train_sequences, "label": train_labels}
    train_dataframe = pd.DataFrame(train_dic)
    train_dataframe.to_csv(Dataset_Path + metal_name + "_train.csv", index=False)

    with open(Dataset_Path + metal_name + "_test.pkl", "rb") as f:
        test = pickle.load(f)

    test_IDs, test_sequences, test_labels = [], [], []
    for ID in test:
        test_IDs.append(ID)
        item = test[ID]
        test_sequences.append(item[0])
        test_labels.append(item[1])

    test_dic = {"ID": test_IDs, "sequence": test_sequences, "label": test_labels}
    test_dataframe = pd.DataFrame(test_dic)
    test_dataframe.to_csv(Dataset_Path + metal_name + "_test.csv", index=False)


if __name__ == "__main__":
    for metal_name in metal_list:
        pickle2csv(metal_name)
    pickle2csv("Metal")

    with open(Dataset_Path + "Metal_train.pkl", "rb") as f:
        metal_train = pickle.load(f)
    for ID in tqdm(metal_train):
        prepare_features(ID, metal_train)

    with open(Dataset_Path + "Metal_test.pkl", "rb") as f:
        metal_test = pickle.load(f)
    for ID in tqdm(metal_test):
        prepare_features(ID, metal_test)
