# run on gpu2
import pickle
import numpy as np

raw_protrans_path = "./raw_emb/"
protrans_output_path = "./emb/"

Max_protrans = []
Min_protrans = []

all_protrans = {}


########## Train ##########

with open("../../Dataset/Metal_train.pkl", "rb") as f:
    metal_train = pickle.load(f)


print("Processing training dataset")
for ID in metal_train:
    print(f"Processing for {ID=}")
    raw_protrans = np.load(raw_protrans_path + ID + ".npy")
    Max_protrans.append(np.max(raw_protrans, axis=0))
    Min_protrans.append(np.min(raw_protrans, axis=0))
    all_protrans[ID] = raw_protrans


########## Test ##########
with open("../../Dataset/Metal_test.pkl", "rb") as f:
    metal_test = pickle.load(f)

print("Processing testing dataset")
for ID in metal_test:
    print(f"Processing for {ID=}")
    raw_protrans = np.load(raw_protrans_path + ID + ".npy")
    all_protrans[ID] = raw_protrans


########## Normalize feature ##########
Max_protrans = np.max(np.array(Max_protrans), axis=0)
Min_protrans = np.min(np.array(Min_protrans), axis=0)

np.save("Max_ProtTrans_repr", Max_protrans)
np.save("Min_ProtTrans_repr", Min_protrans)

print("Normalising features..")
for ID in all_protrans:
    print(f"Processing for {ID=}")
    protrans = (all_protrans[ID] - Min_protrans) / (Max_protrans - Min_protrans)
    np.save(protrans_output_path + ID, protrans)
