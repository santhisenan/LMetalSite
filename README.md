# Setting up environment

1. Run `conda env create -f env.yml`
2. Run `conda activate lmetalsite`

# Generating features and training

1. Run extract.py (placed in script folder of the author's github set of code).

This will produce a folder called scriptProtTrans_repr. This folder contains all the features per protein in .npy format.

2. Move the contents of the script/ProtTrans_repr folder (just the contents, dont move the whole folder) to Feature/ProtTrans/raw_emb (from here onwards will just be using the codes author gave to Shen Cheng)

3. Create a folder named emb in Feature/ProtTrans/

4. Run process_ProtTrans.py and the output of the processed data will be in Feature/ProtTrans/emb. The code also saves the Max_ProtTrans_repr.npy and Min_ProtTrans_repr.npy that was used for normalization.

5. create four empty folders in Feature/input_potrans called CA_label, MG_label, MN_label and ZN_label.

6. Run pad_feature.py which will output the necessary tensors required for training into input_protrans folder.

7. Now can start training using this command line:

   `./run.sh main.py [gpu number] [name of folder to save the training stuff]`

8. Run test: go to run.sh can comment out the train and uncomment the test. then run this same command line

   `./run.sh main.py [gpu number] [name of THE SAME folder of your training stuff]`