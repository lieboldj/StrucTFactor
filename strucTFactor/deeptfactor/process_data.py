import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import re
import pandas as pd


# read in the protein sequences from a fasta
def read_fasta_data(fasta_file, len_criteria=1000):
    result = []
    seq_ids = []
    labels = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_id = seq_record.id
        tmp_str = re.findall("TF=[0-1]", seq_record.description)
        # implementation of length requirement and padding
        if len(tmp_str) != 0:
            label = int(tmp_str[-1].split("=")[1])     
        else: 
            label = 0
        if len(seq) <= len_criteria:
            seq += '_' * (len_criteria-len(seq))
            result.append(str(seq))
            seq_ids.append(seq_id)
            labels.append(label)
    fp.close()
    return np.asarray(result), np.asarray(seq_ids), np.asarray(labels)

# read in the protein sequences from a fasta and the matching spatial strings 
def read_fasta_data_spatial(fasta_file, spatial_file, len_criteria=1000):
    result = []
    seq_ids = []
    spatial_seqs = []
    labels = []
    spatial_data = pd.read_csv(spatial_file, sep='\t', index_col=0, header=0)
    #print(spatial_data.index.values)
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_id = seq_record.id
        tmp_str = re.findall("TF=[0-1]", seq_record.description)
        # implementation of length requirement and padding
        if len(tmp_str) != 0:
            label = int(tmp_str[-1].split("=")[1])
        else:
            label = 0   
        spatial_str = spatial_data.loc[seq_id].values[0]
        if len(seq) <= len_criteria:
            seq += '_' * (len_criteria-len(seq))
            result.append(str(seq))
            seq_ids.append(seq_id)
            spatial_seqs.append(spatial_str)
            labels.append(label)
    fp.close()
    return np.asarray(result), np.asarray(seq_ids), np.asarray(spatial_seqs), np.asarray(labels) 


def split_data(x, y, SEED=42):
    # split data into training, validation and test set in a stratified manner
    x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x, y, test_size=.2, random_state=SEED, stratify= y)
    x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=.5, random_state=SEED, stratify= y_val_and_test)
    return x_train, y_train, x_val, y_val, x_test, y_test