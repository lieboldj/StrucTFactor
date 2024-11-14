import os
import random
# import basic python packages
import numpy as np

# import torch packages
import torch

from sklearn.model_selection import StratifiedKFold

from deeptfactor.process_data import read_fasta_data
from deeptfactor.process_data import read_fasta_data_spatial

from deeptfactor.utils import argument_parser

import time

# run with old scikit-learn version < 0.22
if __name__ == '__main__':


    # parsing options, options are explained in deepTFactor/utils.py
    parser = argument_parser()
    options = parser.parse_args()

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size

    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    protein_data_file = options.seq_file
    spatial_file = options.spatial_file
    spatial_mode = options.spatial_mode
    training = options.training
    learning_rate = options.learning_rate
    es = options.early_stop
    epochs = options.epochs
    start_time = time.perf_counter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.set_num_threads(num_cpu)
    # add deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # seperated into spatial models (altered deepTFactor) and 
    # sequence model (reimplemented original deepTFactor)
    if spatial_file != None:
        if spatial_mode == 0: 
            spatial_mode = 1
        protein_seqs, seq_ids, spatial_seqs, labels = read_fasta_data_spatial(protein_data_file, spatial_file)
        indexes = np.arange(len(protein_seqs))

    else:
        if spatial_mode == 1:      # automatic correction if spatial but no spatial file given
            spatial_mode == 0
        protein_seqs, seq_ids, labels = read_fasta_data(protein_data_file)
        indexes = np.arange(len(protein_seqs))


    #proteinDataset = EnzymeDataset_spatial(protein_seqs, pseudo_labels, spatial_file)
    # train test split function here
    """x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(indexes, labels) # with index or with hole dataset
    x_ cross = np.append(x_train, x_vali)
    y_cross = np.append(y_train, y_vali)"""

    #test_dataset = EnzymeDataset_spatial(protein_seqs[x_test], y_test, spatial_seqs[x_test])
    k = 5
    print(output_dir)
    infos = output_dir.split('/')[-1]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(skf.split(indexes, labels))

    # save the splits in pickle file
    import pickle
    with open(f'{output_dir}/splits_{infos}.pkl', 'wb') as f:
        pickle.dump(splits, f)

    exit()
    