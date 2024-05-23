import os
import random
# import basic python packages
import numpy as np
import pandas as pd


# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deeptfactor.process_data import read_fasta_data
from deeptfactor.process_data import read_fasta_data_spatial
from deeptfactor.process_data import split_data
from deeptfactor.data_loader import EnzymeDataset
from deeptfactor.data_loader import EnzymeDataset_spatial
from deeptfactor.utils import argument_parser
from deeptfactor.models import DeepTFactor
from deeptfactor.test import test_model
from deeptfactor.validation import validate_model
from deeptfactor.train import train_model
from deeptfactor.evaluation import evaluate_model

import time
import copy

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
    training = options.training
    learning_rate = options.learning_rate
    spatial_mode = options.spatial_mode
    es = options.early_stop
    start_time = time.perf_counter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.set_num_threads(num_cpu)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Include the datasets into training validation and testing
    
    if training:
        # seperated into spatial models (altered deepTFactor) and 
        # sequence model (reimplemented original deepTFactor)
        if spatial_file != None:            # spatial mode
            if spatial_mode == 0:           # fix to spatial if sequence (no sm is 0) but spatial file is given
                spatial_mode = 1
            protein_seqs, seq_ids, spatial_seqs, labels = read_fasta_data_spatial(protein_data_file, spatial_file)
            indexes = np.arange(len(protein_seqs))
            # train test split function here
            x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(indexes, labels) # with index or with hole dataset
            
            train_dataset = EnzymeDataset_spatial(protein_seqs[x_train], y_train, spatial_seqs[x_train], spatial_mode)
            vali_dataset = EnzymeDataset_spatial(protein_seqs[x_vali], y_vali, spatial_seqs[x_vali], spatial_mode)
            test_dataset = EnzymeDataset_spatial(protein_seqs[x_test], y_test, spatial_seqs[x_test], spatial_mode)
        else:                          # sequence mode
            if spatial_mode == 1:      # automatic correction if spatial (sm > 0) but no spatial file given
                spatial_mode = 0
            protein_seqs, seq_ids, labels = read_fasta_data(protein_data_file)
            indexes = np.arange(len(protein_seqs))
            

            #proteinDataset = EnzymeDataset(protein_seqs, pseudo_labels)
            # train test split function here
            x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(indexes, labels) # with index or with hole dataset

            train_dataset = EnzymeDataset(protein_seqs[x_train], y_train)
            vali_dataset = EnzymeDataset(protein_seqs[x_vali], y_vali)
            test_dataset = EnzymeDataset(protein_seqs[x_test], y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        model = DeepTFactor(spatial_mode, out_features=[1],)
        model = model.to(device)
        cutoff = 0.5
        last_epoch = 0

        # CNN training loop with early stopping
        epoch_number = 50
        early_stopping = 0
        survailence_list = []
        prev_vali_loss, best_vali_loss = 1000, 1000
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) #low information about parameters except lr = 0.01
        criterion = nn.BCELoss()
        for epoch in range(1, epoch_number +1):
            last_epoch = epoch  #to see which epoch the model stopped at
            print(f"Epoch {epoch} of {epoch_number}:")
            train_loss, train_acc = train_model(model, device, train_loader, optimizer, criterion, cutoff)
            best_vali_loss, vali_loss, vali_acc = validate_model(model, vali_loader, device, best_vali_loss, batch_size, criterion, cutoff)
            survailence_list.append([epoch, train_loss, train_acc, vali_loss, vali_acc])
            if best_vali_loss == prev_vali_loss:  # check for early stopping criteria
                early_stopping += 1
                if early_stopping == es:
                    print("Early Stopping Reached")
                    model.load_state_dict(best_model)
                    break
            else: 
                # load the best performing model
                early_stopping = 0
                prev_vali_loss = best_vali_loss
                best_model = copy.deepcopy(model.state_dict())
        
        torch.save({
            'model': model.state_dict() 
            }, f'{output_dir}/own_model.pt')
        survailence_df = pd.DataFrame(survailence_list, columns = [f'{learning_rate} epoch', 'train_loss', 'vali_loss', 'train_acc', 'vali_acc'])
        survailence_df.to_csv(f'{output_dir}/survailence.tsv', sep='\t', index=False)
        
    else:
        # use tool with selftrained model, no training
        if spatial_file != None:
            if spatial_mode == 0: 
                spatial_mode = 1
            protein_seqs, seq_ids, spatial_seqs, labels = read_fasta_data_spatial(protein_data_file, spatial_file)
            indexes = np.arange(len(protein_seqs))
            # create x_test that has values from 0 to len(protein_seqs)-1 
            x_test = indexes
            y_test = labels
            test_dataset = EnzymeDataset_spatial(protein_seqs[x_test], labels, spatial_seqs[x_test], spatial_mode)
        else:
            if spatial_mode == 1:      # automatic correction if spatial but no spatial file given
                spatial_mode = 0
            protein_seqs, seq_ids, labels = read_fasta_data(protein_data_file)
            indexes = np.arange(len(protein_seqs))
            #pseudo_labels = np.zeros((len(protein_seqs)))
            x_test = indexes
            y_test = labels
            test_dataset = EnzymeDataset(protein_seqs[x_test], labels)

        model = DeepTFactor(spatial_mode, out_features=[1],)
        model = model.to(device)
        cutoff = 0.5
        last_epoch = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        ckpt = torch.load(f'{checkpt_file}', map_location=device)
        model.load_state_dict(ckpt['model'])

    # score and evaluation section
    scores = test_model(model, test_loader, device, x_test)
    end_time = time.perf_counter()
    needed = end_time - start_time

    evaluate_model(seq_ids[x_test], y_test, scores, [needed, last_epoch], output_dir)



