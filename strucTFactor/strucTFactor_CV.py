import os
import random
# import basic python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from deeptfactor.process_data import read_fasta_data
from deeptfactor.process_data import read_fasta_data_spatial
from deeptfactor.data_loader import EnzymeDataset
from deeptfactor.data_loader import EnzymeDataset_spatial
from deeptfactor.utils import argument_parser
from deeptfactor.models import DeepTFactor
from deeptfactor.test import test_model
from deeptfactor.validation import validate_model
from deeptfactor.train import train_model
from deeptfactor.evaluationcv import evaluate_model
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
    spatial_mode = options.spatial_mode
    training = options.training
    learning_rate = options.learning_rate
    es = options.early_stop
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

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    contbl = np.zeros(7)
    model = DeepTFactor(spatial_mode, out_features=[1])
    model = model.to(device)
    cutoff = 0.5
    # Seperation into folds
    for fold, [train_vali_idx, x_test] in enumerate(skf.split(indexes, labels)):
        print(f"crossvalidationfold: {fold}")
        x_train, x_vali, y_train, y_vali = train_test_split(indexes[train_vali_idx], labels[train_vali_idx], test_size=.1, random_state=42, stratify= labels[train_vali_idx])
        if spatial_file != None:
            train_dataset = EnzymeDataset_spatial(protein_seqs[x_train], labels[x_train], spatial_seqs[x_train], spatial_mode)
            vali_dataset = EnzymeDataset_spatial(protein_seqs[x_vali], labels[x_vali], spatial_seqs[x_vali], spatial_mode)
            test_dataset = EnzymeDataset_spatial(protein_seqs[x_test], labels[x_test], spatial_seqs[x_test], spatial_mode)
        else:
            train_dataset = EnzymeDataset(protein_seqs[x_train], labels[x_train])
            vali_dataset = EnzymeDataset(protein_seqs[x_vali], labels[x_vali])
            test_dataset = EnzymeDataset(protein_seqs[x_test], labels[x_test])
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if fold == 0:
            torch.save(model, f'{output_dir}basestate_model.pt')
            
        else:
            model = torch.load(f'{output_dir}basestate_model.pt')
        

        if training:
            batch_size = batch_size
            epoch_number = 50
            early_stopping = 0
            survailence_list = []
            prev_vali_loss, best_vali_loss = 1000, 1000
            optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
            criterion = nn.BCELoss()
            for epoch in range(1, epoch_number +1):
                print(f"Epoch {epoch} of {epoch_number}:")
                train_loss, train_acc = train_model(model, device, train_loader, optimizer, criterion, cutoff)
                best_vali_loss, vali_loss, vali_acc = validate_model(model, vali_loader, device, best_vali_loss, batch_size, criterion, cutoff)
                survailence_list.append([epoch, train_loss, train_acc, vali_loss, vali_acc])
                if best_vali_loss == prev_vali_loss:  
                    early_stopping += 1
                    if early_stopping == es:
                        print("Early Stopping Reached")
                        model = torch.load(f'{output_dir}tmp_best_model.pt')
                        break
                else: 
                    # load the best performing model
                    early_stopping = 0
                    prev_vali_loss = best_vali_loss
                    torch.save(model, f'{output_dir}tmp_best_model.pt')

                
            torch.save({
                'model': model.state_dict() 
                }, f'{output_dir}/own_model{fold}.pt')
                
            survailence_df = pd.DataFrame(survailence_list, columns = [f'{learning_rate} epoch', 'train_loss', 'vali_loss', 'train_acc', 'vali_acc'])
            survailence_df.to_csv(f'{output_dir}/survailence.tsv', sep='\t', index=False)
        else:
            ckpt = torch.load(f'{checkpt_file}', map_location=device)
            model.load_state_dict(ckpt['model'])


        # score and evaluation section
        scores = test_model(model, test_loader, device, x_test)
        end_time = time.perf_counter()
        needed = end_time - start_time



        contbl = np.add(contbl, evaluate_model(seq_ids[x_test], labels[x_test], scores, needed, output_dir, fold))
    # temporary mean over the folds (seperate aggregated used)
    contbl = contbl / k
    TP = contbl[0]
    TN = contbl[1]
    FP = contbl[2]
    FN = contbl[3]
    roc_auc = contbl[4]
    AUPRC = contbl[5]
    MCC_middle = contbl[6]
    print("All Folds together:")
    print(f'TP: {TP}\tFP: {FP}\n TN: {TN}\tFN: {FN}')
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FP)
    print(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}')


    with open(f'{output_dir}/result_info.txt', 'a') as fp:
        fp.write(f'Fold: all; Time for run: {needed} \n')
        fp.write(f'accuracy: {accuracy:0.4f}, sensitivity: {sensitivity:0.4f}, specificity: {specificity:0.4f}, recall: {recall:0.4f}\n')
        fp.write(f'TP: {TP}\tFP: {FP}\nTN: {TN}\tFN: {FN}\n')
        fp.write(f'ROC curve (area = {roc_auc:0.4f})\n')
        fp.write(f'AUPRC (AP): {AUPRC:0.4f})\n')
        fp.write(f'MCC: {MCC_middle}\n')
        fp.write("\n\n")



