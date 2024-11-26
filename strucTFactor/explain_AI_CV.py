import os
# import basic python packages
import numpy as np
import pandas as pd
# import torch packages
import torch
from torch.utils.data import DataLoader
from deeptfactor.process_data import read_fasta_data
from deeptfactor.process_data import read_fasta_data_spatial
from deeptfactor.data_loader import EnzymeDataset
from deeptfactor.data_loader import EnzymeDataset_spatial
from deeptfactor.utils import argument_parser
from deeptfactor.models import DeepTFactor
import captum.attr as captum_attr
from tqdm import tqdm

def scoring_function(values_list):
    domain_lower = 1 #not needed anymore
    domain_upper = 138
    # count the mean of values_list in domain
    tmp_mean = values_list.mean()
    best_value = 0
    best_position = None
    best_size = None
    for Dsize in range(domain_lower,domain_upper +1):
        for j in range(0, len(values_list)-Dsize):              
            tmp_value = values_list[j:j+Dsize].sum()             
            tmp_value = tmp_value - (Dsize * tmp_mean)          # introduce penalty as mean over all in this run 
            if tmp_value > best_value:
                best_value = tmp_value
                best_position = j
                best_size = i
    return [best_value, best_position, best_size]

#TODO: outside: get min and max of  domain, cluster domain names
#      inside: get domain areas of each cluster and calc sum in each area 

def read_fasta_file(file_path):
    sequences = {}
    with open(file_path, 'r') as fasta_file:
        sequence_id = ''
        sequence = ''
        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id != '' and 'TF=1' in sequence_id:
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                # Check if 'TF=1' is present in the sequence ID
                if 'TF=1' not in sequence_id:
                    sequence_id = ''
                    sequence = ''
                else:
                    sequence = ''
            else:
                sequence += line
        # Add the last sequence if TF=1 was found in the last sequence ID
        if sequence_id != '' and 'TF=1' in sequence_id:
            sequences[sequence_id] = sequence
    return sequences

def write_fasta_file(sequences, output_path):
    with open(output_path, 'w') as fasta_file:
        for sequence_id, sequence in sequences.items():
            fasta_file.write('>' + sequence_id + '\n')
            fasta_file.write(sequence + '\n')
# python tf_running_CV_fast.py -i ../../dataset/data/test_explainable.fasta -s ../../dataset/data/AF90ofAAwith70_spatial_str.tsv -o ../../results/explainable/explainable_test/ -ckpt ../../results/final/final_AF_norm/spatial_clu03esNo/own_model.pt -g cuda:0

if __name__ == '__main__':

    setting = "03_norm_Com"
    datasettype = "AFfiltered"

    
    clu = "clu" + setting.split("_")[0]
    ratio = setting.split("_")[1]
    if True:#not os.path.exists(f"../data/Fold_0_sequences_{setting}_{datasettype}.fasta"):
        # Read the data file to get folds
        with open(f'../results/CV_{datasettype}_expTFs_{clu}_{ratio}_spatial0_Com/prediction_result.tsv', 'r') as data_file:
            data_lines = data_file.readlines()

        fold_start_indices = [i for i, line in enumerate(data_lines) if line.startswith('Fold')]
        for idx, start_idx in enumerate(fold_start_indices):
            if idx == len(fold_start_indices) - 1:
                fold_data = data_lines[start_idx:]
            else:
                fold_data = data_lines[start_idx:fold_start_indices[idx + 1]]

            fold_name = fold_data[0].split(';')[0].strip()
            fold_sequence_data = [line.strip().split('\t') for line in fold_data[1:]]
            sequence_ids = [row[0] for row in fold_sequence_data]
            #remove "" from sequence_ids
            sequence_ids = [x for x in sequence_ids if x]

            # Read FASTA file containing all sequences
            all_sequences = read_fasta_file(f'../data/{datasettype}/{setting}.fasta')
            output_path = f'../data/{fold_name.replace(" ", "_")}_sequences_{setting}_{datasettype}.fasta'
            with open(output_path, 'w') as fasta_file:
                for seq_id in sequence_ids:
                    if seq_id + " TF=1" in all_sequences:
                        fasta_file.write(f'>{seq_id}\n{all_sequences[seq_id + " TF=1"]}\n')
            print(f'Sequences for {fold_name} written to {output_path}')

    secondary_pre = "../results/" #'/home/bay9657/thesis/results/final_crossCluster/final_AF_norm'
    spatial_file_sec = '../data/uniprot/04_final/AFall_spatial_str.tsv'
    domain_df = pd.read_csv("../data/uniprot/05_default_files/uniprot_DBD_AFs_tfs.tsv", sep='\t', header=0)
    domain_df['start'] = domain_df['start'].astype(int)
    domain_df['end'] = domain_df['end'].astype(int)
    parser = argument_parser()
    options = parser.parse_args()

    device = options.gpu
    num_cpu = options.cpu_num
    batch_size = options.batch_size
    batch_size = 1
    output_dir = options.output_dir
    checkpt_file = options.checkpoint
    protein_data_file = options.seq_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #custom_cmap = ListedColormap(cmap_colors)
    specialComp = ["P37682", "Q46855"] #,"P77700"] # comparison to paper YiaU, YqhC and YahB

    column_names = [f"CV_AFfiltered_expTFs_{clu}_norm_seq0_Com_score", f"CV_AFfiltered_expTFs_{clu}_norm_spatial0_Com_score", f"CV_AFfiltered_expTFs_{clu}_norm_seq0_Com_onehot", f"CV_AFfiltered_expTFs_{clu}_norm_spatial0_Com_onehot", f"CV_AFfiltered_expTFs_{clu}_norm_seq0_Com_attr", f"CV_AFfiltered_expTFs_{clu}_norm_spatial0_Com_attr"]

    # extract ID series from first column of domain_df
    ID_series = domain_df.iloc[:,0]
    # remove duplicates from ID_series
    ID_series = ID_series.drop_duplicates()
    # add a header to ID_series
    ID_series = pd.DataFrame(ID_series, columns=['ID'])
    # Reset the index to ensure unique index values
    ID_series.reset_index(drop=True, inplace=True)


    df = pd.DataFrame(np.empty((len(ID_series),len(column_names)),dtype=object), columns=column_names)
    df['ID'] = ID_series
    df.set_index('ID', inplace=True)

    torch.set_num_threads(num_cpu)
    kernel_size = 1
    #clu = "03"
    not_preproc = True
    if not_preproc:
        for foldID in range(5):
            filename = f"Fold_{foldID}_sequences_{setting}_{datasettype}.fasta"
            print(f"Start {filename}")

            for mode in ['spatial0','seq0']: # add 'seq' if you want to compare to DeepTFactor
                fold_directory = f'../data/'
                protein_data_file = f"{fold_directory}{filename}"
                print(f"Start with {mode}")

                if mode == 'seq0':
                    spatial_mode = 0
                    spatial_file = None
                    prefix = secondary_pre + f"CV_AFfiltered_expTFs_{clu}_norm_{mode}_Com/"

                    protein_seqs, seq_ids, labels = read_fasta_data(protein_data_file)
                    pseudo_labels = np.zeros((len(protein_seqs)))
                    proteinDataset = EnzymeDataset(protein_seqs, pseudo_labels)

                    col_name = f"CV_AFfiltered_expTFs_{clu}_norm_{mode}_Com" #f"no_spatial_{clu}esNo"

                elif mode == 'spatial0':
                    spatial_mode = kernel_size
                    spatial_file = spatial_file_sec
                    prefix = secondary_pre + f"CV_AFfiltered_expTFs_{clu}_norm_{mode}_Com/" # secondary_pre + f"/{spatial_filter}_{clu}esNo/"

                    protein_seqs, seq_ids, spatial_seqs, labels = read_fasta_data_spatial(protein_data_file, spatial_file)
                    pseudo_labels = np.zeros((len(protein_seqs)))
                    proteinDataset = EnzymeDataset_spatial(protein_seqs, labels, spatial_seqs, spatial_mode)

                    col_name = f"CV_AFfiltered_expTFs_{clu}_norm_{mode}_Com"

                # extract the sequence of the IDs that are in specialComp from protien_seqs and seq_ids
                # get the index of the ID in seq_ids
                # get the sequence from protein_seqs
                indices = np.isin(seq_ids, specialComp)
                selected_sequences = protein_seqs[indices]
                # turn those into a dictionary
                special_dict = dict(zip(specialComp, selected_sequences))


                checkpt_file = f"{prefix}own_model{foldID}.pt"
                model = DeepTFactor(spatial_mode, out_features=[1],)
                model = model.to(device)
                cutoff = 0.5
                last_epoch = 0
                test_loader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)
                ckpt = torch.load(f'{checkpt_file}', map_location=device)
                model.load_state_dict(ckpt['model'])


                y_pred = torch.zeros([len(seq_ids), 1])
                captum_ig = captum_attr.IntegratedGradients(model)
                sample_pairs = []
                with torch.no_grad():
                    model.eval()
                    cnt = 0
                    for x, _ in tqdm(test_loader):
                        x = x.type(torch.FloatTensor)
                        x_length = x.shape[0]


                        attributions, _ = captum_ig.attribute(x.to(device), target=0, return_convergence_delta=True)
                        attributions = attributions.cpu().numpy()

                        # Store attributions and x_sample for this batch as a pair
                        for i in range(x_length):
                            sample_pairs.append((attributions[i], x[i].cpu().numpy()))

                        output = model(x.to(device))
                        prediction = output.cpu()
                        y_pred[cnt:cnt+x_length] = prediction
                        cnt += x_length
                # Initialize a list to store feature attributions for all sequences

                scores = y_pred[:,0]
                for idx, proteinID in enumerate(seq_ids):
                    df.at[proteinID, col_name+"_score"] = scores[idx]
                    df.at[proteinID, col_name+"_attr"] = sample_pairs[idx][0]
                    df.at[proteinID, col_name+"_onehot"] = sample_pairs[idx][1]
                print(f"Done with {mode} {kernel_size} for {filename}")

        print("Start evaluation")
        df.dropna(how='all', inplace=True)

        # save the df to a csv file
        df.to_pickle(f"tmp_df_{clu}_all.pkl")
    else:
        df = pd.read_pickle(f"tmp_df_{clu}_all.pkl")
    df = pd.read_pickle(f"tmp_df_{clu}_all.pkl")

    no_spatial_domain_sum = []
    secondary_domain_sum = []
    #pocket_domain_sum = []
    domainFind_eval = []
    # difference Analysis
    no_spatial_domain_dif_sum = []
    secondary_domain_dif_sum = []

    no_neg_spatial_domain_dif_sum = []
    secondary_neg_domain_dif_sum = []
    #pocket_domain_dif_sum = []
    no_spatial_neg_sum = []
    secondary_neg_sum = []

    # logo maker 
    tmp_special_df = []
    sec_spatial_data = pd.read_csv(spatial_file_sec, sep='\t', index_col=0, header=0)
    #tert_spatial_data = pd.read_csv(spatial_file_pocket, sep='\t', index_col=0, header=0)
    # extract domain and non-domain information

    for ID in df.index:
        no_spatial_sample_arr = df.loc[ID, f"CV_AFfiltered_expTFs_{clu}_norm_seq0_Com_onehot"]
        no_spatial_attr_arr = df.loc[ID, f"CV_AFfiltered_expTFs_{clu}_norm_seq0_Com_attr"]
        secondary_attr_arr = df.loc[ID, f"CV_AFfiltered_expTFs_{clu}_norm_spatial0_Com_attr"]
        #pocket_attr_arr = df.loc[ID, f"pocket2_{clu}esNo_attr"]

        upto = np.count_nonzero(no_spatial_sample_arr.squeeze().sum(axis=1))
        #no_spatial_sample_arr = no_spatial_sample_arr.squeeze()[:upto,:] # remove padding
        no_spatial_attr_arr = no_spatial_attr_arr.squeeze()[:upto,:] # remove padding
        secondary_attr_arr = secondary_attr_arr.squeeze()[:upto,:] # remove padding
        #pocket_attr_arr = pocket_attr_arr.squeeze()[:upto,:] # remove padding

        domain_arr = np.zeros(upto, dtype=float)
        # get a sub domain  df that only  has the rows where ID matches the ID column
        sub_domain_df = domain_df[domain_df.ID == ID]
        # change the values in domainstr from subdomain.start to subdomain.end to "d"
        for rowidx in sub_domain_df.index:
            domain_arr[int(domain_df.at[rowidx, "start"])-1:int(domain_df.at[rowidx, "end"])] = 1.0
        tmp_domain_pos = np.where(domain_arr == 1.0)[0]
        tmp_domain_neg = np.where(domain_arr == 0.0)[0]

        no_spatial_attr_sum = np.sum(no_spatial_attr_arr, axis=1)
        secondary_attr_sum = np.sum(secondary_attr_arr, axis=1)

        tmp_dif_no_spatial_sum = no_spatial_attr_sum[tmp_domain_pos] #no_spatial_attr_sum[1:] - no_spatial_attr_sum[:-1]
        tmp_dif_secondary_sum = secondary_attr_sum[tmp_domain_pos] #secondary_attr_sum[1:] - secondary_attr_sum[:-1]

        no_spatial_domain_sum.append(np.sum(no_spatial_attr_sum[tmp_domain_pos]))
        secondary_domain_sum.append(np.sum(secondary_attr_sum[tmp_domain_pos]))

        no_spatial_neg_sum.append(np.sum(no_spatial_attr_sum[tmp_domain_neg]))
        secondary_neg_sum.append(np.sum(secondary_attr_sum[tmp_domain_neg]))

        tmp_special_df.append([ID, no_spatial_attr_sum, secondary_attr_sum, domain_arr])

    tmp_special_df = pd.DataFrame(tmp_special_df, columns=["ID", "no_spatial", "secondary", "domain"])
    tmp_special_df.to_csv(f"tmp_all_df_{clu}_paper.csv")
