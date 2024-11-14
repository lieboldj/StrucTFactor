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
import matplotlib.pyplot as plt
import logomaker as lm

def parse_fasta(filename):
    sequences = {}
    with open(filename, 'r') as file:
        sequence_id = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id:
                    sequences[sequence_id] = sequence
                if len(line) > 7:
                    sequence_id = line[1:-5]
                else:
                    sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        if sequence_id:
            sequences[sequence_id] = sequence
    return sequences

def extract_sequence_from_fasta(sequences, sequence_id):
    return sequences.get(sequence_id, None)

# python tf_running_one_protein.py -i ../data/test_seqs.fasta -s ../data/uniprot/04_final/AFall_spatial.tsv -o ../../results/ -ckpt ../results/CV_AFfiltered_expTFs_clu03_norm_spatial_Com/own_model0.pt -g cuda:0
if __name__ == '__main__':
    # change your prefered setting here
    create_plot_per_protein = True # set to True and you get a motif plot per protein
    setting = "03_norm_Com"
    datasettype = "AFfiltered"
    clu = "clu" + setting.split("_")[0]
    ratio = setting.split("_")[1]
    mode = "spatial" # change to "seq" for DeepTFactor comparison
    #filename = f"../data/{datasettype}/{setting}.fasta"# write your proteins of choice in a fasta, e.g. f"../data/test_seqs.fasta"
    filename = f"../data/test_seqs.fasta"

    secondary_pre = "../results/"
    spatial_file_sec = '../data/uniprot/04_final/AFall_spatial_str.tsv' # to get spatial information for all seqs 

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

    column_names = ["score", "onehot", "attr"]
    torch.set_num_threads(num_cpu)
    
    kernel_size = 1 

    if mode == 'seq':
        spatial_file = None

        protein_seqs, seq_ids, labels = read_fasta_data(protein_data_file)
        pseudo_labels = np.zeros((len(protein_seqs)))
        proteinDataset = EnzymeDataset(protein_seqs, pseudo_labels)

        
    elif mode == 'spatial':
        spatial_mode = kernel_size
        spatial_file = spatial_file_sec

        protein_seqs, seq_ids, spatial_seqs, labels = read_fasta_data_spatial(protein_data_file, spatial_file)
        pseudo_labels = np.zeros((len(protein_seqs)))
        proteinDataset = EnzymeDataset_spatial(protein_seqs, labels, spatial_seqs, spatial_mode)

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
            #attributions, _ = captum_ig.attribute(x.to(device), target=0, return_convergence_delta=True)
            #attributions = attributions.cpu().numpy()

            x = x.to(device)  # Move input to appropriate device (e.g., GPU)
            target = 0  # Target class index for attribution
            
            # Initialize Integrated Gradients
            ig = captum_attr.IntegratedGradients(model)
            model.train()
            # run in training mode for backward in lstm
            # Compute attributions
            attributions = ig.attribute(x, target=target)
            model.eval()
            
            # Move attributions to CPU and convert to numpy
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
    explain_dict = pd.DataFrame(index=seq_ids, columns=column_names)
    for idx, proteinID in enumerate(seq_ids):
        upto = np.count_nonzero(sample_pairs[idx][1].squeeze().sum(axis=1))
        explain_dict.at[proteinID, "score"] = scores[idx]
        explain_dict.at[proteinID, "attr"] = sample_pairs[idx][0].squeeze()[:upto,:]
        explain_dict.at[proteinID, "onehot"] = sample_pairs[idx][1].squeeze()[:upto,:]

    print("Done!")

    if create_plot_per_protein:
        # analyse the impact score with the domain information
        secondary_domain_sum = []
        secondary_neg_sum = []

        # logo maker 
        tmp_df = []

        # extract domain and non-domain information for each protein in dict
        domain_df = pd.read_csv(f"../data/uniprot/05_default_files/uniprot_DBD_AFs.tsv", sep="\t")
        domain_df = domain_df[domain_df["ID"].isin(explain_dict.index)]
        domain_df = domain_df[["ID", "start", "end"]]
        domain_df = domain_df.sort_values(by="ID")

        for ID in explain_dict.index:
            domain_arr = np.zeros(explain_dict.at[ID, "attr"].shape[0], dtype=float)
            # get a sub domain  df that only  has the rows where ID matches the ID column
            sub_domain_df = domain_df[domain_df.ID == ID]
            # change the values in domainstr from subdomain.start to subdomain.end to "d"
            for rowidx in sub_domain_df.index:
                print(sub_domain_df, rowidx)
                domain_arr[int(domain_df.at[rowidx, "start"])-1:int(domain_df.at[rowidx, "end"])] = 1.0
            tmp_domain_pos = np.where(domain_arr == 1.0)[0]
            tmp_domain_neg = np.where(domain_arr == 0.0)[0]

            secondary_attr_arr = explain_dict.at[ID, "attr"]
            secondary_attr_sum = np.sum(secondary_attr_arr, axis=1)

            tmp_dif_secondary_sum = secondary_attr_sum[tmp_domain_pos] #secondary_attr_sum[1:] - secondary_attr_sum[:-1]
            secondary_domain_sum.append(np.sum(secondary_attr_sum[tmp_domain_pos]))
            secondary_neg_sum.append(np.sum(secondary_attr_sum[tmp_domain_neg]))

            tmp_df.append([ID, secondary_attr_sum, domain_arr])

        tmp_special_df = pd.DataFrame(tmp_df, columns=["ID", "struct", "domain"])
        # for overall evaluation, save file
        #tmp_special_df.to_csv(f"explained_{datasettype}_{setting}_test.csv")


        # df = pd.read_csv(f'explained_{datasettype}_{setting}_test.csv') # change to the csv file
        df = tmp_special_df

        sequences = parse_fasta(protein_data_file)

        # plot a motif plot for analysis
        for idx, p in df.iterrows():
            p.seq = extract_sequence_from_fasta(sequences, p.ID)

            # p.domain is a numpy array wit 0 and 1, tell me all position numbers that are 1
            domain_list_pre = p.domain

            domain_list = np.array([float(i) for i in domain_list_pre])
            nums = np.where(domain_list == 1.)

            secondary_list_pre = p.struct
            secondary_list = [float(i) for i in secondary_list_pre]

            # for a good plot size
            y_max = max(secondary_list) * 1.1
            y_min = min(secondary_list) * 1.1

            fig,ax2 = plt.subplots(figsize=(12,4.5), constrained_layout=True)# layout="constrained")#sharey=True)
            secondary_logo = lm.saliency_to_matrix(seq=p.seq, values=secondary_list)
            lo2 = lm.Logo(secondary_logo, ax=ax2, figsize=(12,4.5), color_scheme='chemistry')
            #lo2 = lm.Logo(secondary_logo, ax=ax2, figsize=(18,3), color_scheme=secondary_scheme)
            ax2.set_ylim(y_min, y_max)
            ax2.set_title("StrucTFactor", loc="left")

            if len(nums[0]) != 0:
                nums = [int(i) for i in nums[0]]
                for i in nums:
                    lo2.highlight_position(p=i, color='pink')

            ax2.set_xlabel("Residue position", labelpad=20)
            fig.text(-0.04, 0.5, 'Integrated gradient score', va='center', rotation='vertical')
            fig.savefig(f"{p.ID}_clu{setting}_{datasettype}.pdf", bbox_inches='tight')

