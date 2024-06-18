import numpy as np
import pandas as pd

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

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18

clu = "03" #"03" #"No"

#df = pd.read_csv(f'./tmp_special_df_clu03_paper.csv', header=0) # change to the csv file
df = pd.read_csv(f'./d_rl_nr_3_domain.csv', header=0) # change to the csv file
protein_data_file = f'../../data/uniprot/04_final/AFall_seqs.fasta' # change to your fasta file

sequences = parse_fasta(protein_data_file)
# plot a motif plot for analysis
for idx, p in df.iterrows():
    if p.ID == "P11470":
        p.seq = extract_sequence_from_fasta(sequences, p.ID)
        # p.domain is a numpy array wit 0 and 1, tell me all position numbers that are 1
        domain_list_pre = p.domain
        domain_list_pre = p.domain.lstrip("[ ").rstrip(" ]").split()
        domain_list = np.array([float(i) for i in domain_list_pre])
        nums = np.where(domain_list == 1.)
        secondary_list_pre = p.secondary
        secondary_list_pre = secondary_list_pre.lstrip("[ ").rstrip(" ]").split()
        secondary_list = [float(i) for i in secondary_list_pre]
        # for a good plot size
        y_max = max(secondary_list) * 1.1
        y_min = min(secondary_list) * 2.2

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)  # Create subplots
        
        # Plot first subplot
        secondary_logo = lm.saliency_to_matrix(seq=p.seq, values=secondary_list)
        lo1 = lm.Logo(secondary_logo, ax=ax1, figsize=(12, 4.5), color_scheme='chemistry')
        ax1.set_ylim(y_min, y_max)
        ax1.set_title("StrucTFactor", loc="left")
        if len(nums[0]) != 0:
            nums = [int(i) for i in nums[0]]
            for i in nums:
                lo1.highlight_position(p=i, color='pink')

        # Plot second subplot
        seq_list_pre = p.no_spatial
        seq_list_pre = seq_list_pre.lstrip("[ ").rstrip(" ]").split()
        seq_list = [float(i) for i in seq_list_pre]

        secondary_logo = lm.saliency_to_matrix(seq=p.seq, values=seq_list)  # Assuming you want the same data
        lo2 = lm.Logo(secondary_logo, ax=ax2, figsize=(12, 4.5), color_scheme='chemistry')
        #y_max = max(seq_list) * 1.1
        #y_min = min(seq_list) * 1.1
        ax2.set_ylim(y_min, y_max)
        ax2.set_title("DeepTFactor", loc="left")
        nums = np.where(domain_list == 1.)
        if len(nums[0]) != 0:
            nums = [int(i) for i in nums[0]]
            for i in nums:
                lo2.highlight_position(p=i, color='pink')
        ax2.set_xlabel("Residue position", labelpad=20)
        #ax2.set_ylabel('Integrated gradient score')

        fig.text(-0.04, 0.5, 'Integrated gradient score', va='center', rotation='vertical')
        fig.savefig(f"../plots/Paper/{p.ID}_motif.pdf", bbox_inches='tight')