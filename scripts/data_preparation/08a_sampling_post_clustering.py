import sys
import random
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

def get_TF_proteins(TF_ids_file):
    return set(pd.read_csv(TF_ids_file, sep="\t", header=None, squeeze=True))

def process_cluster(filename, TFs):
    df = pd.read_csv(filename, sep="\t", header=None, names=["rep", "members"])
    sampled_reps_TF = set()
    sampled_reps_nonTF = set()
    for unique, members in tqdm(df.groupby("rep")):
        if unique in TFs:
            sampled_reps_TF.add(unique)
        else:
            candidate = members[members.isin(TFs)].dropna()  # Drop NaN values
            if not candidate.empty:
                sampled_reps_TF.add(candidate.iloc[0])  # Extract value from Series
            else:
                sampled_reps_nonTF.add(unique)

    return sampled_reps_TF, sampled_reps_nonTF

def subsample_fasta(fasta_seqs, sampled_reps_TF, sampled_reps_nonTF, set_name, cluster_size, ratio):
    len_tf = len(sampled_reps_TF)
    if ratio == "norm":
        norm = random.sample(sampled_reps_nonTF, 3 * len_tf)
    if ratio == "small":
        norm = random.sample(sampled_reps_nonTF, 5 * len_tf)
    if ratio == "large":
        norm = random.sample(sampled_reps_nonTF, 10 * len_tf)

    # save three fasta files for each norm, small, large and save all sampled reps TF and norm, small, large
    with open(f'{set_name}/{cluster_size}_all.fasta', 'w') as f_TF, \
         open(f'{set_name}/{cluster_size}_{ratio}_Com.fasta', 'w') as f_norm:
         for record in fasta_seqs:
            if record.id in sampled_reps_TF:
                f_TF.write(f'>{record.id} TF=1\n{record.seq}\n')
                f_norm.write(f'>{record.id} TF=1\n{record.seq}\n')
            elif record.id in norm:
                f_norm.write(f'>{record.id} TF=0\n{record.seq}\n')
                f_TF.write(f'>{record.id} TF=0\n{record.seq}\n')
            elif record.id in sampled_reps_nonTF:
                f_TF.write(f'>{record.id} TF=0\n{record.seq}\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print("Usage: python 08a_sampling_post_clustering.py <set_name>\n\
              e.g. python 08a_sampling_post_clustering.py AFall norm")
        sys.exit(1)

    data_path = "../../data/"
    
    set_name = data_path + sys.argv[1] # type of dataset of flexible file naming
    ratio = sys.argv[2] # norm, small, large

    TF_ids_file = data_path + f"uniprot/02_without_duplicateIDs/uniprot_TF_ids_NOdupl.tsv"
    TFs = get_TF_proteins(TF_ids_file)
    

    cluster_size = "03"
    print("run", cluster_size)
    fasta_seqs = SeqIO.parse(data_path+"uniprot/04_final/"+sys.argv[1]+"_seqs.fasta", "fasta")
    
    sampled_reps_TF, sampled_reps_nonTF = process_cluster(f"{set_name}/{cluster_size}_cluster.tsv", TFs)
    print(f"clu{cluster_size} done clu{cluster_size}_TF: {len(sampled_reps_TF)}, clu{cluster_size}_nonTF: {len(sampled_reps_nonTF)}") 
    subsample_fasta(fasta_seqs, sampled_reps_TF, sampled_reps_nonTF, set_name, cluster_size, ratio)
