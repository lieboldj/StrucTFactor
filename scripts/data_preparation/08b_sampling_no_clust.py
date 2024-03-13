import sys
import random
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm 

def get_TF_proteins(TF_ids_file):
    return set(pd.read_csv(TF_ids_file, sep="\t", header=None, squeeze=True))

def subsample_fasta(fasta_name, set_name, TFs, ratio):
    print(fasta_name)
    sampled_reps_TF = set()
    with open(fasta_name, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                if 'TF=1' in line:
                    sampled_reps_TF.add(line.split()[0][1:])
    fasta_seqs = SeqIO.parse(fasta_name, "fasta")
    #sampled_reps_TF = set(record.id for record in fasta_seqs if record.id in TFs)  # All TFs are sampled TFs
    len_tf = len(sampled_reps_TF)
    print("TFs extracted.", len_tf)

    if ratio == "norm":
        fasta_seqs = SeqIO.parse(fasta_name, "fasta")
        sampled_reps_nonTF = set(record.id for record in fasta_seqs if record.id not in TFs)
        norm = random.sample(sampled_reps_nonTF, 3 * len_tf)
        print("1:3 set created.")

    if ratio == "small":
        fasta_seqs = SeqIO.parse(fasta_name, "fasta")
        sampled_reps_nonTF = set(record.id for record in fasta_seqs if record.id not in TFs)
        norm = random.sample(sampled_reps_nonTF, 5 * len_tf)
        print("1:5 set created.")

    if ratio == "large":
        fasta_seqs = SeqIO.parse(fasta_name, "fasta")
        sampled_reps_nonTF = set(record.id for record in fasta_seqs if record.id not in TFs)
        norm = random.sample(sampled_reps_nonTF, 10 * len_tf)
        print("1:10 set created.")

    print(len(norm))
    fasta_seqs = SeqIO.parse(fasta_name, "fasta")
    sampled_reps_TF = set(record.id for record in fasta_seqs if record.id in TFs)  # All TFs are sampled TFs
    print("TFs extracted.")
    fasta_seqs = SeqIO.parse(fasta_name, "fasta")
    # Save the TF sequences and sampled non-TF sequences to separate files
    # "No" stands for "no clustering"
    with open(f'{set_name}/No_{ratio}_Com.fasta', 'w') as f_norm:
        for record in fasta_seqs:
            if record.id in sampled_reps_TF:
                f_norm.write(f'>{record.id} TF=1\n{record.seq}\n')
            elif record.id in norm:
                f_norm.write(f'>{record.id} TF=0\n{record.seq}\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python 08b_sampling_no_clust.py <set_name>\n\
                    e.g. python 08b_sampling_no_clust.py AFall norm")
        sys.exit(1)

    data_path = "../../data/"
    set_name = data_path + sys.argv[1] # type of dataset of flexible file naming

    ratio = sys.argv[2]
    TF_ids_file = data_path + f"uniprot/02_without_duplicateIDs/uniprot_TF_ids_NOdupl.tsv"
    TFs = get_TF_proteins(TF_ids_file)
    print("TFs extracted")
    
    fasta_seqs = data_path+"uniprot/04_final/"+sys.argv[1]+f"_seqs.fasta"

    subsample_fasta(fasta_seqs, set_name, TFs, ratio)
