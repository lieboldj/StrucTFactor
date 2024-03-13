# write a fucntion that reads in a fasta file with SeqIO and gets the sequence and the id and returns the id if the length of the sequence is more then 1000 or if the sequence contains non standard amino acids
import os
import sys 
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def get_id(filename):
    with open(filename, 'r') as f:
        for record in tqdm(SeqIO.parse(f, 'fasta')):
            if len(record.seq) > 1000:
                yield record.id.split(' ')[0]
            else:
                for aa in record.seq:
                    if aa not in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                        yield record.id.split(' ')[0]
                        break

def create_not_fitting_file(input_filename, output_filename):
    fasta_ids = list(get_id(input_filename))
    df = pd.DataFrame(fasta_ids)
    df.to_csv(output_filename, sep='\t', index=False, header=False)

def get_ids_to_remove(filename):
    ids_to_remove = set()
    with open(filename, 'r') as f:
        for line in f:
            ids_to_remove.add(line.strip())
    return ids_to_remove


def filter_fasta(input_fasta, output_fasta, ids_to_remove):
    with open(input_fasta, 'r') as fasta_in, open(output_fasta, 'w') as fasta_out:
        for record in SeqIO.parse(fasta_in, 'fasta'):
            if record.id.split(' ')[0] not in ids_to_remove:
                # Remove newline characters from the sequence string
                sequence = str(record.seq).rstrip('\n')
                # Write the record without newline characters to the output FASTA file
                fasta_out.write(f'>{record.id} {record.description}\n{sequence}\n')

# get a updated list of all ids available in the fasta file
def extract_ids_from_fasta(fasta_file):
    ids = []
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
    return ids

def save_ids_to_tsv(ids, tsv_file):
    df = pd.DataFrame({"ID": ids})
    df.to_csv(tsv_file, sep='\t', index=False, header=False)

# either create not fitting file or use the one that is already there to filter the fasta file
if len(sys.argv) == 3:
    input_fasta_file = sys.argv[1]
    ids_to_remove_file = sys.argv[2]
elif len(sys.argv) == 4:
    input_fasta_file = sys.argv[1]
    ids_to_remove_file = sys.argv[2]
    output_fasta_file = sys.argv[3]
elif len(sys.argv) == 5:
    input_fasta_file = sys.argv[1]
    ids_to_remove_file = sys.argv[2]
    output_fasta_file = sys.argv[3]
    filtered_tsv = sys.argv[4]
else:
    input_fasta_file = '../../data/uniprot/03_filtered/uniprot_preproc.fasta'
    ids_to_remove_file = '../../data/uniprot/03_filtered/uniprot_not_fitting.tsv'
    output_fasta_file = '../../data/uniprot/03_filtered/uniprot_filtered.fasta'
    filtered_tsv = '../../data/uniprot/03_filtered/uniprot_ids_filtered.tsv'

if not os.path.exists(ids_to_remove_file):
    create_not_fitting_file(input_fasta_file, ids_to_remove_file)

# if output fasta file is not given then exit
if len(sys.argv) == 3:
    sys.exit(0)
    
ids_to_remove = get_ids_to_remove(ids_to_remove_file)
filter_fasta(input_fasta_file, output_fasta_file, ids_to_remove)

if len(sys.argv) == 4:
    sys.exit(0)

save_ids_to_tsv(extract_ids_from_fasta(output_fasta_file), filtered_tsv)
