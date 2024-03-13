import sys
import pandas as pd

def filter_fasta(in_file, out_file, protein_ids, tf_id_file):
    tf_1_ids = set()
    with open(tf_id_file, 'r') as tf_id_file:
        for line in tf_id_file:
            id = line.strip()
            tf_1_ids.add(id)
    with open(in_file, 'r') as infile, open(out_file, 'w') as outfile:
        current_entry_id = None
        current_entry_lines = []
        for line in infile:
            if line.startswith('>'):
                # Process the previous entry, if any
                if current_entry_id is not None:
                    if current_entry_id in protein_ids:                        
                        outfile.write(''.join(current_entry_lines))
                # Start processing the new entry
                current_entry_id = line.strip()[1:].split()[0]
                tf_value = '1' if current_entry_id in tf_1_ids else '0'
                current_entry_lines = [f">{current_entry_id} TF={tf_value}\n"]
            else:
                # Append sequence lines to the current entry
                current_entry_lines.append(line)

        # Process the last entry in the file
        if current_entry_id is not None and current_entry_id in protein_ids:
            outfile.write(''.join(current_entry_lines))


def get_alpha_ids(ofAA, threshold, AF_file, scores_file):
    scores = pd.read_csv(scores_file, sep="\t", index_col=0, header=0)
    ids = set(pd.read_csv(AF_file, sep="\t", index_col=0, header=0).index.values)
    scores = scores[scores[ofAA] > threshold]
    seq_ids = set(scores.index.values)
    seq_ids = seq_ids.intersection(ids)
    return seq_ids

def remove_Com_TFs(tf_id_file, in_file, out_file):
    com_TFs = set()
    with open(tf_id_file, 'r') as file:
        for line in file:
            entry = line.strip()  # Remove any leading/trailing whitespace or newline characters
            com_TFs.add(entry)

    with open(in_file, 'r') as fasta_file, open(out_file, 'w') as output:  
        for line in fasta_file:
            if line.startswith('>'):
                keep_entry = True
                if "TF=1" in line:
                    for entry in com_TFs:
                        if entry in line:
                            keep_entry = False
                            break
                    if keep_entry:
                        output.write(line)  # Write the fasta header
                elif "TF=0" in line:
                    output.write(line)
            elif keep_entry:
                output.write(line)  # Write the sequence if the entry is to be kept


if __name__ == '__main__':
    # make the setting how to choose structures from alphafold based on AA and threshold
    ofAA = "90%"
    threshold = 70
    output_file =   "../../data/uniprot/05_default_files/AF_ids.tsv"
    scores_file =   "../../data/uniprot/05_default_files/AF_scores.tsv"
    in_file =       "../../data/uniprot/04_final/AFall_seqs.fasta"
    out_file =      "../../data/uniprot/04_final/AFfiltered_seqs.fasta"
    tf_id_file =    "../../data/uniprot/02_without_duplicateIDs/uniprot_TF_ids_NOdupl.tsv"

    filter_fasta(in_file, out_file, get_alpha_ids(ofAA, threshold, output_file, scores_file), tf_id_file)