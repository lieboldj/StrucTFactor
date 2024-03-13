import os
import sys
from tqdm import tqdm

def read_in_step(prediction_file, data_file, output_fasta):
    # Read the list of IDs from the prediction file
    with open(prediction_file, 'r') as file:
        all_ids = set(line.strip() for line in file)
    
    # Extract sequences for the unique IDs from the data file
    sequences = {}
    current_ids = []
    current_sequence = ""
    with open(data_file, 'r') as file:
        for line in tqdm(file):
            if line.startswith("AC"):
                # Extract the IDs from the AC line
                current_ids.extend(line.strip().split()[1].split(";"))
            elif line.startswith("SQ"):
                # Start extracting sequence
                current_sequence = ""
                for next_line in file:
                    if next_line.startswith("//"):
                        # Reached the end of sequence, store the sequence for each ID
                        for current_id in current_ids:
                            if current_id in all_ids:
                                sequences[current_id] = current_sequence.replace(" ", "")
                        current_ids = []  # Clear the list of ACs
                        break
                    # Remove spaces and concatenate to the sequence
                    current_sequence += next_line.strip().replace(" ", "")
    # Write the filtered sequences to a FASTA file
    with open(output_fasta, 'w') as f:
        for uniprot_id, sequence in sequences.items():
            f.write(f'>{uniprot_id}\n{sequence}\n')

if __name__ == "__main__":

    if os.path.exists("../../data/uniprot/03_filtered/"):
        pass
    else:
        os.makedirs("../../data/uniprot/03_filtered/")

    read_in_step("../../data/uniprot/02_without_duplicateIDs/uniprot_ids_NOdupl.tsv", "../../data/uniprot/01_all_uniprot/uniprot_sprot.dat", "../../data/uniprot/03_filtered/uniprot_preproc.fasta")
