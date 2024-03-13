import os
import sys
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, DSSP
from multiprocessing import Pool, cpu_count

########################################################################################
# add the path to your alphafold directory 

########################################################################################

def check_file_existence(id_list_file, output_file, alphafold_path):
    existing_ids = []
    with open(id_list_file, 'r') as file:
        for line in tqdm(file):
            id = line.strip()
            file_path = os.path.join(alphafold_path, f'AF-{id}-F1-model_v4.pdb')
            if os.path.exists(file_path):
                existing_ids.append(id)
    with open(output_file, 'w') as file:
        for id in existing_ids:
            file.write(f'{id}\n')


def get_alphafold_pdb_filepath(seq_id, pdb_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if os.path.isfile(os.path.join(pdb_dir, f))]
    pdb_files = [f for f in pdb_files if seq_id in f]
    if len(pdb_files) == 1:
        return os.path.join(pdb_dir, pdb_files[0])
    elif len(pdb_files) > 1:
        for f in pdb_files:
            if "F1" in f:
                return os.path.join(pdb_dir, f)
    return None

def get_spatial_string(args):
    seq_id, PDB_filename, TF_ids = args
    parser = PDBParser()
    try:
        structure = parser.get_structure(seq_id, PDB_filename)
        model = structure[0]
        dssp = DSSP(model, PDB_filename, dssp='dssp')
        dssp_keys = list(dssp.keys())
    except Exception as e:
        print(f"Problem with id: {seq_id}: {e}")
        return None

    seq = []
    spatial_str = []
    for key in dssp_keys:
        if dssp[key][2] in ('B', 'E'):
            spatial_str.append('b')
        elif dssp[key][2] in ('H', 'G', 'I'):
            spatial_str.append('a')
        else:
            spatial_str.append('N')
        seq.append(dssp[key][1])

    spatial_str = ''.join(spatial_str)
    seq = ''.join(seq)

    TF = 1 if seq_id in TF_ids else 0
    return [seq_id, seq, spatial_str, f'TF={TF}']

def apply_dssp_to_alpha(seq_ids, alphafold_path, if_TF_file, spatial_str_file, fasta_file):
    alphafasta = []
    TF_ids = pd.read_csv(if_TF_file, sep="\t", header=None)[0].tolist()
    args_list = [(seq_id, f"{alphafold_path}AF-{seq_id}-F1-model_v4.pdb", TF_ids) for seq_id in seq_ids]

    with Pool(cpu_count() - 1) as pool:  # Use all available CPU cores except one
        results = list(tqdm(pool.imap(get_spatial_string, args_list), total=len(seq_ids)))
        for result in results:
            if result is not None:
                alphafasta.append(result)

    with open(spatial_str_file, 'a') as f1, open(fasta_file, 'a') as f2:
        for seq in alphafasta:
            f1.write(f'{seq[0]}\t{seq[2]}\n')
            f2.write(f'>{seq[0]} {seq[3]}\n{seq[1]}\n')

def get_sufficent_alpha_ids(AF_file):
    scores = pd.read_csv(AF_file, sep="\t", index_col=0, header=0)
    seq_ids = scores.index.values
    return seq_ids

if __name__ == '__main__':

    if len(sys.argv) == 3:
        id_list_file = sys.argv[1]
        output_file = sys.argv[2]
    else: 
        id_list_file = "../../data/uniprot/03_filtered/uniprot_ids_filtered.tsv"
        output_file = "../../data/uniprot/05_default_files/AF_ids.tsv"
    
    #if os.path.exists(output_file):
    #    print(f"{output_file} already exists, please remove it first")
    #    sys.exit(1)
    #else:
    #    os.makedirs(os.path.dirname("../../data/uniprot/05_default_files/"), exist_ok=True)
    #
    alphafold_path = "./../../../alphafold/"
    
    if_TF_file = "../../data/uniprot/02_without_duplicateIDs/uniprot_TF_ids_NOdupl.tsv"

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname("../../data/uniprot/05_default_files/"), exist_ok=True)
        check_file_existence(id_list_file, output_file, alphafold_path)

    # add header to the files
    # save the file first here to not overwrite the orignal 
    spatial_str_file = f'../../data/uniprot/04_final/AFall_spatial_str.tsv'
    if not os.path.exists(spatial_str_file):
        os.makedirs(os.path.dirname("../../data/uniprot/04_final/"), exist_ok=True)

    
    fasta_file = f'../../data/uniprot/05_default_files/AFall_seqs.fasta'
    with open(spatial_str_file, 'w') as f1, open(fasta_file, 'w') as f2:
        f1.write('\tspatial_str\n')
        f2.write('\n')
    
    apply_dssp_to_alpha(get_sufficent_alpha_ids(output_file), alphafold_path, if_TF_file, spatial_str_file, fasta_file)

