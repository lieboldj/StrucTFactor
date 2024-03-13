# given a pdb file as argument, this file predict, whether it is an transcription factor or not. 
# you can choose your own pre-trainefd model or take the default model
# run it via: python predictTFwithStrucTFactor.py -i example.pdb -o output.csv -ckpt strucTFactor_model.pt -g cuda:0
import torch
from torch.utils.data import DataLoader

from deeptfactor.data_loader import EnzymeDataset_spatial
from deeptfactor.utils import argument_parser
from deeptfactor.models import DeepTFactor

import argparse
from Bio.PDB import PDBParser, DSSP

def argument_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cpu', help='Specify gpu')
    parser.add_argument('-ckpt', '--checkpoint', required=False, 
                        default="strucTFactor_model.pt", help='Checkpoint file needed!')
    parser.add_argument('-i', '--pdb_file', required=False, 
                        default='example.pdb', help='PDB file')
    parser.add_argument('-o', '--output', required=False, 
                        default='output.csv', help='Output file')
    return parser



def get_spatial_string(PDB_filename, seq_id):
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

    return [seq, spatial_str]


if __name__ == '__main__':

    parser = argument_parser()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    checkpt_file = args.checkpoint
    pdb_file = args.pdb_file
    output_file = args.output
    # Load the model
    model = DeepTFactor(1, out_features=[1],)
    model = model.to(device)
    cutoff = 0.5

    # run DSSP to get spatial information
    protein_seqs, spatial_seqs = get_spatial_string(pdb_file, 'test')
    pdb_file.split('.')[-2]
    seq_ids = pdb_file.split('.')[-2]
    protein_seqs += '_' * (1000-len(protein_seqs)) # zero-padding
    proteinDataset = EnzymeDataset_spatial([protein_seqs],torch.zeros([1, 1]), [spatial_seqs], 1)

    test_loader = DataLoader(proteinDataset, batch_size=1, shuffle=False)
    ckpt = torch.load(f'{checkpt_file}', map_location=device)
    model.load_state_dict(ckpt['model'])

    y_pred = torch.zeros([1, 1])
    with torch.no_grad():
        model.eval()
        cnt = 0
        for x, _ in test_loader:
            x = x.type(torch.FloatTensor)
            output = model(x.to(device))
            prediction = output.cpu()
    if prediction > cutoff:
        print(f"Prediction for {seq_ids}: Transcription Factor")
    else:
        print(f"Prediction for {seq_ids}: Not a Transcription Factor")
    with open(output_file, 'w') as f:
        f.write(f"{seq_ids}\t{prediction.item()}\n")