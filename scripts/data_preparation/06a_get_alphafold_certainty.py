import os
import gzip
import glob
from tqdm import tqdm

def process_pdb_file(filename):
    with gzip.open(filename, 'rt') if filename.endswith(".pdb.gz") else open(filename, 'r') as f:
        lines = f.readlines()
    list_values = []
    for line in lines:
        if line.startswith('ATOM'):
            if "CA" in line:
                fields = line.split()
                value = float(fields[-2])
                list_values.append(value)

    percentages = []
    for threshold in thresholds:
        # sort list_values
        list_values.sort()
        thres_ID = int(len(list_values) * ((100-threshold) / 100))
        threshold = list_values[thres_ID]
        percentages.append(list_values[thres_ID])
    return percentages

directory = "./../../../alphafold/"
output_file = "../../data/uniprot/05_default_files/AF_scores.tsv"
thresholds = [90, 80, 70, 60, 50]

with open(output_file, 'w') as f:
    f.write("ID\t90%\t80%\t70%\t60%\t50%\n")
    pdb_files = glob.glob(os.path.join(directory, "*.pdb*"))

    for i, pdb_file in tqdm(enumerate(pdb_files)):
        pdb_id = os.path.basename(pdb_file).split("-")[1]
        percentages = process_pdb_file(pdb_file)
        percentage_str = '\t'.join(str(p) for p in percentages)
        f.write("{}\t{}\n".format(pdb_id, percentage_str))
