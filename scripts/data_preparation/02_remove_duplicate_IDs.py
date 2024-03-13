import os
def remove_duplicates(input_file, output_file):
    unique_rows = set()
    counter = 0
    set_dupl = set()
    with open(input_file, 'r') as infile:
        for line in infile:
            row = line.strip()
            if row not in unique_rows:
                unique_rows.add(row)
            else:
                counter += 1
                if row not in set_dupl:
                    set_dupl.add(row)
    print(f'Number of duplicates: {counter}')
    print(f'Number of unique rows: {len(unique_rows)}')
    print(f'Number of unique duplicates: {len(set_dupl)}')
    
    with open(output_file, 'w') as outfile:
        for row in unique_rows:
            outfile.write(row + '\n')

if os.path.exists('../../data/uniprot/02_without_duplicateIDs'):
    pass
else:
    os.makedirs('../../data/uniprot/02_without_duplicateIDs')

# concatenate '../../data/uniprot/01_all_uniprot/uniprot_TF_ids.tsv' nd '../../data/uniprot/01_all_uniprot/uniprot_nonTF_ids.tsv' into one file
# and remove duplicates
# load both files
with open('../../data/uniprot/01_all_uniprot/uniprot_TF_ids.tsv', 'r') as file:
    TF_ids = set(line.strip() for line in file)
with open('../../data/uniprot/01_all_uniprot/uniprot_nonTF_ids.tsv', 'r') as file:
    nonTF_ids = set(line.strip() for line in file)
# concatenate both sets
all_ids = TF_ids.union(nonTF_ids)
# write the concatenated set to a file
with open('../../data/uniprot/01_all_uniprot/uniprot_ids.tsv', 'w') as file:
    for uniprot_id in all_ids:
        file.write(f'{uniprot_id}\n')

#For all uniprot IDs
input_file = '../../data/uniprot/01_all_uniprot/uniprot_ids.tsv'
output_file = '../../data/uniprot/02_without_duplicateIDs/uniprot_ids_NOdupl.tsv'
remove_duplicates(input_file, output_file)

# Consistency check for the TF and nonTF IDs
# TF
input_file = '../../data/uniprot/01_all_uniprot/uniprot_TF_ids.tsv'
output_file = '../../data/uniprot/02_without_duplicateIDs/uniprot_TF_ids_NOdupl.tsv'
remove_duplicates(input_file, output_file)

# nonTF
input_file = '../../data/uniprot/01_all_uniprot/uniprot_nonTF_ids.tsv'
output_file = '../../data/uniprot/02_without_duplicateIDs/uniprot_nonTF_ids_NOdupl.tsv'
remove_duplicates(input_file, output_file)



