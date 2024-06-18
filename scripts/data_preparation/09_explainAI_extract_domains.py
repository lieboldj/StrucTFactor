import pandas as pd
from tqdm import tqdm

def get_info(fileIN, IDs):
    print("Start loop")
    with open("../../data/uniprot/05_default_files/uniprot_DBD_AFs.tsv", "w") as f_out:
        f_out.write("ID\tstart\tend\ttype\n")
        with open(fileIN, "r") as f_in:
            to_add = False
            nextLine = False
            prev_AC = False
            DNA_BIND = False
            for line in tqdm(f_in):
                if nextLine:
                    if line.startswith("FT                   /note="):
                        domain_str = line[5:].strip()
                        domain_name = domain_str.split("=")[1].strip('"')
                        if DNA_BIND:
                            domain_nums.append(domain_name)
                        else:
                            if domain_name.startswith(("H-T-H motif", "HTH ", "Leucine-zipper", "bZIP")):
                                domain_nums.append(domain_name)
                            else:
                                continue
                        try:
                            domain = [
                                (i, int(start), int(end), typ)  # Tuple unpacking and conversion to integers
                                for i, start, end, typ in domain  # Iterate over domain tuples
                                if not (int(start) <= int(domain_nums[1]) <= int(end))  # Filter condition
                            ]
                        except:
                            for i, start, end, typ in domain:
                                #print(f"i: {i}, start: {start}, end: {end}, typ: {typ}, domain_nums[1]: {domain_nums[1]}")
                                # check if start, end or domain_nums[1] is an integer:
                                if not isinstance(start, int):
                                    start = start.replace("?", "")
                                if not isinstance(end, int):
                                    end = end.replace("?", "")
                                if not isinstance(domain_nums[1], int):
                                    domain_nums[1] = domain_nums[1].replace("?", "")                             
                            
                        domain.append(domain_nums)
                    nextLine = False
                    DNA_BIND = False

                if line.startswith("AC"):
                    if prev_AC:
                        continue
                    prev_AC = True
                    if to_add:
                        for row in domain:
                            f_out.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
                        to_add = False
                    tmp_id = line[4:].split(";")[0].strip(" ;")
                    if tmp_id in IDs:
                        to_add = True
                        domain = []

                else:
                    prev_AC = False

                if to_add is False:
                    continue

                if line.startswith("FT   "):
                    domain_str = line[5:]
                    if domain_str.startswith(("DOMAIN", "REGION", "DNA_BIND", "ZN_FING")):
                        domain_str = domain_str.split()[1].strip().split("..")
                        if len(domain_str) < 2:
                            continue
                        domain_nums = [tmp_id, domain_str[0].strip("<"), domain_str[1].strip(">")]
                        if domain_str[0].startswith("DNA_BIND"):
                            DNA_BIND = True
                        nextLine = True
                    elif domain_str.startswith("ZN_FING"):
                        domain_str = domain_str.split()[1].strip().split("..")
                        if len(domain_str) < 2:
                            continue
                        domain_nums = [tmp_id, domain_str[0].strip("<"), domain_str[1].strip(">"), "ZN_FING"]
                        domain = [(i, int(start), int(end), typ) for i, start, end, typ in domain if
                                  not (int(start) <= int(domain_nums[1]) <= int(end))]
                        domain.append(domain_nums)

    print("Loop end")

fileIN = "../../data/uniprot/01_all_uniprot/uniprot_sprot.dat"
ID_file = pd.read_csv("../../data/uniprot/05_default_files/AF_ids.tsv", sep="\t", header=0, index_col=0)
IDs = set(ID_file.index)
get_info(fileIN, IDs)

# make sure that we only have TFs in this list
tfs = pd.read_csv('../../data/uniprot/01_all_uniprot/uniprot_TF_ids.tsv', sep='\t', header=None)
tfs = set(tfs[0])
dbds = pd.read_csv('../data/uniprot/05_default_files/uniprot_DBD_AFs.tsv', sep='\t')
dbds = dbds[dbds['ID'].isin(tfs)]
dbds.to_csv('../data/uniprot/05_default_files/uniprot_DBD_AFs_tfs.tsv', sep='\t', index=False)
