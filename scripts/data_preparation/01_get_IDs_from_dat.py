import sys
from tqdm import tqdm

def seperatebyTF(filename):
    print("Start Readin...")
    # open filename and read line by line
    with open(filename, 'r') as f:
        dat_lines = f.readlines()
    print("Readin Done...")
    TF = False
    DNA = False
    Reg = False
    AC = False
    PDB = False
    PDB_counter = 0
    isTF = list()
    isnoTF = list()
    isnoAC = list()
    uniprot_id = None
    # create lists
    TF_GO =  ["GO:0000976", "GO:0000977", "GO:0000978", "GO:0000979", "GO:0000981", "GO:0000984",
               "GO:0000985", "GO:0000986", "GO:0000987", "GO:0000992", "GO:0000995", "GO:0001046",
               "GO:0001163", "GO:0001164", "GO:0001165", "GO:0001216", "GO:0001227", "GO:0003700",
               "GO:0034246", "GO:0098531", "GO:0106250"]
    TReg_GO = ["GO:0001228", "GO:0006351", "GO:0006355", "GO:0043433", "GO:0045892", "GO:0045893", 
                "GO:0051090", "GO:0051091", "GO:2000142", "GO:2000143", "GO:2000144"]
    DNA_GO = ["GO:0003677", "GO:0008301", "GO:0043565", "GO:0050692"]
    TF_GOnum = 0
    TReg_GOnum = 0
    DNA_GOnum = 0
    combi_num = 0
    #read line by line 
    for line in tqdm(dat_lines):
        if line.startswith('ID   '):
            if AC and PDB:
                PDB_counter += 1
            if TF and AC:
                isTF.append(uniprot_id)
            elif uniprot_id != None:
                if AC:
                    isnoTF.append(uniprot_id)
                else:
                    if TF:
                        uniprot_id = "!" + uniprot_id
                    isnoAC.append(uniprot_id)
            TF = False
            DNA = False
            Reg = False
            AC = False
            PDB = False
        if line.startswith('AC   '):
            uniprot_id = line.split()[1].rstrip(';')
            AC = True
        if line.startswith('DR   GO;') and not TF:
            GO = line.split(';')[1]
            GO = GO.strip()
            if GO in TF_GO:
                TF_GOnum += 1
                TF = True
            elif GO in DNA_GO:
                DNA_GOnum += 1
                DNA = True
            elif GO in TReg_GO:
                TReg_GOnum += 1
                Reg = True
            if DNA and Reg:
                combi_num += 1
                TF = True
        if line.startswith('DR   PDB'):
            PDB = True
    if TF and AC:
        isTF.append(uniprot_id)
    else:
        if AC:
            isnoTF.append(uniprot_id)
        else:
            if TF:
                uniprot_id = "!" + uniprot_id
            isnoAC.append(uniprot_id)

    # turn isTF and isnoTF into a tsv file
    with open('../../data/uniprot/01_all_uniprot/uniprot_TF_ids.tsv', 'w') as f:
        #f.write("uniprotIDs of TranscriptionFactors\n")
        for item in isTF:
            f.write("%s\n" % item)
    with open('../../data/uniprot/01_all_uniprot/uniprot_nonTF_ids.tsv', 'w') as f:
        #f.write("uniprotIDs of non TranscriptionFactors\n")
        for item in isnoTF:
            f.write("%s\n" % item)

    print("TF_GOnum: ", TF_GOnum)
    print("TReg_GOnum: ", TReg_GOnum)
    print("DNA_GOnum: ", DNA_GOnum)
    print("combi_num: ", combi_num)
    print(PDB_counter)
if __name__ == '__main__':
    # uniprot_sprot.data
    filename = "../../data/uniprot/01_all_uniprot/uniprot_sprot.dat"
    seperatebyTF(filename)

