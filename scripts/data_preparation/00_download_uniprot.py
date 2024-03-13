import requests
import gzip
import shutil
import os

# download the current release of the UniProtKB/Swiss-Prot database
# URL of the gzipped file
url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
# Local filename to save the gzipped file
if os.path.exists("../../data/uniprot/01_all_uniprot"):
    pass
else:
    os.makedirs("../../data/uniprot/01_all_uniprot")

local_filename = "../../data/uniprot/01_all_uniprot/uniprot_sprot.dat.gz"
# Local filename to save the extracted file
extracted_filename = "../../data/uniprot/01_all_uniprot/uniprot_sprot.dat"

if os.path.exists(extracted_filename):
    print("File already exists.")
    exit()
# Download the file
with requests.get(url, stream=True) as response:
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

# Extract the gzipped file
with gzip.open(local_filename, 'rb') as f_in:
    with open(extracted_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Remove the gzipped file
os.remove(local_filename)

print("File downloaded and extracted successfully.")
   
