#!/bin/bash
echo "Download Uniprot data and prepare it for the analysis."
python 00_download_uniprot.py

echo "Extract the IDs from the .dat file."
python 01_get_IDs_from_dat.py

echo "Remove duplicate IDs in lists."
python 02_remove_duplicate_IDs.py

echo "Get the fasta sequences for the IDs."
python 03_get_fasta_seqs_dat.py

echo "Filter the fasta sequences based on our criteria."
python 04_filter_fasta_seq.py

echo "Get the spatial strings for the fasta sequences running DSSP."
python 05_get_af_spatial.py

echo "Copy the fasta sequences to the final directory."
scp ../../data/uniprot/05_default_files/AFall_seqs.fasta ../../data/uniprot/04_final/AFall_seqs.fasta

echo "Get the certainty of all our AlphaFold predictions."
python 06a_get_alphafold_certainty.py

echo "Create fasta files for each experiment setup."
python 06b_fasta_per_setup.py
cd ../../data/

echo "Cluster the sequences using MMseqs2."
clust_list=("03")
exp_setup=("AFall" "AFfiltered")
comm_list=()
# create directories if not exits
mkdir -p tmp_files
for exp in "${exp_setup[@]}"; do
    mkdir -p $exp
done

for clust in "${clust_list[@]}"; do
    for exp in "${exp_setup[@]}"; do
        if [ "$clust" == "03" ]; then
            comm_list+=("mmseqs easy-cluster uniprot/04_final/${exp}_seqs.fasta ${exp}/${clust} tmp_files/ --min-seq-id 0.3 -c 0.8 --cov-mode 1")
        fi
    done
done

# Execute the commands
for command in "${comm_list[@]}"; do
    echo "$command"
    eval "$command"
done

# remove tmp_files
rm -r tmp_files

echo "Create the fasta files for the different clustering ratios."
cd ../scripts/data_preparation/

ratio_list=("norm" "small" "large")
exp_setup=("AFfiltered" "AFall")

comm_list=()

for ratio in "${ratio_list[@]}"; do
    for exp in "${exp_setup[@]}"; do
        comm_list+=("python 08a_sampling_post_clustering.py ${exp} ${ratio}")
        comm_list+=("python 08b_sampling_no_clust.py ${exp} ${ratio}")
    done
done

# Execute the commands
for command in "${comm_list[@]}"; do
    echo "$command"
    eval "$command"
done

echo "Extract the domains from the fasta sequences to run explainable AI."
python 09_explainAI_extract_domains.py