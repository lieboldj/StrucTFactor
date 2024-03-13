#!/bin/bash
#### IMPORTANT ######
# run file from data directory
#####################

# ./../scripts/data_preparation/07_run_mmseq_for_cluster.sh
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
