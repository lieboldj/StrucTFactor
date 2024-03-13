#!/bin/bash

clust_list=("03" "No")
ratio_list=("norm" "small" "large")
exp_list=("AFfiltered" "AFall")
comm_list=()

# ./../data/uniprot/04_final/AFall_spatial_str.tsv this is your file with all spatial strings

for clust in "${clust_list[@]}"; do
    for ratio in "${ratio_list[@]}"; do
        for exp in "${exp_list[@]}"; do
            comm_list+=("python strucTFactor.py -t -i ./../data/${exp}/${clust}_${ratio}_Com.fasta -s ./../data/uniprot/04_final/AFall_spatial_str.tsv -sm 1 -es 100 -o ./../results/${exp}_clu${clust}_${ratio}_Com -g cuda:0")
            # compare with DeepTFactor, uncomment the following line
            #comm_list+=("python strucTFactor.py -t -i ./../data/${exp}/${clust}_${ratio}_Com.fasta -sm 0 -es 100 -o ./../results/${exp}_clu${clust}_${ratio}_seq_Com -g cuda:0")
        done
    done
done

# Execute the commands
for command in "${comm_list[@]}"; do
    echo "$command"
    eval "$command"
done