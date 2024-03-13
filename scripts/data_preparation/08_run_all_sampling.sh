#!/bin/bash
#### IMPORTANT ######
# run file from here
#####################
ratio_list=("norm" "small" "large")
exp_setup=("AFfiltered" "AFall")

# "04" "05")
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
