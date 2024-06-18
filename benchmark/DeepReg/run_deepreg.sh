#!/bin/bash

# Arrays to hold file names and fold numbers
files_all=("AFall/03_norm_Com.fasta" "AFall/03_small_Com.fasta" "AFall/03_large_Com.fasta")
files_filtered=("AFfiltered/03_norm_Com.fasta" "AFfiltered/03_small_Com.fasta" "AFfiltered/03_large_Com.fasta")
folds=(0 1 2 3 4)

# Loop over each file and fold combination for the GPU of your choice
for file in "${files_all[@]}"; do
  for fold in "${folds[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python ./train_model.py "$file" 1 "$fold"
  done
done

# Loop over each file and fold combination for the GPU of your choice
for file in "${files_filtered[@]}"; do
  for fold in "${folds[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python ./train_model.py "$file" 1 "$fold"
  done
done
