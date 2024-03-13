#!/bin/bash
# make sure to change line 80 and 81 in explain_AI_CV and explain_AI_one_protein for your settings regarding the dataset, default: D1
 
# for a few test proteins in test_seqs.fasta, make sure to change the model you want to use. 
python explain_AI_one_protein.py -i ../data/test_seqs.fasta -s ../data/uniprot/04_final/AFall_spatial.tsv -o ../../results/ -ckpt strucTFactor_model.pt -g cuda:0

# for explainable AI cross validation, change setup within file
python explain_AI_CV.py