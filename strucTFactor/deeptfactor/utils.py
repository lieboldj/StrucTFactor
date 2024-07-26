import re
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import basic python packages
import numpy as np

from Bio import SeqIO

# import torch packages
import torch
import torch.nn as nn

# import scikit learn packages
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score


def argument_parser(version=None):
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', required=False, default="./",
                        help='Output directory')
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cpu', help='Specify gpu')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=128, help='Batch size')
    parser.add_argument('-ckpt', '--checkpoint', required=False, 
                        default="Please add path to checkpoint file via '--ckpt'", help='Checkpoint file needed!')
    parser.add_argument('-i', '--seq_file', required=False, 
                        default='../data/test_seqs.fasta', help='Sequence data')
    parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                        default=1, help='Number of cpus to use')    
    parser.add_argument('-s', '--spatial_file', required=False, default=None, 
                        help='Use spatial information')
    parser.add_argument('-t', '--training', required=False, dest='training', 
                         action='store_true', default=False, help='Self training mode')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=0.001
                        , help='Learning rate')
    parser.add_argument('-sm', '--spatial_mode', required=False, type=int, default=0, 
                        help='Spatial mode: 0 for no spatial, 1 for spatial, 3 for spatial with Filters times 3')
    parser.add_argument('-es', '--early_stop', required=False, type=int, default=5,
                        help='Early stop epoch') 
    parser.add_argument('-e', '--epochs', required=False, type=int, default=50,
                        help='Number of epochs') 
    return parser