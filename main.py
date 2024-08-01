""" 
Main module for target cleaning and preparation
"""
import argparse
import os

import datetime
import time
import sys
import random
import torch
import pandas as pd
import numpy as np


from dataset.preparation import Cleaner
from dataset.processing import process_molecules_and_calculate_descriptors
from utils.args import data_cleaning_args, model_args
from utils.file_utils import load_data

from models.classifiers import train_classifier
from models.regressors import train_regressor


conf_path = os.getcwd()
sys.path.append(conf_path)



def parser_args():
    """
    Parse the arguments
    :return: the arguments
    """
    parser = argparse.ArgumentParser(description = 'Data Cleaning')
    data_cleaning_args(parser)
    parser.add_argument('--path', type = str, default = '/home/luca/LAB/LAB_federica/chembl1865/EGFR.csv',
                        help = 'Specify the path of the data')
    parser.add_argument('--model', type=str, choices=['classifier', 'regressor'], required=True, help='Type of model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    model_args(parser)
    return parser.parse_args()

#per riproducibilità dei risultati con i modelli
def set_random_seed(seed: int) -> None:
    """set the seeds at a certain value
    :param seed: the seed value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        print("Could not set cuda seed.")
        

def main():
    args = parser_args()

    data = load_data(args.path) 

    cleaner = Cleaner(args)
    cleaned_data = cleaner.clean_data(data)

    df=process_molecules_and_calculate_descriptors(cleaned_data)
    
    if args.model_type == 'classifier':
        train_classifier(df,args)
    elif args.model_type == 'regressor':
        train_regressor(df,args)

if __name__ == '__main__':
    main()
