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
    parser.add_argument('--model_type', type=str, choices=['classifier', 'regressor'], required=True, help='Type of model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
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

def load_data(data_path):
    """
    Load the data from a directory or a file, reading header only from the first file
    :param data_path: the path of the data
    :return: the data
    """
    if os.path.isdir(data_path):
        files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')])
        comined_data = pd.DataFrame()
        first_file = True

        for file in files:
            if first_file and 'part1' in file:
                data = pd.read_csv(file, sep=';', low_memory=False).copy()
                first_file = False
            else:
                data = pd.read_csv(file, sep=';', low_memory=False, header=None).copy() 
            
            comined_data = pd.concat([comined_data, data], ignore_index=True)
    
    elif os.path.isfile(data_path):
        comined_data = pd.read_csv(data_path, sep=';', low_memory=False).copy()
    
    else:
        print(f"Path {data_path} not found.")
        sys.exit(1)
    
    return comined_data

def main():
    args = parser_args()

    data = load_data(args.path) #make a copy of the data directly during the loading
    #copy = data.copy()

    cleaner = Cleaner(args, data)
    cleaned_data = cleaner.clean_data(data)

    df=process_molecules_and_calculate_descriptors(cleaned_data)
    
    if args.model_type == 'classifier':
        train_classifier(df,args)
    elif args.model_type == 'regressor':
        train_regressor(df,args)

if __name__ == '__main__':
    main()
