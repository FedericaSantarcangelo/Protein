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
from utils.args import data_cleaning_args


conf_path = os.getcwd()
sys.path.append(conf_path)



def parser_args():
    """
    Parse the arguments
    :return: the arguments
    """
    parser = argparse.ArgumentParser(description = 'Data Cleaning')
    data_cleaning_args(parser)
    #mutation(parser)
    parser.add_argument('--path', type = str, default = '/home/luca/LAB/LAB_federica/chembl1865/EGFR.csv',
                        help = 'Specify the path of the data')
    print(parser.parse_args())
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
    Load the data from the path
    :param data_path: the path of the data
    :return: the data
    """
    if not os.path.exists(data_path):
        print(f"Path {data_path} does not exist")
        sys.exit(1)
    else:
        try:
            data = pd.read_csv(data_path, sep=';', low_memory=False)
            return data
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
    return None

def main():
    args = parser_args()

    data = load_data(args.path)
    copy = data.copy()

    cleaner = Cleaner(args, copy)
    cleaned_data = cleaner.clean_data(copy)

    df=process_molecules_and_calculate_descriptors(cleaned_data)
    print(df)

if __name__ == '__main__':
    main()
