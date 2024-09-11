""" 
Main module for target cleaning and preparation
"""
import argparse
import os

import sys
import random
import torch
import numpy as np

from dataset.preparation import Cleaner

from dataset.processing import process_molecules_and_calculate_descriptors
from utils.args import data_cleaning_args, model_args
from utils.file_utils import load_file, process_directory, drop_columns
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
    parser.add_argument('--path_db', type = str, default = '/home/federica/LAB2/chembl33_20240216',
                        help = 'Specify the path of the database')
    parser.add_argument('--model', type=str, choices=['classifier', 'regressor'], required=True, help='Type of model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    model_args(parser)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    set the seeds at a certain value
    :param seed: the seed value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        print("Could not set cuda seed.")
        
def process_data(cleaner, args):
    """
    Workflow for data processing before model training
    :param cleaner: the cleaner instance
    :param args: the arguments
    :return: the processed dataframe
    """
    if os.path.isdir(args.path_db):
        df = process_directory(args.path_db, cleaner)
    else:
        df = load_file(args.path_db)
        df = drop_columns(df)
        df = cleaner.clean_data(df)
    return df

def main():
    args = parser_args()

    cleaner = Cleaner(args)
    
    cleaned_data = process_data(cleaner, args)
    
    df=process_molecules_and_calculate_descriptors(cleaned_data)
    if args.model_type == 'classifier':
        train_classifier(df,args)
    elif args.model_type == 'regressor':
        train_regressor(df,args)

if __name__ == '__main__':
    main()
