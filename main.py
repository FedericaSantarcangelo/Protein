""" 
Main module for target cleaning and preparation

@Author: Federica Santarcangelo
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from dataset.preparation import Cleaner
from models.pca_tsne import DimensionalityReducer
from models.utils import bin_random_split, check_train_test_similarity
from dataset.processing import process_molecules_and_calculate_descriptors
from utils.data_handling import prepare_data
from utils.args import data_cleaning_args, file_args, reducer_args, qsar_args
from utils.file_utils import load_file, process_directory, drop_columns, add_protein_family, calculate_similarity_scores

conf_path = os.getcwd()
sys.path.append(conf_path)

def parser_args() -> argparse.Namespace:
    """
    Parse the arguments
    :return: the arguments
    """
    parser = argparse.ArgumentParser(description='Data Cleaning')
    file_args(parser)
    data_cleaning_args(parser)
    parser.add_argument('--path_db', type=str, default='/home/federica/LAB2/chembl35/chembl35_20250219',
                        help='Specify the path of the database')
    reducer_args(parser)
    parser.add_argument('--qsar_pilot', action='store_true', help='Run QSAR Pilot analysis with predefined molecules')
    parser.add_argument('--input_file', type=str, help='Path to the input file with precomputed descriptors')
    qsar_args(parser)
    return parser.parse_args()

def process_data(cleaner, args) -> pd.DataFrame:
    """
    Workflow for data processing before model training
    :return: the processed dataframe
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    name = 'data_' + timestamp + '/'
    args.path_output = os.path.join(args.path_output, name)
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
    if os.path.isdir(args.path_db):
        df = process_directory(args.path_db, cleaner)
    else:
        df = load_file(args.path_db)
        df = drop_columns(df)
        df = add_protein_family(df, args.path_proteinfamily)
        df = cleaner.clean_data(df)
    return df

def run_qsar_pilot(input_file, args) -> pd.DataFrame:
    """
    Run the QSAR pilot study
    """
    df_x = pd.read_csv(input_file)
    df_y = pd.read_csv("/home/luca/LAB/LAB_federica/chembl1865/egfr_qsar/41molecule.csv")
    df_y = calculate_similarity_scores(df_y)
    
    df_y = process_molecules_and_calculate_descriptors(df_y)
    df_y = prepare_data(df_y)

    df_x = process_molecules_and_calculate_descriptors(df_x)
    df_x = prepare_data(df_x)

    df_x,df_y=bin_random_split(df_x,df_y)
    check_train_test_similarity(df_x, df_y)
    df_x.to_csv("/home/luca/LAB/LAB_federica/chembl1865/egfr_qsar/x.csv", index=False)
    df_y.to_csv("/home/luca/LAB/LAB_federica/chembl1865/egfr_qsar/y.csv", index=False)
    
    numerical_data_y = df_y.select_dtypes(include=[np.number])
    numerical_data_y = numerical_data_y.drop(columns=['Standard Value', 'Log Standard Value'])
    numerical_data_y = numerical_data_y.fillna(0)
    numerical_data_x = df_x.select_dtypes(include=[np.number])
    numerical_data_x = numerical_data_x.fillna(0)
    numerical_data_x = numerical_data_x.drop(columns=['Standard Value', 'Log Standard Value'])

    reducer = DimensionalityReducer(args)
    reducer.fit_transform(numerical_data_x, df_x['Log Standard Value'], numerical_data_y,df_y['Log Standard Value'])
    return 1

def main():
    args = parser_args()
    if args.qsar_pilot:
        if not args.input_file:
            print("Error: --qsar_pilot requires --input_file to be specified.")
            return
        df = run_qsar_pilot(args.input_file, args)
    else:
        cleaner = Cleaner(args)
        cleaned_data = process_data(cleaner, args)

if __name__ == "__main__":
    main()