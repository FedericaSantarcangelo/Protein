import os
import pandas as pd
import sys
import argparse
from dataset.preparation import Cleaner 
from utils.args import data_cleaning_args, mutation
from dataset.processing import process_molecules_and_calculate_descriptors

def parser_args():
    parser = argparse.ArgumentParser(description = 'Data Cleaning')
    data_cleaning_args(parser)
    #mutation(parser)
    parser.add_argument('--path', type = str, default = '/home/federica/chembl1865/chembl1865.csv', help = 'Specify the path of the data')
    print(parser.parse_args())
    return parser.parse_args()


def load_data(data_path):
    if not os.path.exists(data_path):
        print(f"Path {data_path} does not exist")
        sys.exit(1)
    else:
         try:
            data = pd.read_csv(data_path, sep=';')
            return data
         except pd.errors.ParserError as e:
             print(f"ParserError: {e}")
    return None

def main():
    args = parser_args()

    data = load_data(args.path)
    copy = data.copy()

    cleaner = Cleaner(args, copy) # Cleaner class is defined in dataset/preparation.py for data cleaning
    cleaned_data = cleaner.clean_data(copy) # clean_data method is defined in Cleaner class to call all the data cleaning methods

    #df=process_molecules_and_calculate_descriptors(cleaned_data) # process_molecule_with_logging is defined in dataset/processing.py to process the molecules and calculate the descriptors
    #print(df)

if __name__ == '__main__':
    main()

