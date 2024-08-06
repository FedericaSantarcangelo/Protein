import os 
import pandas as pd
import sys
from datetime import datetime

def detect_delimiter(path: str, num_lines = 5) -> str:
    """
    Detect the delimiter of the file by reading the first num_lines of the file
    :param path: the path of the file
    :param num_lines: the number of lines to read
    :return: the delimiter of the file
    """
    delimiters = [';', ',', '\t', '|']
    with open(path, 'r') as f:
        lines = [f.readline() for _ in range(num_lines)]
    for delimiter in delimiters:
        if all([delimiter in line for line in lines]):
            return delimiter
    return '\t' # Assuming \t as default delimiter if no delimiter is found

def load_file(path : str, delimiter = None, header='infer'):
    """
    Main function to load the data of any kind like csv, tsv, excel
    :param path: the path of the file 
    :param delimiter: the delimiter of the file
    :param header: the header of the file
    """
    try: 
        if delimiter is None:
            delimiter = detect_delimiter(path)
        
        df = pd.read_csv(path, delimiter = delimiter, header = header, low_memory=False)
        return df
    except Exception as e:
        print(f"Error during the loading of the file {path}: {e}")
        sys.exit(1)

def process_directory(path: str, cleaner):
    """"
    Process the directory containing the files to be cleaned: for each file in the directory,
    apply the cleaning function and save the cleaned file
    :param path: the path of the directory
    :param cleaner: the cleaner instance
    return: the cleaned dataframe
    """
    from dataset.preparation import Cleaner

    return

def save_other_files(file: pd.DataFrame, output_path: str, name: str):
    """
    Save the file different from mutation such mixed and semi_sintetic_data
    :param file: the file to be saved
    :param output_path: the output path
    :param name: the name of the file
    """

    full_path = os.path.join(output_path + 'other', name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    ext = '.csv'
    if os.path.exists(full_path):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        full_path = f"{full_path}/{name}{timestamp}{ext}"

    try:
        file.to_csv(full_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error during the saving of the file {full_path}: {e}")



def save_data_report(base_path: str, data_dict: dict):
    """
    Save the data in the report folder or in the filtered folder if the file is not a report
    :param base_path: the base path
    :param data_dict: the data dictionary
    """
    dataset_path = os.path.join(base_path, 'data', 'filtered')
    report_path = os.path.join(base_path, 'data', 'report')

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    for filename, df in data_dict.items():
        if 'report' in filename:
            full_path = os.path.join(report_path, filename)
        else:
            full_path = os.path.join(dataset_path, filename)

        if os.path.exists(full_path):
            base, ext = os.path.splitext(full_path)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            full_path = f"{base}_{timestamp}{ext}"

        try:
            df.to_csv(full_path, index=False, encoding='utf-8')
        except Exception as e:
            print(f"Errore durante il salvataggio del file {full_path}: {e}")