import os 
import pandas as pd
import sys
from datetime import datetime

def load_data(data_path: str, sep: str = ';', low_memory: bool = False) -> pd.DataFrame:
    """
    Load the data from a directory or a file, reading header only from the first file
    :param data_path: the path of the data
    :param sep: the separator of the data
    :param low_memory: whether to use low memory mode or not
    :return: the data
    """
    combined_data = pd.DataFrame()

    if os.path.isdir(data_path):
        files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')])
        for idx, file in enumerate(files):
            try:
                if idx == 0:
                    data = pd.read_csv(file, sep=sep, low_memory=low_memory).copy()
                else:
                    data = pd.read_csv(file, sep=sep, low_memory=low_memory, header=None).copy()
                combined_data = pd.concat([combined_data, data], ignore_index=True)
            except Exception as e:
                print(f"Errore durante il caricamento del file {file}: {e}")
    elif os.path.isfile(data_path):
        try:
            combined_data = pd.read_csv(data_path, sep=sep, low_memory=low_memory)
        except Exception as e:
            print(f"Errore durante il caricamento del file {data_path}: {e}")
    else:
        print(f"Il percorso {data_path} non è stato trovato.")
        sys.exit(1)

    return combined_data

def load_uniprot_data(file_path: str, sep: str = '\t') -> pd.DataFrame:
    """
    Carica i dati Uniprot da un file.

    :param file_path: percorso del file Uniprot
    :param sep: separatore per il file Uniprot
    :return: DataFrame con i dati Uniprot
    """
    if not os.path.exists(file_path):
        print(f"Il percorso {file_path} non esiste")
        sys.exit(1)
    try:
        uniprot = pd.read_csv(file_path, sep=sep, dtype='str', low_memory=False)
        return uniprot
    except pd.errors.ParserError as e:
        print(f"Errore di parsing nel file {file_path}: {e}")
    except Exception as e:
        print(f"Errore durante il caricamento del file {file_path}: {e}")
    return pd.DataFrame()

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