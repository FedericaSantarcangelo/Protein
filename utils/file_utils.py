"""script to manage files and data"""
import os 
import glob
import pandas as pd
import sys

def detect_delimiter(path: str, num_lines = 5) -> str:
    """
    Detect the delimiter of the file by reading the first num_lines of the file
    :return: the delimiter of the file
    """
    delimiters = [';', ',', '\t', '|']
    with open(path, 'r') as f:
        lines = [f.readline() for _ in range(num_lines)]
    for delimiter in delimiters:
        if all([delimiter in line for line in lines]):
            return delimiter
    return '\t'

def load_file(path : str, delimiter = None, header='infer'):
    """
    Main function to load the data of any kind like csv, tsv, excel
    """
    try: 
        if delimiter is None:
            delimiter = detect_delimiter(path)
        
        df = pd.read_csv(path, delimiter = delimiter, header = header, low_memory=False)
        return df
    except Exception as e:
        print(f"Error during the loading of the file {path}: {e}")
        sys.exit(1)

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the columns from the dataframe
    :return: the dataframe without the columns
    """
    return df.drop(columns=['Molecule Name','Molecular Weight','#RO5 Violations','AlogP','pChEMBL Value',
                            'Data Validity Comment','Comment','Uo Units','Ligand Efficiency BEI',
                            'Ligand Efficiency LE','Ligand Efficiency LLE','Ligand Efficiency SEI',
                            'Potential Duplicate','BAO Format ID','Assay Tissue ChEMBL ID','Assay Tissue Name',
                            'Assay Subcellular Fraction','Assay Parameters','Assay Variant Accession','Source ID',
                            'Document Journal','Document Year','Properties','Properties','Action Type'])

def process_directory(path: str, cleaner):
    """"
    Process the directory containing the files to be cleaned: for each file in the directory,
    apply the cleaning function 
    return: the cleaned dataframe
    """
    if not path.endswith('/'):
        path = path + '/'
    files = sorted(glob.glob(os.path.join(path, '*.csv')))
    header = None
    cleaned_dfs = []

    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"File {file_path} not found")
            continue
        if 'part' not in file_path:
            df = load_file(file_path)
            header = df.columns
        else:
            if header is None:
                raise ValueError("Header file not found. Ensure that the first file contains the header.")
            df = load_file(file_path,header=None)
            df.columns = header
        df = drop_columns(df)
        cleaned_dfs.append(cleaner.clean_data(df))
    cleaned_df = pd.concat(cleaned_dfs, ignore_index=True)    
    return cleaned_df

def compentence(data: pd.DataFrame, assay: pd.DataFrame) -> pd.DataFrame:
    """Filter data based on the confidence score in the assays file
            :return: the filtered data
    """
    data = data.copy()
    assay_ids = assay[assay['Confidence Score'] >= 8]['ChEMBL ID'].tolist()
    data = data[data['Assay ChEMBL ID'].isin(assay_ids)]
    data = data[data['Assay Type'].isin(['F', 'B'])]
    return data

def split_second(second: pd.DataFrame):
    """
    Split the second quality data in two different dataframes: one with assay type B and Bao label single protein or assay format
    and the other with assay type B or F and Bao label cell based if there are duplicates preference is given to B
    """
    df1 = second[(second['Assay Type'] == 'B') &
                 (second['BAO Label'].isin(['single protein format','assay format']))].copy()
    df2 = second.drop(df1.index)
    df2 = df2[(df2['Assay Type'] == 'B') | (df2['Assay Type'] == 'F') & (df2['BAO Label'] == 'cell-based format')].copy()
    return df1, df2

def save_other_files(file: pd.DataFrame, output_path: str,name: str, flag: str = '1'):
    """
    Save the file different from mutation such mixed 
    """
    full_path = os.path.join(output_path+name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    full_path = os.path.join(full_path+ f'/{name}' +f'_{flag}')  
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if os.path.exists(full_path):
        try:
            group = file.groupby('Target ChEMBL ID')
            for name, df in group:
                output_file = os.path.join(full_path, f"{name}_{flag}.csv")
                if os.path.exists(output_file):
                    try:
                        existing_data = pd.read_csv(output_file)
                        df = pd.concat([existing_data, df], ignore_index=True).drop_duplicates()
                    except Exception as e:
                        print(f"Error during the reading of the file {output_file}: {e}")
                try:
                    df.to_csv(output_file, index=False)
                except Exception as e:
                    print(f"Error during the saving of the file {output_file}: {e}")
        except Exception as e:
            print(f"Error during the saving of the file {full_path}: {e}")

def save_data_report(base_path: str, data_dict: dict):
    """
    Save the data in the report folder or in the filtered folder if the file is not a report
    """
    data_path = os.path.join(base_path, 'filtered')
    report_path = os.path.join(base_path, 'report')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(report_path):
        os.makedirs(report_path)    
    
    for filename, df in data_dict.items():
        if 'report' in filename:
            full_path = os.path.join(report_path, filename)
        else:
            full_path = os.path.join(data_path, filename)

        if os.path.exists(full_path):
            try:
                existing_data = pd.read_csv(full_path, low_memory=False)
                df = pd.concat([existing_data, df], ignore_index=True).drop_duplicates()
            except Exception as e:
                print(f"Error during the reading of the file {full_path}: {e}")
        try:
            df.to_csv(full_path, index=False, encoding='utf-8')
        except Exception as e:
            print(f"Errore durante il salvataggio del file {full_path}: {e}")


