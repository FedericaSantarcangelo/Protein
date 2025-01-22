"""Script function to manage data
@Author: Federica Santarcangelo
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.file_utils import competence

# Patterns for finding mutations in the assay description field
aminoacids = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E',
    'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F',
    'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}

patterns = [
    r'\bmutant\s*[A-Z]\d{1,4}\b',
    r'\b[A-Z]\d{1,4}\s*mutant\b',
    r'\b[A-Z]\d{2,4}[A-Z]\b',
    r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}\b',
    r'\b[A-Z]\d{2,4}-[A-Z]\d{2,4}\b',
    r'\bd\d{1,4}-\d{1,4}\b',
    r'\b[A-Z][A-Z]\d{1,4}[A-Z]\s*mutant\b',
    r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del/[A-Z]\d{1,4}[A-Z]\b',
    r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del,?\s*[A-Z]\d{1,4}[A-Z]\b',
    r'\b[A-Z]\d{1,4}/[A-Z]del\b',
    r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del\b',
    r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del,Sins\b',
    r'\([A-Z]\d{1,4}-[A-Z]\d{1,4}\s+ins\s+[A-Z]+\)',
    r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}ins\b',
    r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}\s*ins\b',
    r'[A-Z]\d{1,4}_[A-Z]\d{1,4}ins[A-Z]+',
    r'\b[A-Z]\d{1,4}[-][A-Z]\d{1,4}\s*ins\s*[A-Z]+\b',
    r'\b[A-Z]\d{1,4}ins',
    rf"\b({'|'.join(aminoacids.keys())})(\d+)(?:({'|'.join(aminoacids.keys())}))?\b",
    r'\bDel\s*\d{1,4}\b',
    r'\bdel\d{1,4}\b',
    r'\bex\d{1,2}del\b',
    r'\bexon\s*\d{1,2}\s*deletion\b',
    r'\bd(\d{1,4}-\d{1,4})/[A-Z]\d{1,4}[A-Z]\b',
    r'[A-Z]\d{1,4}[A-Z]/del\s*\(\d{1,4}\s*to\s*\d{1,4}\s*residues\)',
    r'\bdel\s*\(\s*\d{1,4}\s*to\s*\d{1,4}\s*(?:residues?)?\s*\)',
    r'\bdel \s*\d{1,4}-\d{1,4}\b',
    r'FLT3[-\s]?ITD',
    r'\b\(\d{1,4}\s*to\s*\d{1,4}\s*residues\)',
    r'\b\d{1,4}\s*to\s*\d{1,4}\s*deletion/[A-Z]\d{1,4}[A-Z]\b',
    r'\(([A-Z]\d{1,4})-[A-Z]\d{1,4}del(?:,\s*[A-Z]\d{1,4}[A-Z]?)?\)',
    r'\b([A-Z]\d{1,4}-[A-Z]\d{1,4})\s*ins\b',
    r'\(([A-Z]\d{1,4})-[A-Z]\d{1,4}del(?:,\s*(Sins))?\)'
]

conversion_map = {
    'pIC50': 'IC50',
    'pEC50': 'EC50',
    'Log IC50': 'IC50',
    'pKi': 'Ki',
    'Log Ki': 'Ki'
}

def data_perc_f(thr_perc, data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Filter the data perc if are less or greater than the threshold
    :return: the filtered data
    """
    def filter_conditions(row):
        if ('Inhibition' in row['Standard Type'] or 'INH' in row['Standard Type']) and row['Standard Value'] > thr_perc:
            return True
        elif 'Activity' in row['Standard Type'] and row['Standard Value'] < thr_perc:
            return True
        else:
            return False

    filtered_data = data[data.apply(filter_conditions, axis=1)]
    return filtered_data

def data_log_f(log_types, data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Log the data and convert standard types
    :return: the logarithmic data
    """
    for log_type, linear_type in conversion_map.items():
        mask = data['Standard Type'] == log_type
        if log_type in log_types:  
            if 'p' in log_type:
                data.loc[mask, 'Standard Value'] = data.loc[mask, 'Standard Value'].apply(lambda x: round(10**(-float(x)), 3))
            else:
                data.loc[mask, 'Standard Value'] = data.loc[mask, 'Standard Value'].apply(lambda x: round(10**(float(x)), 3))
            
            if (data.loc[mask, 'Standard Units'] == 'mM').any():
                data.loc[mask & (data['Standard Units'] == 'mM'), 'Standard Value'] *= 1000000
            elif (data.loc[mask, 'Standard Units'].isin(['uM', 'µM'])).any():
                data.loc[mask & data['Standard Units'].isin(['uM', 'µM']), 'Standard Value'] *= 1000
            
            data.loc[mask, 'Standard Units'] = 'nM'
            data.loc[mask, 'Standard Type'] = linear_type
    
    return data

def data_act_f(data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Act on the data based on standard units
    :return: the activated data
    """
    data = data.copy()
    if (data['Standard Units'] == 'mM').any():
        data.loc[data['Standard Units'] == 'mM', 'Standard Value'] *= 1000000
        data.loc[data['Standard Units'] == 'mM', 'Standard Units'] = 'nM'
    elif (data['Standard Units'].isin(['uM', 'µM'])).any():
        data.loc[data['Standard Units'].isin(['uM', 'µM']), 'Standard Value'] *= 1000
        data.loc[data['Standard Units'].isin(['uM', 'µM']), 'Standard Units'] = 'nM'

    return data

def other_checks(data: pd.DataFrame, standard_type_act) -> pd.DataFrame:
    """ 
    Other checks on the data columns: Potential Duplicate, Standard Units, Data Validity Comment, pChEMBL Value
    :return: the data with the other checks
    """
    data = data.copy()
    data = data.loc[data['Data Validity Comment'].isnull()]
    data = data.loc[(data['Potential Duplicate'].isnull()) | (data['Potential Duplicate'] == 0)]
    data = data.loc[(data['Standard Units'] == 'nM') | (data['Standard Units'] == '%')]
    
    std_act = standard_type_act[0].split(',')
    mask = data['Standard Type'].isin(std_act) & (data['pChEMBL Value'].isnull())
    
    if not mask.any():
        return data

    data.loc[mask, 'pChEMBL Value'] = -np.log10(data.loc[mask, 'Standard Value']) + 9
    data = data.loc[data['pChEMBL Value'] >= 4]
    return data

def remove_salts(data: pd.DataFrame, assay, standard_type_act) -> pd.DataFrame:
    """ 
    Remove the salts from the SMILES
    :return: the SMILES without salts
    """
    cleaned_smiles = []

    for smiles in data['Smiles']:
        components = smiles.split('.')
        main_component = max(components, key=len)
        cleaned_smiles.append(main_component)
    data['Smiles'] = cleaned_smiles
    data = competence(data, assay)
    data = other_checks(data, standard_type_act)
    return data

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Prepare the data by dropping unnecessary columns and computing log and root squared values
    :return: the prepared data
    """
    df_prepared = df.drop(columns=['Smiles (RDKit Mol)', 'Document ChEMBL ID'])

    if (df_prepared['Standard Value'] <= 0).any():
        raise ValueError("Standard Value with negative value. Cannot compute -log.")
    df_prepared['Log Standard Value'] = -np.log10(df_prepared['Standard Value'])
    df_prepared['Root Squared Standard Value'] = np.sqrt(df_prepared['Standard Value'])

    Q1 = df_prepared['Log Standard Value'].quantile(0.25)
    Q3 = df_prepared['Log Standard Value'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df_prepared['Log Standard Value'] < (Q1 - 1.5 * IQR)) | (df_prepared['Log Standard Value'] > (Q3 + 1.5 * IQR))
    df_prepared = df_prepared[~outliers]
    return df_prepared

def select_optimal_clusters(inertia_scores, silhouette_scores):
    """
    Select the optimal number of clusters
    :return: the optimal number of clusters
    """
    min_length = min(len(inertia_scores), len(silhouette_scores))
    inertia_scores = inertia_scores[:min_length]
    silhouette_scores = silhouette_scores[:min_length]
    silhouette_scaler = MinMaxScaler()
    normalized_silhouette_scores = silhouette_scaler.fit_transform(np.array(silhouette_scores).reshape(-1, 1)).flatten()
    inertia_scaler = MinMaxScaler()
    normalized_inertia_scores = inertia_scaler.fit_transform(np.array(inertia_scores).reshape(-1, 1)).flatten()
    normalized_inertia_scores = 1 - normalized_inertia_scores
    combined_scores = normalized_silhouette_scores + normalized_inertia_scores
    optimal_clusters = combined_scores.argmax() + 2
    return optimal_clusters