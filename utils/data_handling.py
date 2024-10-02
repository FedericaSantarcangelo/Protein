"""script function to manage data"""

import pandas as pd
import numpy as np
from utils.file_utils import compentence

#patterns for mutation
patterns = [
            r'\b[A-Z]\d{1,4}[A-Z]\b',  # Mutazione singola, e.g., L747S
            r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}\b',  # Mutazione tra due amminoacidi, e.g., A763_Y764
            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}\b',  # Mutazione tra due amminoacidi con trattino, e.g., D770-N771
            r'\b[A-Z]\d{1,4}\b',  # Mutazione singola, e.g., L747

            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del/[A-Z]\d{1,4}[A-Z]\b',  # Delezione tra due amminoacidi con separatore di barra, e.g., E746-A750del/L858R
            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del, [A-Z]\d{1,4}[A-Z]\b',  # Delezione tra due amminoacidi con separatore di barra, e.g., E746-A750del,L858R

            r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}insFHEA',  # Delezione tra due amminoacidi, e.g., A763_Y764insFHEA

            r'\b[A-Z]\d{1,4}/[A-Z]del\b',  # Mutazione doppia, e.g., L747S/T751del
            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del\b',  # Mutazione d'intervallo, e.g., L747-T751del
            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del,Sins\b',  # Mutazione d'intervallo con inserzione, e.g., L747-T751del,Sins
            r'\bDel [A-Z]\d{1,4}/[A-Z]\d{1,4}\b',  # Delezione tra due amminoacidi con separatore di barra, e.g., Del E746/A750
            r'\bdel \d{1,4}-\d{1,4}\b',  # Delezione con intervallo numerico, e.g., del 746-750
            r'\bDel\s*\d{1,4}\b',  # Delezione, e.g., Del19
            
            r'\bex\d{1,2}del\b',  # Delezione con notazione esone, e.g., ex19del
            r'\bexon\d{1,2} deletion\b',  # Delezione con notazione esone, e.g., exon19 deletion
            r'\bexon \d{1,2} deletion\b',  # Delezione con notazione esone, e.g., exon19 deletion
            
            r'\b del \(\d{1,4} to \d{1,4}\)\b',  # Delezione con intervallo numerico tra parentesi, e.g., del (746 to 750)
            r'\b \d{1,4} to \d{1,4}\s* deletion\b',  # Delezione con intervallo numerico tra parentesi, e.g., 746 to 750 deletion

            r'\bd(\d{1,4}-\d{1,4})\/([A-Z]\d{1,4}[A-Z])\b',  # Delezione con intervallo numerico e mutazione, e.g., d746-750/L858R

            r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}\s*ins\b',  # Mutazione di inserzione, e.g., D770-N771ins
            r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}\s*ins\b',  # Inserzione tra due amminoacidi, e.g., A763_Y764ins
        ]


def data_perc_f(thr_perc, data: pd.DataFrame) -> pd.DataFrame:
        """ Filter the data perc if are less or greater than the threshold
            :return: the filtered data
        """
        def filter_conditions(row):
            if row['Standard Type'] == 'Inhibition' and row['Standard Value'] > thr_perc:
                return True
            elif row['Standard Type'] == 'Activity' and row['Standard Value'] < thr_perc:
                return True
            else:
                return False
        filtered_data = data[data.apply(filter_conditions, axis=1)]
        return filtered_data

def data_log_f(standard_type_log ,data: pd.DataFrame) -> pd.DataFrame:
        """ Log the data and convert standard types
            :return: the logarithmic data
        """
        if (data['Standard Units'] == 'mM').any():
            data.loc[data['Standard Units'] == 'mM', 'Standard Value'] = [
                round(np.log(np.exp(float(jj)) * 1000000), 3) for jj in data.loc[data['Standard Units'] == 'mM', 'Standard Value'].values.tolist()
            ]
            data.loc[data['Standard Units'] == 'mM', 'Standard Units'] = 'nM'
        elif (data['Standard Units'].isin(['uM', 'µM'])).any():
            data.loc[data['Standard Units'].isin(['uM', 'µM']), 'Standard Value'] = [
                round(np.log(np.exp(float(jj)) * 1000), 3) for jj in data.loc[data['Standard Units'].isin(['uM', 'µM']), 'Standard Value'].values.tolist()
            ]
            data.loc[data['Standard Units'].isin(['uM', 'µM']), 'Standard Units'] = 'nM'
        for log_type in standard_type_log:
            standard_type = log_type.replace('p', '')
            data.loc[data['Standard Type'] == log_type, 'Standard Type'] = standard_type

        return data

def data_act_f(data: pd.DataFrame) -> pd.DataFrame:
        """ Act on the data based on standard units
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

def remove_salts( data: pd.DataFrame, assay) -> pd.DataFrame:
    """ Remove the salts from the SMILES
        :return: the SMILES without salts
    """
    cleaned_smiles = []

    for smiles in data['Smiles']:
        components = smiles.split('.')
        main_component = max(components, key=len)
        cleaned_smiles.append(main_component)
    data['Smiles'] = cleaned_smiles
    data = compentence(data, assay)
    return data

import requests
import pandas as pd
import time
import json
import xml.etree.ElementTree as ET

def enrich_dataframe_with_protein_classifications(chembl_df):
    """
        Add a new column 'Protein Classification ID' to the DataFrame with ChEMBL data.
        Returns: pd.DataFrame: DataFrame with 'Protein Classification ID'.
    """
    target_ids = chembl_df['Target ChEMBL ID'].unique()

    def get_protein_classification(target_id):
        """
        This function performs a GET request to the ChEMBL API to retrieve the protein classification for a given target ID.
        Returns: str: Protein classification ID.
        """
        target_url = f"https://www.ebi.ac.uk/chembl/api/data/protein_classification/{target_id}"
        response = requests.get(target_url)

        if response.status_code == 200:
            try:
                # Verifica se la risposta è XML
                if response.headers['Content-Type'].startswith('application/xml') or response.headers['Content-Type'].startswith('text/xml'):
                    root = ET.fromstring(response.content)
                    protein_classifications = root.findall('.//protein_classification')
                    for classification in protein_classifications:
                        pref_name = classification.find('pref_name')
                        if pref_name is not None and pref_name.text and pref_name.text != 'Protein class':
                            return pref_name.text
                    print(f"Nessuna classificazione proteica trovata per il target ID {target_id}")
                    return None
                    
                if response.text.strip():
                    data = response.json()
                    if 'protein_classification' in data and 'protein_class_id' in data['protein_classification']:
                        return data['protein_classification']['protein_class_id']
                    else:
                        return None
                else:
                     print(f"No data found for target ID {target_id}")
                     return None 
            except json.JSONDecodeError:
                print(f"Errore nel decodificare la risposta JSON per il target ID {target_id}")
                return None
        else:
            print(f"Errore nel recuperare il target {target_id}: {response.status_code}")
            return None
        
    protein_classifications = {}
    for target_id in target_ids:
        classification_id = get_protein_classification(target_id)
        protein_classifications[target_id] = classification_id
        time.sleep(1)  #delay to avoid too many requests

        
    chembl_df['Protein Classification'] = chembl_df['Target ChEMBL ID'].map(protein_classifications)

    return chembl_df

