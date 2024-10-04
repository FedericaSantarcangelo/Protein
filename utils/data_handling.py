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

