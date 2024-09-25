import pandas as pd
import numpy as np
from utils.file_utils import compentence

def data_perc_f(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Filter the data perc if are less or greater than the threshold
            :return: the filtered data
        """
        def filter_conditions(row):
            if row['Standard Type'] == 'Inhibition' and row['Standard Value'] > self.args.thr_perc:
                return True
            elif row['Standard Type'] == 'Activity' and row['Standard Value'] < self.args.thr_perc:
                return True
            else:
                return False
        filtered_data = data[data.apply(filter_conditions, axis=1)]
        return filtered_data

def data_log_f(self, data: pd.DataFrame) -> pd.DataFrame:
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
        for log_type in self.args.standard_type_log:
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

def remove_salts(self, data: pd.DataFrame) -> pd.DataFrame:
    """ Remove the salts from the SMILES
        :return: the SMILES without salts
    """
    cleaned_smiles = []

    for smiles in data['Smiles']:
        components = smiles.split('.')
        main_component = max(components, key=len)
        cleaned_smiles.append(main_component)
    data['Smiles'] = cleaned_smiles
    data = compentence(data,self.assay)
    return data