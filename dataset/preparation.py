import pandas as pd
import os
import numpy as np
from argparse import Namespace
from utils.args import *
import re

def get_parser() -> ArgumentParser:
    """ Get the parser """
    parser = ArgumentParser()
    data_cleaning_args(parser)
    return parser

class Cleaner():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Clean the data
            :param data: the data
            :return: the cleaned data
        """
        data = self.remove_row(data)
        data = self.filter_data(data)
        data = self.remove_salts(data)
        data = self.remove_duplicate(data)
        
        data_report, whole_dataset, whole_act, whole_inact, inc_data = self.active_inactive(data)
        directory_path = os.path.dirname(self.args.path)
        self.save_data_report(directory_path, data_report, whole_dataset, whole_act, whole_inact, inc_data)
        
        return data

    def remove_row(self,data: pd.DataFrame):
        """ Remove the row with missing values
            :param data: the data
            :return: the data without missing values
        """
        
        data = data.dropna(subset=['Smiles',
                            'Standard Type',
                            'Standard Relation',
                            'Standard Value',
                            'Standard Units'])
        
        data = data.loc[data['Standard Value'] > 0]

        """ Remove the rows with no interesting values """
        # Filter based on the parser arguments
        if self.args.assay_type != 'None':
            data = data.loc[data['Assay Type'] == self.args.assay_type]
        if self.args.assay_organism != 'None':
            data = data.loc[data['Assay Organism'] == self.args.assay_organism]
        if self.args.BAO_Label != 'None':  
            data = data.loc[data['BAO Label'] == self.args.BAO_Label]
        if self.args.target_type != 'None':
            data = data.loc[data['Target Type'] == self.args.target_type]

        return data

    def filter_data(self, data: pd.DataFrame):
        """ Filter the data based on the standard type
            :param data: the data
            :return: the filtered data
        """
        if self.args.standard_type_log != 'None':
            l_type = self.args.standard_type_log[0].split(',')
            data_log = data[data['Standard Type'].isin(l_type)]
            data_log = self.data_log(data_log)

        if self.args.standard_type_act != 'None':
            s_type = self.args.standard_type_act[0].split(',')
            data_act = data[data['Standard Type'].isin(s_type)]
            data_act = self.data_act(data_act)

        if self.args.standard_type_perc != 'None' and self.args.assay_description_perc != 'None':
            p_type = self.args.standard_type_perc[0].split(',')
            assay_desc = self.args.assay_description_perc[0].split(',')
            data_perc = data[data['Standard Type'].isin(p_type)]
            pattern = '|'.join(map(re.escape, assay_desc))
        
        # Filtra i dati in base all'espressione regolare
            data_perc = data_perc[data_perc['Assay Description'].str.contains(pattern, regex=True, na=False)]
            data_perc = self.data_perc(data_perc)
        
        return pd.concat([data_log, data_act, data_perc])
    
    def data_perc(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Filter the data perc if are less or greater than the threshold
            :param data: the data
            :return: the filtered data
        """
        def filter_conditions(row):
            if row['Standard Type'] == 'Inhibition' and row['Standard Value'] > self.args.thr_perc:
                return True
            elif row['Standard Type'] == 'Activity' and row['Standard Value'] < self.args.thr_perc:
                return True
            else:
                return False

    # Applica la funzione a ogni riga e filtra il DataFrame
        filtered_data = data[data.apply(filter_conditions, axis=1)]
        return filtered_data
           

    def data_log(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Log the data and convert standard types
            :param data: the data
            :return: the logarithmic data
        """
        # Logarithmic transformation based on 'Standard Units'
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

        # Convert standard types for logarithmic values
        for log_type in self.args.standard_type_log:
            standard_type = log_type.replace('p', '')
            data.loc[data['Standard Type'] == log_type, 'Standard Type'] = standard_type

        return data
    
    def data_act(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Act on the data based on standard units
            :param data: the data
            :return: the activated data
        """
        # Copy the DataFrame to avoid modifying the original data
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
            :param smiles: the SMILES
            :return: the SMILES without salts
        """
         # Crea una nuova lista per i SMILES senza sali
        cleaned_smiles = []
    
    # Itera su ogni SMILES nel DataFrame
        for smiles in data['Smiles']:
        # Divide il SMILES sui punti per separare i componenti
            components = smiles.split('.')
        # Trova il componente più lungo
            main_component = max(components, key=len)
        # Aggiunge il componente principale alla lista
            cleaned_smiles.append(main_component)
    
    # Sostituisce la colonna 'Smiles' con i SMILES puliti
        data['Smiles'] = cleaned_smiles
        return data
    
    ####################################################DA GENERALIZZARE####################################################
    
    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Remove the duplicates from the data based on the 'Molecule ChEMBL ID'
            :param data: the data
            :return: the data without duplicates
        """
        # Identifica i duplicati basandoti su 'Molecule ChEMBL ID'
        duplicates = data.duplicated(subset='Molecule ChEMBL ID', keep=False)

    # Filtra il DataFrame per mantenere solo i duplicati
        duplicate_data = data[duplicates]

    # Ordina i duplicati in base ai criteri di scelta
    # e altri criteri che desideri utilizzare
        rel_pri={"'='":1,"'=<'":2,"'>='":3,"'>'":4,"'<'":5} #relation priority
        #rel_pri=dict(zip(self.args.rel_priority[::2], map(int, self.args.rel_priority[1::2])))

        sty_pri={"IC50":1,"Ki":2,"Kd":3,"EC50":4,"Inhibition":5,"Activity":6} #standard type priority
        #sty_pri=dict(zip(self.args.sty_priority[::2], map(int, self.args.sty_priority[1::2])))
        src_pri={"Scientific Literature":1, "BindingDB Database":2,"Fraunhofer HDAC6":3} #source description priority
        # Create a temporary DataFrame to calculate composite sort keys
        temp_df = duplicate_data.copy()
        temp_df['src_sort_key'] = temp_df['Source Description'].map(src_pri)
        temp_df['sty_sort_key'] = temp_df['Standard Type'].map(sty_pri)
        temp_df['rel_sort_key'] = temp_df['Standard Relation'].map(rel_pri)
        # Sort the temporary DataFrame using the calculated sort keys
        temp_df_sorted = temp_df.sort_values(by=['src_sort_key', 'sty_sort_key', 'rel_sort_key'], ascending=[True, False, False])

    # drop the temporary sort key columns if you want
        duplicate_data_sorted = temp_df_sorted.drop(columns=['src_sort_key', 'sty_sort_key', 'rel_sort_key'])

    # Rimuovi i duplicati, mantenendo la prima occorrenza dopo l'ordinamento
        data_without_duplicates = duplicate_data_sorted.drop_duplicates(subset='Molecule ChEMBL ID', keep='first')

    #  dati non duplicati con quelli da cui abbiamo rimosso i duplicati
        non_duplicate_data = data[~duplicates]
        final_data = pd.concat([non_duplicate_data, data_without_duplicates], ignore_index=True)

        return final_data
    
    def active_inactive(self, data: pd.DataFrame):
        """ Filter the data based on active and inactive values
            :param data: the data
            :return: the filtered data
        """
        # Filter the data based on the 'Standard Relation' column
        if self.args.standard_type_act != 'None':
            s_type = self.args.standard_type_act[0].split(',')
            df_act = data[data['Standard Type'].isin(s_type)]

        if self.args.standard_type_perc != 'None':
            p_type = self.args.standard_type_perc[0].split(',')
            df_perc = data[data['Standard Type'].isin(p_type)]
            
        df_perc_rev_act = df_perc[df_perc['Standard Type'] == 'Activity']
        df_perc_rev_act ['class'] = 0
        df_perc_rev_inact = df_perc[df_perc['Standard Type'] == 'Inhibition']
        df_perc_rev_inact['class'] = 2
        df_perc_rev = df_perc_rev_inact.copy()

        #extrapolate data from these sets

        perc_rev_act_c = df_perc_rev_act['Standard Value'].count()
        perc_rev_inact_c = df_perc_rev_inact['Standard Value'].count()

        perc_rev_act_min = df_perc_rev_act['Standard Value'].min()
        perc_rev_act_max = df_perc_rev_act['Standard Value'].max()

        perc_rev_inact_min = df_perc_rev_inact['Standard Value'].min()
        perc_rev_inact_max = df_perc_rev_inact['Standard Value'].max()

        # Filter the data based on the 'Standard Value' and relation column for ACTIVATION
        df_act_rev_act = df_act[df_act['Standard Value'] <= self.args.thr_act]
        df_act_rev_act['class'] = 0
        df_act_rev_inact = df_act[df_act['Standard Value'] >= self.args.thr_act*10]
        df_act_rev_inact['class'] = 2
        df_act_rev_inc = df_act.loc[(df_act['Standard Value'] > self.args.thr_act) & (df_act['Standard Value'] < self.args.thr_act * 10)]
        df_act_rev_inc['class'] = 1
        df_act_rev = pd.concat([df_act_rev_act, df_act_rev_inact])

        #extrapolate data from these sets
        #act_rev_c = df_act_rev['Standard Value'].count()
        act_rev_act_c = df_act_rev_act['Standard Value'].count()
        act_rev_inact_c = df_act_rev_inact['Standard Value'].count()

        act_rev_act_min = df_act_rev_act['Standard Value'].min()
        act_rev_act_max = df_act_rev_act['Standard Value'].max()

        act_rev_inact_min = df_act_rev_inact['Standard Value'].min()
        act_rev_inact_max = df_act_rev_inact['Standard Value'].max()

        # whole dataset
        df_whole = pd.concat([df_act_rev, df_perc_rev])
        #df_whole_act = df_act_rev_act.copy() #metti senza copia
        df_whole_inact = pd.concat([df_act_rev_inact, df_perc_rev_inact])

        df_act_ina_act = pd.concat([df_act_rev_act, df_act_rev_inact])
        #df_perc_2 = df_perc_rev_inact.copy() #metti senza copia

        data_report = pd.DataFrame(columns=[
                                    'ratio active/inactive',
                                    'total_df_records',
                                    'total_std_types',
                                    'total_inhibition',
                                    'data_std_active',
                                    'data_std_inactive',
                                    'data_inhi_act',
                                    'data_inhi_ina',
                                    'data_std_active_min',
                                    'data_std_active_max',
                                    'data_std_inactive_min',
                                    'data_std_inactive_max',
                                    'data_inhi_act_min',
                                    'data_inhi_act_max',
                                    'data_inhi_ina_min',
                                    'data_inhi_ina_max'])
        
        # compute value
        ratio_act_ina = len(df_act_rev_act) / len(df_whole_inact)
        total_df_records = len(df_whole)
        total_std_types = len(df_act_ina_act)
        total_inhibition = len(df_perc_rev_inact)


        
    # Create a dictionary of the data to add to the updater dataframe
        data_dict = {
            'ratio active/inactive':ratio_act_ina,
            'total_df_records':total_df_records,
            'total_std_types':total_std_types,
            'total_inhibition':total_inhibition,
            'data_std_active':act_rev_act_c,
            'data_std_inactive':act_rev_inact_c,
            'data_inhi_act':perc_rev_act_c,
            'data_inhi_ina':perc_rev_inact_c,
            'data_std_active_min':act_rev_act_min,
            'data_std_active_max':act_rev_act_max,
            'data_std_inactive_min':act_rev_inact_min,
            'data_std_inactive_max':act_rev_inact_max,
            'data_inhi_act_min':perc_rev_act_min,
            'data_inhi_act_max':perc_rev_act_max,
            'data_inhi_ina_min':perc_rev_inact_min,
            'data_inhi_ina_max':perc_rev_inact_max,
    }
        
        new_row=pd.DataFrame([data_dict])
        data_report = pd.concat([data_report, new_row], ignore_index=True)

        return data_report, df_whole, df_act_rev_act, df_whole_inact, df_act_rev_inc
    
    def save_data_report(self,path, data_report, whole_dataset, whole_act, whole_inact, inc_data):
        dataset_path = os.path.join(path,'data', 'filtered')
        report_path = os.path.join(path,'data', 'report')

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        if not os.path.exists(report_path):
            os.makedirs(report_path)

        filenames = {
            'whole_dataset.csv': whole_dataset,
            'whole_act.csv': whole_act,
            'whole_inact.csv': whole_inact,
            'inc_data.csv': inc_data,
            'data_report.csv': data_report
        }

        for filename, df in filenames.items():
            if 'report' in filename:
                full_path=os.path.join(report_path, filename)
            else:
                full_path=os.path.join(dataset_path, filename)
            
            df.to_csv(full_path, index=True)