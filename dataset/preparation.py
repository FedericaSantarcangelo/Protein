import pandas as pd
import numpy as np
from argparse import Namespace, ArgumentParser
from utils.args import data_cleaning_args
from utils.file_utils import *
import re
from dataset.mutant import Mutation
import ast


def get_parser() -> ArgumentParser:
    """ Get the parser """
    parser = ArgumentParser()
    data_cleaning_args(parser)
    return parser

class Cleaner():
    def __init__(self,args: Namespace):
        self.args = args
        self.assay = load_file(self.args.path_assay)


    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Clean the data
            :param data: the data
            :return: the cleaned data
        """
        data = self.remove_row(data)
        data = self.filter_data(data)
        data = self.remove_salts(data)
        data = self.compentence(data,self.assay)
        first , second = self.selct_quality(data)
        second,third = split_second(second) #function to split the second quality data in 3° quality data
        
        if self.args.mutation:
            mutation_processor = Mutation(self.args)
            mut,wild_type,mixed = mutation_processor.get_mutations(first.copy())    
            mut_2, wild_type_2, mixed_2 = mutation_processor.get_mutations(second.copy(),'2')
            mut_3, wild_type_3, mixed_3 = mutation_processor.get_mutations(third.copy(),'3')
            mut = pd.concat([mut,wild_type])
        data_report, whole_dataset, whole_act, whole_inact, inc_data = self.active_inactive(mut)
        
        filenames = {
            'whole_dataset_out.csv': whole_dataset,
            'whole_act_out.csv': whole_act,
            'whole_inact_out.csv': whole_inact,
            'inc_data_out.csv': inc_data,
            'data_report_out.csv': data_report,
        }
        save_data_report(self.args.path_output,filenames)
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
        data = data.loc[data['Standard Value'] > 0] # Remove the rows with negative values
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
        
            data_perc = data_perc[data_perc['Assay Description'].str.contains(pattern, regex=True, na=False)]
            data_perc = self.data_perc(data_perc)
        
        df= pd.concat([data_log, data_act, data_perc])
        return df
    
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
        filtered_data = data[data.apply(filter_conditions, axis=1)]
        return filtered_data
           

    def data_log(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Log the data and convert standard types
            :param data: the data
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
    
    def data_act(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Act on the data based on standard units
            :param data: the data
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
    
    def compentence(self, data: pd.DataFrame, assay: pd.DataFrame) -> pd.DataFrame:
        """"Filter data based on the confidence score in the assays file
            :param data: the data
            :param assay: the assay file
            :return: the filtered data
        """
        data = data.copy()
        assay_ids = assay[assay['Confidence Score'] >= 8]['ChEMBL ID'].tolist()
        data = data[data['Assay ChEMBL ID'].isin(assay_ids)]
        data = data[data['Assay Type'].isin(['F', 'B'])]
        return data
    
    def selct_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            In data there are only the records of interest: they represent the first quality data. 
            In other there are records that are not of interest: they represent the second quality data.
        """ 
        other = data.copy()
        if self.args.assay_type != 'None':
            data = data.loc[data['Assay Type'] == self.args.assay_type]
        if self.args.assay_organism != 'None':
            data = data.loc[data['Assay Organism'] == self.args.assay_organism]
        if self.args.BAO_Label != 'None':  
            data = data.loc[data['BAO Label'] == self.args.BAO_Label]
        if self.args.target_type != 'None':
            data = data.loc[data['Target Type'] == self.args.target_type]

        other = other.loc[~other.index.isin(data.index)] # second level data 
        other = other[~other['Molecule ChEMBL ID'].isin(data['Molecule ChEMBL ID'])] #remove the ID already present in the first dataset
        return data,other
    
    def remove_salts(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Remove the salts from the SMILES
            :param smiles: the SMILES
            :return: the SMILES without salts
        """
        cleaned_smiles = []

        for smiles in data['Smiles']:
            components = smiles.split('.')
            main_component = max(components, key=len)
            cleaned_smiles.append(main_component)
    
        data['Smiles'] = cleaned_smiles
        return data
    
    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame: 
        """ Remove duplicate appling a priority to the data
            :param data: the data
            :return: the filtered data
        """
        rel_pri = ast.literal_eval(self.args.rel_pri)
        for key in list(rel_pri.keys()):
            new_key = f"'{key}'"
            rel_pri[new_key] = rel_pri.pop(key)
            
        sty_pri = ast.literal_eval(self.args.sty_pri)
        src_pri = ast.literal_eval(self.args.src_pri)

        duplicates = data.duplicated(subset='Molecule ChEMBL ID', keep=False)
        duplicate_data = data[duplicates]
        unique_data = data[~duplicates] 

        g_dupl = duplicate_data.groupby('Molecule ChEMBL ID')

        indexes = [] #lista indici da tenere

        for _,grouper in g_dupl:
            std_type_count = grouper['Standard Type'].value_counts()
            dominant = std_type_count.idxmax()

            grouper = grouper[grouper['Standard Type'] == dominant]

            if dominant:
                grouper.loc[:,'Standard Relation'] = grouper['Standard Relation'].map(rel_pri)
                grouper.loc[:,'Source Description'] = grouper['Source Description'].map(src_pri)
                grouper = grouper.sort_values(by=['Standard Relation', 'Source Description'], ascending=[True, True])
            else:
                grouper.loc[:,'Standard Type'] = grouper['Standard Type'].map(sty_pri)
                grouper.loc[:,'Standard Relation'] = grouper['Standard Relation'].map(rel_pri)
                grouper.loc[:,'Source Description'] = grouper['Source Description'].map(src_pri)
                grouper = grouper.sort_values(by=['Standard Type', 'Standard Relation', 'Source Description'], ascending=[True, True, True])

            g_docs = grouper.groupby('Document ChEMBL ID')
            max_group = None
            max_len=0
            for _,g in g_docs:
                if len(g) > max_len:
                    max_len = len(g)
                    max_group = g
            if max_group is not None:
                min_val = max_group["Standard Value"].idxmin()
                indexes.append(min_val)
            

        filter_data = duplicate_data.loc[indexes]

        return pd.concat([unique_data,filter_data])

    def active_inactive(self, data: pd.DataFrame):
        """ Filter the data based on active and inactive values
            :param data: the data
            :return: the filtered data
        """
        # Filter the data based on the 'Standard Relation' column

        s_type = self.args.standard_type_act[0].split(',')
        df_act = data[data['Standard Type'].isin(s_type)]

        p_type = self.args.standard_type_perc[0].split(',')
        df_perc = data[data['Standard Type'].isin(p_type)]
        
        if 'Activity' not in df_perc['Standard Type'].unique():
        # Se 'Activity' non è presente, filtra i dati in base alla soglia
            df_perc_act = df_perc[df_perc['Standard Value'] > self.args.thr_perc]
            df_perc_inact = df_perc[df_perc['Standard Value'] < self.args.thr_perc]
        else:
        # Se 'Activity' è presente, filtra i dati in base al tipo standard
            df_perc_act = df_perc[df_perc['Standard Type'] != 'Activity']
            df_perc_act = df_perc_act[df_perc_act['Standard Value'] > self.args.thr_perc]
            df_perc_inact = df_perc[df_perc['Standard Type'] != 'Activity']
            df_perc_inact = df_perc_inact[df_perc_inact['Standard Value'] < self.args.thr_perc]
        
            df_perc_act_i = df_perc[df_perc['Standard Type'] == 'Activity']
            df_perc_act_i = df_perc_act_i[df_perc_act_i['Standard Value'] < self.args.thr_perc]
            df_perc_inact_i = df_perc[df_perc['Standard Type'] == 'Activity']
            df_perc_inact_i = df_perc_inact_i[df_perc_inact_i['Standard Value'] > self.args.thr_perc]

        # Concatenare i risultati
        df_perc_act = pd.concat([df_perc_act, df_perc_act_i], ignore_index=True)
        df_perc_inact = pd.concat([df_perc_inact, df_perc_inact_i], ignore_index=True)

        df_perc_rev_inact = df_perc_inact.copy()
        df_perc_rev_inact['Class'] = 0
        df_perc_rev_act = df_perc_act.copy()
        df_perc_rev_act['Class'] = 1

        perc_rev_act_c = df_perc_act['Standard Value'].count()
        perc_rev_inact_c = df_perc_inact['Standard Value'].count()

        perc_rev_act_min = df_perc_act['Standard Value'].min()
        perc_rev_act_max = df_perc_act['Standard Value'].max()

        perc_rev_inact_min = df_perc_inact['Standard Value'].min()
        perc_rev_inact_max = df_perc_inact['Standard Value'].max()

        df_act_act = df_act[df_act['Standard Value'] <= self.args.thr_act]
        df_act_rev_act = df_act_act.copy()
        df_act_rev_act['Class'] = 1
        
        df_act_inact = df_act[df_act['Standard Value'] >= self.args.thr_act*10]
        df_act_rev_inact = df_act_inact.copy()
        df_act_rev_inact['Class'] = 0

        df_act_inc = df_act.loc[(df_act['Standard Value'] > self.args.thr_act) & (df_act['Standard Value'] < self.args.thr_act * 10)]
        df_act_rev_inc = df_act_inc.copy()
        df_act_rev_inc['Class'] = 2

        act_rev_act_c = df_act_rev_act['Standard Value'].count()
        act_rev_inact_c = df_act_rev_inact['Standard Value'].count()

        act_rev_act_min = df_act_rev_act['Standard Value'].min()
        act_rev_act_max = df_act_rev_act['Standard Value'].max()

        act_rev_inact_min = df_act_rev_inact['Standard Value'].min()
        act_rev_inact_max = df_act_rev_inact['Standard Value'].max()

        df_whole = pd.concat([df_act_rev_act, df_act_rev_inact, df_act_rev_inc, df_perc_rev_act, df_perc_rev_inact])
        df_whole_inact = pd.concat([df_act_rev_inact, df_perc_rev_inact])
        df_whole_act = pd.concat([df_act_rev_act, df_perc_rev_act])

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
        
        ratio_act_ina = len(df_act_rev_act) / len(df_whole_inact)
        total_df_records = len(df_whole)
        total_std_types = len(df_whole_act)
        total_inhibition = len(df_perc_rev_inact)


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

        return data_report, df_whole, df_whole_act, df_whole_inact, df_act_rev_inc
    
 