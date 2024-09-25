import pandas as pd
import numpy as np
from argparse import Namespace, ArgumentParser
from utils.args import data_cleaning_args
from utils.file_utils import *
from utils.data_handling import *
import re
from dataset.mutants import Mutation
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
        """Main function to clean the data
            :return: the cleaned data
        """
        data = self.remove_row(data)
        data = self.filter_data(data)
        data = remove_salts(data) 
        first , second, third = selct_quality(data)   
        if self.args.mutation: #togliere le ripetizioni di codice per le qualità
            mutation_processor = Mutation(self.args)
            mut,wild_type,mixed = mutation_processor.get_mutations(first.copy())    
            mut_2, wild_type_2, mixed_2 = mutation_processor.get_mutations(second.copy(),'2')
            mut_3, wild_type_3, mixed_3 = mutation_processor.get_mutations(third.copy(),'3')
            mut_1 = pd.concat([mut,wild_type])
            mut_1.loc[:,'Quality'] = 1
            mut_2 = pd.concat([mut_2,wild_type_2])
            mut_2.loc[:,'Quality'] = 2
            mut_3 = pd.concat([mut_3,wild_type_3])
            mut_3.loc[:,'Quality'] = 3

        data_report_1, whole_dataset_1, whole_act_1, whole_inact_1, inc_data_1 = self.active_inactive(mut_1,1)
        data_report_2, whole_dataset_2, whole_act_2, whole_inact_2, inc_data_2 = self.active_inactive(mut_2,2)
        data_report_3, whole_dataset_3, whole_act_3, whole_inact_3, inc_data_3 = self.active_inactive(mut_3,3)
        
        def dict_file(whole_dataset, whole_act, whole_inact, inc_data,data_report):
            filenames = {
                'whole_dataset_out.csv': whole_dataset,
                'whole_act_out.csv': whole_act,
                'whole_inact_out.csv': whole_inact,
                'inc_data_out.csv': inc_data,
                'data_report_out.csv': data_report,
            }
            save_data_report(self.args.path_output,filenames)
#togliere le ripetizioni di codice per salvare i file
        dict_file(whole_dataset_1, whole_act_1, whole_inact_1, inc_data_1,data_report_1)
        dict_file(whole_dataset_2, whole_act_2, whole_inact_2, inc_data_2,data_report_2)
        dict_file(whole_dataset_3, whole_act_3, whole_inact_3, inc_data_3,data_report_3)
        
        return pd.concat([mut_1,mut_2,mut_3])

    def remove_row(self,data: pd.DataFrame):
        """Remove the row with missing values
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
            :return: the filtered data
        """
        if self.args.standard_type_log != 'None':
            l_type = self.args.standard_type_log[0].split(',')
            data_log = data[data['Standard Type'].isin(l_type)]
            data_log = data_log_f(data_log)

        if self.args.standard_type_act != 'None':
            s_type = self.args.standard_type_act[0].split(',')
            data_act = data[data['Standard Type'].isin(s_type)]
            data_act = data_act_f(data_act)

        if self.args.standard_type_perc != 'None' and self.args.assay_description_perc != 'None':
            p_type = self.args.standard_type_perc[0].split(',')
            assay_desc = self.args.assay_description_perc[0].split(',')
            data_perc = data[data['Standard Type'].isin(p_type)]
            pattern = '|'.join(map(re.escape, assay_desc))
            data_perc = data_perc[data_perc['Assay Description'].str.contains(pattern, regex=True, na=False)]
            data_perc = data_perc_f(data_perc)
        
        df= pd.concat([data_log, data_act, data_perc])
        return df

    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame: 
        """ Remove duplicate appling a priority to the data
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

    def active_inactive(self, data: pd.DataFrame,flag):
        """ Filter the data based on active and inactive values
            :return: the filtered data
        """
        # Filter the data based on the 'Standard Relation' column
#modificare i nomi usati nella funzione per renderla più leggibile
        s_type = self.args.standard_type_act[0].split(',')
        df_act = data[data['Standard Type'].isin(s_type)]

        p_type = self.args.standard_type_perc[0].split(',')
        df_perc = data[data['Standard Type'].isin(p_type)]
        
#modificare questa parte di codice per renderla più leggibile
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
        df_act_rev_act.loc[:,'Class'] = 1
        
        df_act_inact = df_act[df_act['Standard Value'] >= self.args.thr_act*10]
        df_act_rev_inact = df_act_inact.copy()
        df_act_rev_inact.loc[:,'Class'] = 0

        df_act_inc = df_act.loc[(df_act['Standard Value'] > self.args.thr_act) & (df_act['Standard Value'] < self.args.thr_act * 10)]
        df_act_rev_inc = df_act_inc.copy()
        df_act_rev_inc.loc[:,'Class'] = 2

        act_rev_act_c = df_act_rev_act['Standard Value'].count()
        act_rev_inact_c = df_act_rev_inact['Standard Value'].count()

        act_rev_act_min = df_act_rev_act['Standard Value'].min()
        act_rev_act_max = df_act_rev_act['Standard Value'].max()

        act_rev_inact_min = df_act_rev_inact['Standard Value'].min()
        act_rev_inact_max = df_act_rev_inact['Standard Value'].max()

        df_whole = pd.concat([df_act_rev_act, df_act_rev_inact, df_act_rev_inc, df_perc_rev_act, df_perc_rev_inact])
        df_whole_inact = pd.concat([df_act_rev_inact, df_perc_rev_inact])
        df_whole_act = pd.concat([df_act_rev_act, df_perc_rev_act])

#spostare questi dataframe e dizionario in un'altra funzione o file per rendere il codice più leggibile
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
                                    'data_inhi_ina_max',
                                    'quality'])
        
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
            'quality': flag
    }
        for key,row in data_dict.items():
            if pd.isna(row):
                data_dict[key] = 0

        new_row=pd.DataFrame([data_dict])
        data_report = pd.concat([data_report, new_row], ignore_index=True)
        return data_report, df_whole, df_whole_act, df_whole_inact, df_act_rev_inc
    
 