import pandas as pd
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
        """
        Main function to clean the data
        :return: the cleaned data
        """
        data = self.remove_row(data)
        df_act , df_perc = self.filter_data(data)
        def process_df(data,label):
            data = remove_salts(data, self.assay)
            first , second, third = self.selct_quality(data)
            quality_data = [(first,1),(second,2),(third,3)]
            all_mutations=[]
            all_mixed = []
            if self.args.mutation:
                mutation_processor = Mutation(self.args)
                for quality_data_item,quality_level in quality_data:
                    mut,wild_type,mixed = mutation_processor.get_mutations(quality_data_item.copy(), label, str(quality_level)) 
                    combined_mut = pd.concat([mut,wild_type])
                    combined_mut['Quality'] = str(quality_level)
                    all_mutations.append(combined_mut)
                    all_mixed.append(mixed)
                    if not combined_mut.empty:
                        data_report, whole_dataset, whole_act,whole_inact, inc_data = self.active_inactive(combined_mut, quality_level)
                    else:
                        data_report, whole_dataset, whole_act,whole_inact, inc_data = self.active_inactive(mixed, quality_level)
                    filenames = {
                        'whole_dataset_out.csv': whole_dataset,
                        'whole_act_out.csv': whole_act,
                        'whole_inact_out.csv': whole_inact,
                        'inc_data_out.csv': inc_data,
                        'data_report_out.csv': data_report,
                    }
                    save_data_report(self.args.path_output, label, filenames)
            if all_mutations and len(all_mutations) > 100:
                return pd.concat(all_mutations)
            else:
            #all_mixed.append(all_mutations)
                return pd.concat(all_mixed)
        act = process_df(df_act, 'act')
        perc = process_df(df_perc, 'perc')

        return pd.concat([act, perc])

    def remove_row(self,data: pd.DataFrame):
        """
        Remove the row with missing values
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
        """
        Filter the data based on the standard type
        :return: the filtered data
        """
        if self.args.standard_type_log != 'None':
            l_type = self.args.standard_type_log[0].split(',')
            data_log = data[data['Standard Type'].isin(l_type)]
            data_log = data_log_f(self.args.standard_type_log ,data_log)

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
            data_perc = data_perc_f(self.args.thr_perc , data_perc)
        
        df_act= pd.concat([data_log, data_act])
        
        if not data_perc.empty:
            df_act_ids = df_act['Molecule ChEMBL ID'].unique()  # Identificatori unici di df_act
            data_perc = data_perc[~data_perc['Molecule ChEMBL ID'].isin(df_act_ids)]  # Filtraggio di data_perc

        return df_act, data_perc

    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Remove duplicate appling a priority to the data
        :return: the filtered data
        """            
        rel_pri = ast.literal_eval(self.args.rel_pri)
        for key in list(rel_pri.keys()):
            new_key = f"'{key}'"
            rel_pri[new_key] = rel_pri.pop(key)
            
        sty_pri = ast.literal_eval(self.args.sty_pri)
        src_pri = ast.literal_eval(self.args.src_pri)

        duplicates = data.duplicated(subset=['Molecule ChEMBL ID','mutant'], keep=False)
        duplicate_data = data[duplicates]
        unique_data = data[~duplicates] 

        g_dupl = duplicate_data.groupby(['Molecule ChEMBL ID','mutant'])
        indexes = []

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
    
    def selct_quality( self, data: pd.DataFrame) -> pd.DataFrame:
        """
        In data there are only the records of interest: they represent the first quality data. 
        In other there are records that are not of interest: they represent the second quality data.
        return: first, second and third quality data
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

        other = other.loc[~other.index.isin(data.index)]
        other = other[~other['Molecule ChEMBL ID'].isin(data['Molecule ChEMBL ID'])] 
        second,third = split_second(other)
        return data,second,third

    def active_inactive(self, data: pd.DataFrame,flag):
        """ 
        Filter the data based on active and inactive values
        :return: the filtered data
        """
        # Filter the data based on the 'Standard Relation' column
        s_type = self.args.standard_type_act[0].split(',')
        df_act = data[data['Standard Type'].isin(s_type)]

        p_type = self.args.standard_type_perc[0].split(',')
        df_perc = data[data['Standard Type'].isin(p_type)]
        
        if 'Activity' not in df_perc['Standard Type'].unique():
            df_perc_act = df_perc[df_perc['Standard Value'] > self.args.thr_perc]
            df_perc_inact = df_perc[df_perc['Standard Value'] < self.args.thr_perc]
        else:
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
        if not df_act_rev_act.empty:
            df_act_rev_act.loc[:, 'Class'] = 1
        
        df_act_inact = df_act[df_act['Standard Value'] >= self.args.thr_act*10]
        df_act_rev_inact = df_act_inact.copy()
        if not df_act_rev_inact.empty:
            df_act_rev_inact.loc[:,'Class'] = 0

        df_act_inc = df_act.loc[(df_act['Standard Value'] > self.args.thr_act) & (df_act['Standard Value'] < self.args.thr_act * 10)]
        df_act_rev_inc = df_act_inc.copy()
        if not df_act_rev_inc.empty:
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
        if not df_whole_inact.empty:
            ratio_act_ina = len(df_act_rev_act) / len(df_whole_inact)
        else:
            ratio_act_ina = len(df_act_rev_act) / 1

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

        for key,row in data_dict.items():
            if pd.isna(row):
                data_dict[key] = 0

        new_row=pd.DataFrame([data_dict])
        data_report = pd.concat([data_report, new_row], ignore_index=True)
        return data_report, df_whole, df_whole_act, df_whole_inact, df_act_rev_inc
    
 