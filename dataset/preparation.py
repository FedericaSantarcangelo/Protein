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
            data = remove_salts(data, self.assay, self.args.standard_type_act)
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
                        data_report, whole_dataset, whole_act,whole_inact, inc_data = self.active_inactive(combined_mut, mixed)
                    else:
                        data_report, whole_dataset, whole_act,whole_inact, inc_data = self.active_inactive(combined_mut, mixed)
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
        data = data.loc[data['Standard Value'] > 0] 
        return data

    def filter_data(self, data: pd.DataFrame):
        """
        Filter the data based on the standard type
        :return: the filtered data
        """
        if self.args.standard_type_log != 'None':
            l_type = self.args.standard_type_log[0].split(',')
            data_log = data[data['Standard Type'].isin(l_type)]
            data_log = data_log_f(l_type ,data_log)
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
            df_act_ids = df_act['Molecule ChEMBL ID'].unique()
            data_perc = data_perc[~data_perc['Molecule ChEMBL ID'].isin(df_act_ids)]

        return df_act, data_perc

    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame: 
        """ 
        Remove duplicate applying a priority to the data
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

        for _, grouper in g_dupl:
            std_type_count = grouper['Standard Type'].value_counts()
            dominant = std_type_count.idxmax()

            grouper = grouper[grouper['Standard Type'] == dominant]

            if dominant:
                grouper.loc[:, 'Standard Relation'] = grouper['Standard Relation'].map(rel_pri)
                grouper.loc[:, 'Source Description'] = grouper['Source Description'].map(src_pri)
                grouper = grouper.sort_values(by=['Standard Relation', 'Source Description'], ascending=[True, True])
            else:
                grouper.loc[:, 'Standard Type'] = grouper['Standard Type'].map(sty_pri)
                grouper.loc[:, 'Standard Relation'] = grouper['Standard Relation'].map(rel_pri)
                grouper.loc[:, 'Source Description'] = grouper['Source Description'].map(src_pri)
                grouper = grouper.sort_values(by=['Standard Type', 'Standard Relation', 'Source Description'], ascending=[True, True, True])

            g_docs = grouper.groupby('Document ChEMBL ID')
            max_group = None
            max_len = 0
            for _, g in g_docs:
                if len(g) > max_len:
                    max_len = len(g)
                    max_group = g
            if max_group is not None:
                sorted_values = max_group["Standard Value"].sort_values()
                if len(sorted_values) > 2:
                    trimmed_values = sorted_values.iloc[1:-1]  
                    median_idx = (trimmed_values - trimmed_values.median()).abs().idxmin() 
                else:
                    median_idx = sorted_values.idxmin() 
                indexes.append(median_idx)

        filter_data = duplicate_data.loc[indexes]

        return pd.concat([unique_data, filter_data])

    
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

    def active_inactive(self, data: pd.DataFrame,mixed: pd.DataFrame):
        """ 
        Filter the data based on active and inactive values
        :return: the filtered data
        """
        s_type = self.args.standard_type_act[0].split(',')
        df_act = data[data['Standard Type'].isin(s_type)]

        p_type = self.args.standard_type_perc[0].split(',')
        df_perc = data[data['Standard Type'].isin(p_type)]

        if 'Activity' not in df_perc['Standard Type'].unique():
            df_perc_act = df_perc[df_perc['Standard Value'] > self.args.thr_perc]
            df_perc_inact = df_perc[df_perc['Standard Value'] < self.args.thr_perc]
        else:
            df_perc_act = df_perc[(df_perc['Standard Type'] != 'Activity') & (df_perc['Standard Value'] > self.args.thr_perc)]
            df_perc_inact = df_perc[(df_perc['Standard Type'] != 'Activity') & (df_perc['Standard Value'] < self.args.thr_perc)]

            df_perc_act_activity = df_perc[(df_perc['Standard Type'] == 'Activity') & (df_perc['Standard Value'] < self.args.thr_perc)]
            df_perc_inact_activity = df_perc[(df_perc['Standard Type'] == 'Activity') & (df_perc['Standard Value'] > self.args.thr_perc)]

            df_perc_act = pd.concat([df_perc_act, df_perc_act_activity], ignore_index=True)
            df_perc_inact = pd.concat([df_perc_inact, df_perc_inact_activity], ignore_index=True)

        df_perc_rev_inact = df_perc_inact.copy()
        df_perc_rev_inact['Class'] = 0
        df_perc_rev_act = df_perc_act.copy()
        df_perc_rev_act['Class'] = 1

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

        df_whole = pd.concat([df_act_rev_act, df_act_rev_inact, df_act_rev_inc, df_perc_rev_act, df_perc_rev_inact], ignore_index=True)
        df_whole_active = pd.concat([df_act_rev_act, df_perc_rev_act], ignore_index=True)
        df_whole_inactive = pd.concat([df_act_rev_inact, df_perc_rev_inact], ignore_index=True)

        
        data_report = pd.DataFrame(columns=['Target Chembl ID', 'Target Name', 'Accession Code', 'Mutant',
                                    'Number of Molecules', 'Ratio active/inactive', 'Total_std_type', 'Total_inhi',
                                    'data_std_active', 'data_std_inactive', 'data_std_inc', 'data_inhi_active', 
                                    'data_inhi_inactive', 'data_std_active_min', 'data_std_inactive_min', 
                                    'data_std_active_max', 'data_std_inactive_max', 'data_std_inc_min', 
                                    'data_std_inc_max', 'data_inhi_active_min', 'data_inhi_inactive_min', 
                                    'data_inhi_active_max', 'data_inhi_inactive_max', 'Quality'])

        target_mutant_groups = df_whole.groupby(['Target ChEMBL ID', 'mutant'])
        for (target_id,mutant), g in target_mutant_groups:
            num_std_active = len(g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 1)])
            num_std_inc= len(g[g['Class'] == 2])
            num_std_inactive = len(g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 0)])

            num_perc_active = len(g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 1)])
            num_perc_inactive = len(g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 0)])

            ratio_active_inactive = 'inf'

            if num_std_inactive !=0:
                ratio_active_inactive = num_std_active / num_std_inactive
            if num_perc_inactive !=0:
                ratio_active_inactive = num_perc_active / num_perc_inactive
            data_dict = {
                'Target Chembl ID': target_id,
                'Target Name': g['Target Name'].iloc[0],
                'Accession Code': g['Accession Code'].iloc[0],
                'Mutant': mutant,
                'Number of Molecules': len(g),
                'Ratio active/inactive': ratio_active_inactive,
                'Total_std_type': len(g[g['Standard Type'].isin(s_type)]),
                'Total_inhi': len(g[g['Standard Type'].isin(p_type)]),
                'data_std_active': num_std_active,
                'data_std_inactive': num_std_inactive,
                'data_std_inc': num_std_inc,
                'data_inhi_active': num_perc_active,
                'data_inhi_inactive': num_perc_inactive,

                'data_std_active_min': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 1)]['Standard Value'].min(),
                'data_std_inactive_min': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 0)]['Standard Value'].min(),
                'data_std_inc_min': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 2)]['Standard Value'].min(),
                'data_std_active_max': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 1)]['Standard Value'].max(),
                'data_std_inactive_max': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 0)]['Standard Value'].max(),
                'data_std_inc_max': g[(g['Standard Type'].isin(s_type)) & (g['Class'] == 2)]['Standard Value'].max(),
                'data_inhi_active_min': g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 1)]['Standard Value'].min(),
                'data_inhi_inactive_min': g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 0)]['Standard Value'].min(),
                'data_inhi_active_max': g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 1)]['Standard Value'].max(),
                'data_inhi_inactive_max': g[(g['Standard Type'].isin(p_type)) & (g['Class'] == 0)]['Standard Value'].max(),
                'num_mixed': len(mixed[mixed['Target ChEMBL ID'] == target_id]),
                'Quality': g['Quality'].iloc[0]
            }
            new_row_df = pd.DataFrame([data_dict])
            data_report = pd.concat([data_report, new_row_df], ignore_index=True)

        data_report.sort_values(by='Target Chembl ID', inplace=True)

        return  data_report, df_whole, df_whole_active, df_whole_inactive, df_act_rev_inc

    