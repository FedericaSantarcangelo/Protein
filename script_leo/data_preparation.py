#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:11:08 2024

@author: leonardo
"""

import pandas as pd
import os
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    return df

def process_data_1(df):
    
    df_data = df.copy()
    
    # Remove rows without values
    df_data = df_data.dropna(subset=['Smiles',
                                     'Standard Type',
                                     'Standard Relation',
                                     'Standard Value',
                                     'Standard Units'])
    
    df_data = df_data.loc[df_data['Standard Value'] > 0]
    
    df_data = df_data.loc[df_data['Assay Type'] == 'F']
    df_data = df_data.loc[df_data['Assay Organism'] == 'Homo sapiens']
    df_data = df_data.loc[df_data['BAO Label'] == 'cell-based format']
    df_data = df_data.loc[df_data['Target Type'] == 'CELL-LINE']

    #=============# Filtering of the dataset
    df_data_perc = df_data[df_data['Standard Type'].isin(['Inhibition'])]
    
    df_data_log = df_data[df_data['Standard Type'].isin(['pIC50',
                                                         'pED50',
                                                         'pGI50'])]
    
    df_data_act = df_data[df_data['Standard Type'].isin(['IC50',
                                                         'ED50',
                                                         'GI50'])]
    
    # Filt percentage data
    df_data_perc = df_data_perc[df_data_perc['Standard Units'] == '%']
    
    df_data_perc_5 = df_data_perc[df_data_perc['Assay Description'].str.contains("5 uM")]
    df_data_perc_10 = df_data_perc[df_data_perc['Assay Description'].str.contains("10 uM")]
    df_data_perc_20 = df_data_perc[df_data_perc['Assay Description'].str.contains("20 uM")]
    df_data_perc_30 = df_data_perc[df_data_perc['Assay Description'].str.contains("30 uM")]
    df_data_perc_40 = df_data_perc[df_data_perc['Assay Description'].str.contains("40 uM")]
    df_data_perc_50 = df_data_perc[df_data_perc['Assay Description'].str.contains("50 uM")]
    df_data_perc_100 = df_data_perc[df_data_perc['Assay Description'].str.contains("100 uM")]

    del df_data_perc

    df_data_perc_2 = pd.concat([df_data_perc_5,
                                df_data_perc_10,
                                df_data_perc_20,
                                df_data_perc_30,
                                df_data_perc_40,
                                df_data_perc_50,
                                df_data_perc_100])
    
    # Filt logaritmic data
    df_data_log_mM = df_data_log[df_data_log['Standard Units'] =='mM']
    df_data_log_uM = df_data_log[df_data_log['Standard Units'] =='uM']
    df_data_log_µM = df_data_log[df_data_log['Standard Units'] =='µM']
    df_data_log_nM = df_data_log[df_data_log['Standard Units'] =='nM']
    
    df_data_log_mM['Standard Value'] = [round(np.log(np.exp(float(jj))*1000000), 3) for jj in df_data_log_mM['Standard Value'].values.tolist()]
    df_data_log_uM['Standard Value'] = [round(np.log(np.exp(float(jj))*1000), 3) for jj in df_data_log_uM['Standard Value'].values.tolist()]
    df_data_log_µM['Standard Value'] = [round(np.log(np.exp(float(jj))*1000), 3) for jj in df_data_log_µM['Standard Value'].values.tolist()]

    df_data_log_mM['Standard Units'] = ['nM']*len(df_data_log_mM['Standard Units'])
    df_data_log_uM['Standard Units'] = ['nM']*len(df_data_log_uM['Standard Units'])
    df_data_log_µM['Standard Units'] = ['nM']*len(df_data_log_µM['Standard Units'])
    
    # --- Convert standard types for each logaritmic standard type
    df_data_log_mM_ic50 = df_data_log_mM[df_data_log_mM['Standard Type'].str.contains('pIC50')]
    df_data_log_mM_ic50["Standard Type"] = df_data_log_mM_ic50["Standard Type"].replace(['pIC50'],'IC50')
    df_data_log_mM_ec50 = df_data_log_mM[df_data_log_mM['Standard Type'].str.contains('pED50')]
    df_data_log_mM_ec50["Standard Type"] = df_data_log_mM_ec50["Standard Type"].replace(['pED50'],'ED50') 
    df_data_log_mM_ki = df_data_log_mM[df_data_log_mM['Standard Type'].str.contains('pGI50')]
    df_data_log_mM_ki["Standard Type"] = df_data_log_mM_ki["Standard Type"].replace(['pGI50'],'GI50')
    df_data_log_mM_2 = pd.concat([df_data_log_mM_ic50,
                                 df_data_log_mM_ec50,
                                 df_data_log_mM_ki])

    df_data_log_uM_ic50 = df_data_log_uM[df_data_log_uM['Standard Type'].str.contains('pIC50')]
    df_data_log_uM_ic50["Standard Type"] = df_data_log_uM_ic50["Standard Type"].replace(['pIC50'],'IC50')
    df_data_log_uM_ec50 = df_data_log_uM[df_data_log_uM['Standard Type'].str.contains('pED50')]
    df_data_log_uM_ec50["Standard Type"] = df_data_log_uM_ec50["Standard Type"].replace(['pED50'],'ED50')
    df_data_log_uM_ki = df_data_log_uM[df_data_log_uM['Standard Type'].str.contains('pGI50')]
    df_data_log_uM_ki["Standard Type"] = df_data_log_uM_ki["Standard Type"].replace(['pGI50'],'GI50')
    df_data_log_uM_2 = pd.concat([df_data_log_uM_ic50,
                                  df_data_log_uM_ec50,
                                  df_data_log_uM_ki])

    df_data_log_µM_ic50 = df_data_log_µM[df_data_log_µM['Standard Type'].str.contains('pIC50')]
    df_data_log_µM_ic50["Standard Type"] = df_data_log_µM_ic50["Standard Type"].replace(['pIC50'],'IC50')
    df_data_log_µM_ec50 = df_data_log_µM[df_data_log_µM['Standard Type'].str.contains('pED50')]
    df_data_log_µM_ec50["Standard Type"] = df_data_log_µM_ec50["Standard Type"].replace(['pED50'],'ED50')
    df_data_log_µM_ki = df_data_log_µM[df_data_log_µM['Standard Type'].str.contains('pGI50')]
    df_data_log_µM_ki["Standard Type"] = df_data_log_µM_ki["Standard Type"].replace(['pGI50'],'GI50')
    df_data_log_µM_2 = pd.concat([df_data_log_µM_ic50,
                                  df_data_log_µM_ec50,
                                  df_data_log_µM_ki])

    df_data_log_nM_ic50 = df_data_log_nM[df_data_log_nM['Standard Type'].str.contains('pIC50')]
    df_data_log_nM_ic50["Standard Type"] = df_data_log_nM_ic50["Standard Type"].replace(['pIC50'],'IC50')
    df_data_log_nM_ec50 = df_data_log_nM[df_data_log_nM['Standard Type'].str.contains('pED50')]
    df_data_log_nM_ec50["Standard Type"] = df_data_log_nM_ec50["Standard Type"].replace(['pED50'],'ED50')
    df_data_log_nM_ki = df_data_log_nM[df_data_log_nM['Standard Type'].str.contains('pGI50')]
    df_data_log_nM_ki["Standard Type"] = df_data_log_nM_ki["Standard Type"].replace(['pGI50'],'GI50')
    df_data_log_nM_2 = pd.concat([df_data_log_nM_ic50,
                                  df_data_log_nM_ec50,
                                  df_data_log_nM_ki])

    del df_data_log
    
    df_data_log_2 = pd.concat([df_data_log_mM_2,
                             df_data_log_uM_2,
                             df_data_log_µM_2,
                             df_data_log_nM_2])

    del df_data_log_mM_2
    del df_data_log_uM_2
    del df_data_log_µM_2
    del df_data_log_nM_2
   
    # Filt activity data
    df_data_act_mM = df_data_act[df_data_act['Standard Units'] =='mM']
    df_data_act_uM = df_data_act[df_data_act['Standard Units'] =='uM']
    df_data_act_µM = df_data_act[df_data_act['Standard Units'] =='µM']
    df_data_act_nM = df_data_act[df_data_act['Standard Units'] =='nM']
    
    df_data_act_mM['Standard Value'] = df_data_act_mM['Standard Value']*1000000
    df_data_act_uM['Standard Value'] = df_data_act_uM['Standard Value']*1000
    df_data_act_µM['Standard Value'] = df_data_act_µM['Standard Value']*1000

    df_data_act_mM['Standard Units'] = ['nM']*len(df_data_act_mM['Standard Units'])
    df_data_act_uM['Standard Units'] = ['nM']*len(df_data_act_uM['Standard Units'])
    df_data_act_µM['Standard Units'] = ['nM']*len(df_data_act_µM['Standard Units'])

    del df_data_act
    
    df_data_act_2 = pd.concat([df_data_act_mM,
                             df_data_act_uM,
                             df_data_act_µM,
                             df_data_act_nM])

    del df_data_act_mM
    del df_data_act_uM
    del df_data_act_µM
    del df_data_act_nM
    
    df_data_all = pd.concat([df_data_act_2,
                             df_data_perc_2,
                             df_data_log_2])
    
    del df_data
    
    return df_data_all


def process_data_2(df):
    l_types = ["IC50", "ED50", "GI50"]
    df_other = df[df["Standard Type"].isin(l_types)]
    df_perc = df.loc[df["Standard Type"] == "Inhibition"]
    #------------------------------------------------------------------------#
    # Groupby and obtain stats for other
    grouped_other = df_other.groupby(["Molecule ChEMBL ID"])["Standard Value"].describe()
    grouped_other.sort_values(["count"], ascending=False, inplace=True)
    
    grouped_describe_other = grouped_other[["count", "mean", "std"]]

    grouped_describe_other = grouped_describe_other.reset_index()
    grouped_describe_other = grouped_describe_other.rename(columns={
            "count": "records_n°_mol_count",
            "mean": "records_mol_val_mean",
            "std": "records_mol_val_std"})

    #------------------------------------------------------------------------#
    grouped_doc_other = df_other.groupby(["Document ChEMBL ID"])["Standard Value"].describe()
    grouped_doc_other.sort_values(["count"], ascending=False, inplace=True)
    
    grouped_doc_describe_other = grouped_doc_other[["count", "mean", "std"]]

    grouped_doc_describe_other = grouped_doc_describe_other.reset_index()
    grouped_doc_describe_other = grouped_doc_describe_other.rename(columns={
            "count": "doc_n°_mol_count",
            "mean": "doc_mol_val_mean",
            "std": "doc_mol_val_std"})
    #------------------------------------------------------------------------#
    df_other_stats = pd.merge(left=df_other, right=grouped_describe_other, how="inner", on="Molecule ChEMBL ID")
    df_other_stats = pd.merge(left=df_other_stats, right=grouped_doc_describe_other, how="inner", on="Document ChEMBL ID")

    df_other_stats['lower_std_bound'] = df_other_stats['records_mol_val_mean'] - df_other_stats['records_mol_val_std']
    df_other_stats['upper_std_bound'] = df_other_stats['records_mol_val_mean'] + df_other_stats['records_mol_val_std']
    df_other_stats['diff'] = np.abs(df_other_stats['Standard Value'] - df_other_stats['records_mol_val_mean'])
    df_other_stats.sort_values(['diff'], ascending=False, inplace=True)
    df_other_stats['ponderate_mean'] = (df_other_stats['Standard Value']*df_other_stats['doc_n°_mol_count'])/df_other_stats['records_n°_mol_count']


    other_less = df_other_stats[df_other_stats['records_n°_mol_count'] == 1]
    other_more = df_other_stats[df_other_stats['records_n°_mol_count'] > 1]  
    
    #------------------------------------------------------------------------#
    # Initialize lists to collect DataFrames
    result_other_dfs = []

    # Get the unique values in the 'Molecule ChEMBL ID' column
    unique_ids = other_more['Molecule ChEMBL ID'].unique()

    # Iterate over each unique ID and apply the previous logic
    for id in unique_ids:
        # Subset the DataFrame for the current ID
        subset_df_other = other_more[other_more['Molecule ChEMBL ID'] == id]
    
        # Apply the previous logic to the subset DataFrame
        lowest_diff_other = float('inf')
        highest_ponderate_mean_other = float('-inf')
        current_result_other = None
    
        for index, row in subset_df_other.iterrows():
            if row['doc_n°_mol_count'] > highest_ponderate_mean_other:
                highest_ponderate_mean_other = row['doc_n°_mol_count']
                lowest_diff_other = row['diff']
                current_result_other = row
            elif row['doc_n°_mol_count'] == highest_ponderate_mean_other and row['diff'] < lowest_diff_other:
                    lowest_diff_other = row['diff']
                    current_result_other = row
        
        current_df_other = pd.DataFrame(current_result_other).T
        result_other_dfs.append(current_df_other)

        # Append the resulting row to the final DataFrame
        result_other = pd.concat(result_other_dfs, ignore_index=True)

    result_other.reset_index(drop=True, inplace=True)
    df_other_revised = pd.concat([result_other, other_less])
    l_other_whole = df_other_revised['Molecule ChEMBL ID'].unique().tolist()
    #------------------------------------------------------------------------#
    # Groupby and obtain stats for perc, and remove molecules if already in df_other_revised
    grouped_perc = df_perc.groupby(["Molecule ChEMBL ID"])["Standard Value"].describe()
    grouped_perc.sort_values(["count"], ascending=False, inplace=True)
    
    grouped_describe_perc = grouped_perc[["count", "mean", "std"]]

    grouped_describe_perc = grouped_describe_perc.reset_index()
    grouped_describe_perc = grouped_describe_perc.rename(columns={
            "count": "records_n°_mol_count",
            "mean": "records_mol_val_mean",
            "std": "records_mol_val_std"})

    #------------------------------------------------------------------------#
    grouped_doc_perc = df_other.groupby(["Document ChEMBL ID"])["Standard Value"].describe()
    grouped_doc_perc.sort_values(["count"], ascending=False, inplace=True)
    
    grouped_doc_describe_perc = grouped_doc_perc[["count", "mean", "std"]]

    grouped_doc_describe_perc = grouped_doc_describe_perc.reset_index()
    grouped_doc_describe_perc = grouped_doc_describe_perc.rename(columns={
            "count": "doc_n°_mol_count",
            "mean": "doc_mol_val_mean",
            "std": "doc_mol_val_std"})

    df_perc_stats = pd.merge(left=df_perc, right=grouped_describe_perc, how="inner", on="Molecule ChEMBL ID")
    df_perc_stats = pd.merge(left=df_perc_stats, right=grouped_doc_describe_perc, how="inner", on="Document ChEMBL ID")

    
    df_perc_stats['lower_std_bound'] = df_perc_stats['records_mol_val_mean'] - df_perc_stats['records_mol_val_std']
    df_perc_stats['upper_std_bound'] = df_perc_stats['records_mol_val_mean'] + df_perc_stats['records_mol_val_std']
    df_perc_stats['diff'] = np.abs(df_perc_stats['Standard Value'] - df_perc_stats['records_mol_val_mean'])
    df_perc_stats['ponderate_mean'] = (df_perc_stats['Standard Value']*df_perc_stats['doc_n°_mol_count'])/df_perc_stats['records_n°_mol_count']


    perc_less = df_perc_stats[df_perc_stats['records_n°_mol_count'] == 1] 
    perc_more = df_perc_stats[df_perc_stats['records_n°_mol_count'] > 1] 
    #------------------------------------------------------------------------#
    # Assuming your DataFrame is called 'df'
    result_perc_dfs = []


    # Get the unique values in the 'Molecule ChEMBL ID' column
    unique_ids = perc_more['Molecule ChEMBL ID'].unique()

    # Iterate over each unique ID and apply the previous logic
    for id in unique_ids:
        # Subset the DataFrame for the current ID
        subset_df_perc = perc_more[perc_more['Molecule ChEMBL ID'] == id]
    
        # Apply the previous logic to the subset DataFrame
        lowest_diff_perc = float('inf')
        highest_ponderate_mean_perc = float('-inf')
        current_result_perc = None
    
        for index, row in subset_df_perc.iterrows():
            if row['doc_n°_mol_count'] > highest_ponderate_mean_perc:
                highest_ponderate_mean_perc = row['doc_n°_mol_count']
                lowest_diff_perc = row['diff']
                current_result_perc = row
            elif row['doc_n°_mol_count'] == highest_ponderate_mean_perc and row['diff'] < lowest_diff_perc:
                    lowest_diff_perc = row['diff']
                    current_result_perc = row
    
        current_df_perc = pd.DataFrame(current_result_perc).T
        result_perc_dfs.append(current_df_perc)
    
        # Append the resulting row to the final DataFrame
        result_perc = pd.concat(result_perc_dfs, ignore_index=True)

    result_perc.reset_index(drop=True, inplace=True)
    df_perc_revised = pd.concat([result_perc, perc_less])
    l_perc_whole = df_perc_revised['Molecule ChEMBL ID'].unique().tolist()
    df_perc_revised_2 = df_perc_revised[~df_perc_revised['Molecule ChEMBL ID'].isin(l_other_whole)]
    #------------------------------------------------------------------------#
    
    df_revised = pd.concat([df_other_revised, df_perc_revised_2])
    
    #------------------------------------------------------------------------#


    return df_revised, df_other_revised, df_perc_revised_2, l_other_whole, l_perc_whole


# Create a function to sort and divide the dataset in active/inactive
def process_data_3(df):
    # Copy working dataframe
    df_data = df.copy()
    
    # Divide activity based on standard type
    df_data_perc = df_data[df_data['Standard Type'].isin(['Inhibition'])]
    
    df_data_other = df_data[df_data['Standard Type'].isin(['IC50',
                                                         'ED50',
                                                         'GI50'])]
    
    # Filter inhibition data
    df_data_perc = df_data_perc[df_data_perc['Standard Units'] == '%']

    # Retain only data with = relation
    df_data_perc_eq = df_data_perc[df_data_perc['Standard Relation'] == "'='"]
    df_data_perc_eq_act = df_data_perc_eq[df_data_perc_eq['Standard Value'] > 50.0]
    df_data_perc_eq_ina = df_data_perc_eq[df_data_perc_eq['Standard Value'] < 50.0]
    
    # Retain only data with < relation for those under 50%
    df_data_perc_less = df_data_perc[df_data_perc['Standard Relation'] == "'<'"]
    df_data_perc_less_ina = df_data_perc_less[df_data_perc_less['Standard Value'] < 50.0]

    # Retain only data with > relation for those above 50%
    df_data_perc_more = df_data_perc[df_data_perc['Standard Relation'] == "'>'"]
    df_data_perc_more_act = df_data_perc_more[df_data_perc_more['Standard Value'] > 50.0]
    
    # Merge all the data in one dataframe
    df_data_perc_rev_act = pd.concat([df_data_perc_eq_act,
                                      df_data_perc_more_act])
    
    df_data_perc_rev_ina = pd.concat([df_data_perc_eq_ina,
                                      df_data_perc_less_ina])
    
    df_data_perc_rev = df_data_perc_rev_ina.copy()

    # Extrapolate data for each subset of the dataframe
    df_data_perc_rev_c = df_data_perc_rev['Standard Value'].count()
    df_data_perc_rev_act_c = df_data_perc_rev_act['Standard Value'].count()
    df_data_perc_rev_ina_c = df_data_perc_rev_ina['Standard Value'].count()

    df_data_perc_rev_act_min = df_data_perc_rev_act['Standard Value'].min()
    df_data_perc_rev_act_max = df_data_perc_rev_act['Standard Value'].max()
    
    df_data_perc_rev_ina_min = df_data_perc_rev_ina['Standard Value'].min()
    df_data_perc_rev_ina_max = df_data_perc_rev_ina['Standard Value'].max()
    
    ### Filt other std types
    df_data_other = df_data_other[df_data_other['Standard Units'] == 'nM']
    
    # Retain only data with = relation
    df_data_other_eq = df_data_other[df_data_other['Standard Relation'] == "'='"]
    df_data_other_eq_act = df_data_other_eq.loc[df_data_other_eq['Standard Value'] <= 1000]
    df_data_other_eq_inc = df_data_other_eq.loc[(df_data_other_eq['Standard Value'] > 1000) & (df_data_other_eq['Standard Value'] < 10000)]    
    df_data_other_eq_ina = df_data_other_eq.loc[df_data_other_eq['Standard Value'] >= 10000]
  
    # Retain only data with < relation
    df_data_other_less = df_data_other[df_data_other['Standard Relation'] == "'<'"]
    df_data_other_less_act = df_data_other_less.loc[df_data_other_less['Standard Value'] <= 1000]
    
     # Retain only data with < relation
    df_data_other_more = df_data_other[df_data_other['Standard Relation'] == "'>'"]
    df_data_other_more_ina = df_data_other_more.loc[df_data_other_more['Standard Value'] >= 10000]

    # Merge all the data in one dataframe
    df_data_other_rev_act = pd.concat([df_data_other_eq_act,
                                      df_data_other_less_act])
    
    df_data_other_rev_ina = pd.concat([df_data_other_eq_ina,
                                      df_data_other_more_ina])
    
    df_data_other_rev = pd.concat([df_data_other_rev_act,
                                  df_data_other_rev_ina])
    
    # Extrapolate data for each subset of the dataframe
    df_data_other_rev_c = df_data_other_rev['Standard Value'].count()
    df_data_other_rev_act_c = df_data_other_rev_act['Standard Value'].count()
    df_data_other_rev_ina_c = df_data_other_rev_ina['Standard Value'].count()

    df_data_other_rev_act_min = df_data_other_rev_act['Standard Value'].min()
    df_data_other_rev_act_max = df_data_other_rev_act['Standard Value'].max()

    df_data_other_rev_ina_min = df_data_other_rev_ina['Standard Value'].min()
    df_data_other_rev_ina_max = df_data_other_rev_ina['Standard Value'].max()
    
  
    # Create complete, actives and inactives dataframes by concat
  ##### Create a unique dataframe with the whole dataset processed2
  
    df_whole_dataset = pd.concat([df_data_other_rev,
                                  df_data_perc_rev])
  
    df_whole_act_dataset = df_data_other_rev_act.copy()
    
    df_whole_ina_dataset = pd.concat([df_data_other_rev_ina,
                                      df_data_perc_rev_ina])
    
    # Create datasets of other and perc
    df_other_dataset_2 = pd.concat([df_data_other_rev_act,
                                    df_data_other_rev_ina])
    
    df_perc_dataset_2 = df_data_perc_rev_ina.copy()
    
    # ---------- Store values in a report dataframe and dictionary --------- #
    # Create variables to produce a report of the dataframes
    
    # Create a report dataframe
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
    
    
    #
    ratio_act_ina = len(df_whole_act_dataset)/len(df_whole_ina_dataset)
    total_df_records = len(df_whole_dataset)
    total_std_types = len(df_other_dataset_2)
    total_inhibition = len(df_perc_dataset_2)
    data_std_active= len(df_data_other_rev_act)
    data_std_inactive= len(df_data_other_rev_ina)
    
    # Create a dictionary of the data to add to the updater dataframe
    data_dict = {
        'ratio active/inactive':ratio_act_ina,
        'total_df_records':total_df_records,
        'total_std_types':total_std_types,
        'total_inhibition':total_inhibition,
        'data_std_active':df_data_other_rev_act_c,
        'data_std_inactive':df_data_other_rev_ina_c,
        'data_inhi_act':df_data_perc_rev_act_c,
        'data_inhi_ina':df_data_perc_rev_ina_c,
        'data_std_active_min':df_data_other_rev_act_min,
        'data_std_active_max':df_data_other_rev_act_max,
        'data_std_inactive_min':df_data_other_rev_ina_min,
        'data_std_inactive_max':df_data_other_rev_ina_max,
        'data_inhi_act_min':df_data_perc_rev_act_min,
        'data_inhi_act_max':df_data_perc_rev_act_max,
        'data_inhi_ina_min':df_data_perc_rev_ina_min,
        'data_inhi_ina_max':df_data_perc_rev_ina_max,
    }
    
    # Convert the dictionary to a DataFrame and add it to the existing data_report DataFrame
    new_row_df = pd.DataFrame([data_dict])
    new_row = new_row.dropna(how='all')
    data_report = pd.concat([data_report, new_row_df], ignore_index=True)
    
  
    
    print('Dataset filtered correctly')

    return df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report

def remove_salts(smiles):
    """
    Remove salts from SMILES strings. Salts are assumed to be separated by dots.
    
    Parameters:
    - smiles (str): A SMILES string possibly containing salts.
    
    Returns:
    - str: The SMILES string of the main compound, with salts removed.
    """
    # Split the SMILES string into components based on '.'
    components = smiles.split('.')
    
    # Assume the main compound is the largest component
    main_compound = max(components, key=len)
    return main_compound

def deduplicate_datasets(*datasets, smiles_column='Smiles', diff_column='diff'):
    deduplicated_datasets = []
    deduplication_stats = []  # To store stats about the deduplication
    
    for df in datasets:
        original_count = len(df)
        
        df['clean_smiles'] = df[smiles_column].apply(remove_salts)
        df_sorted = df.sort_values(by=[diff_column], ascending=True)
        df_dedup = df_sorted.drop_duplicates(subset=['clean_smiles'], keep='first')
        
        deduplicated_count = len(df_dedup)
        duplicates_removed = original_count - deduplicated_count
        
        # Store deduplication stats
        deduplication_stats.append({
            'original_count': original_count,
            'deduplicated_count': deduplicated_count,
            'duplicates_removed': duplicates_removed
        })
        
        df_final = df_dedup.drop(columns=['clean_smiles'])
        deduplicated_datasets.append(df_final)
        
    return deduplicated_datasets, deduplication_stats


def save_dataframes_and_report(base_path, df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report):
    # Define the paths for datasets and reports
    datasets_path = os.path.join(base_path, 'data', 'filtered_datasets')
    reports_path = os.path.join(base_path, 'data', 'reports')
    
    # Check if the paths exist, if not, create them
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    # Define filenames for each dataframe and the report
    filenames = {
        'whole_dataset.csv': df_whole_dataset,
        'actives_dataset.csv': df_whole_act_dataset,
        'inactives_dataset.csv': df_whole_ina_dataset,
        'moderate_actives_dataset.csv': df_data_other_eq_inc,
        'dataset_filtration_report.csv': data_report
    }
    
    # Save each dataframe to its respective file
    for filename, df in filenames.items():
        if 'report' in filename:
            # Save reports in the reports folder
            full_path = os.path.join(reports_path, filename)
        else:
            # Save datasets in the filtered_datasets folder
            full_path = os.path.join(datasets_path, filename)
        
        df.to_csv(full_path, index=True)
    
    print('Dataframes and report saved successfully.')



#general_path = '/home/leonardo/LAB/PhD_works/classifiers_ml_project_04_24'
#df = load_data(os.path.join(general_path, 'data/raw_data/chembl3307570_bioactivity_data.csv'))  # Example file path and ID


#df_process_1 = process_data_1(df)
#df_revised, df_other_revised, df_perc_revised_2, l_other_whole, l_perc_whole = process_data_2(df_process_1)
#df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report = process_data_3(df_revised)


# Assume datasets are processed, and you're about to deduplicate
#datasets, dedup_stats = deduplicate_datasets(
#    df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc,
#    smiles_column='Smiles',
#    diff_column='diff'
#)

# Unpack the deduplicated datasets
#df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc = datasets

# Example of how to update the data_report with deduplication information
#for i, stats in enumerate(dedup_stats, start=1):
#    data_report.loc[f'dataset_{i}_original_count'] = stats['original_count']
#    data_report.loc[f'dataset_{i}_deduplicated_count'] = stats['deduplicated_count']
#    data_report.loc[f'dataset_{i}_duplicates_removed'] = stats['duplicates_removed']

#save_dataframes_and_report(general_path, df_whole_dataset, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report)







