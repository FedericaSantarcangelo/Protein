#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:49:41 2024

@author: leonardo
"""

# src/data_preprocessor.py

from src.data_preparation import load_data, process_data_1, process_data_2, process_data_3, deduplicate_datasets, save_dataframes_and_report
from src.config import RAW_DATA_PATH, GENERAL_PATH

def preprocess_and_save_datasets():
    df = load_data(RAW_DATA_PATH)
    df_processed1 = process_data_1(df)
    df_processed2, df_other_revised, df_perc_revised_2, l_other_whole, l_perc_whole = process_data_2(df_processed1)
    df_processed3_whole, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report = process_data_3(df_processed2)

    # Deduplicate datasets and get deduplication stats
    datasets, dedup_stats = deduplicate_datasets(
        df_processed3_whole, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc,
        smiles_column='Smiles',
        diff_column='diff'
    )

    # Update datasets after deduplication
    df_processed3_whole, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc = datasets

    # Update the data_report with deduplication information
    for i, stats in enumerate(dedup_stats, start=1):
        data_report.loc[f'dataset_{i}_original_count'] = stats['original_count']
        data_report.loc[f'dataset_{i}_deduplicated_count'] = stats['deduplicated_count']
        data_report.loc[f'dataset_{i}_duplicates_removed'] = stats['duplicates_removed']

    # Save the deduplicated datasets and updated data report
    save_dataframes_and_report(GENERAL_PATH, df_processed3_whole, df_whole_act_dataset, df_whole_ina_dataset, df_data_other_eq_inc, data_report)
    print("Data preprocessing and saving completed.")
