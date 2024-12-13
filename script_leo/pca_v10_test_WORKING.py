#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:42:51 2024

@author: leonardo
"""


import pandas as pd
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
)
import matplotlib.pyplot as plt

# Function to replace large values and infinities with NaN
def handle_large_values(data, threshold=1e10):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' else x)
    data = data.applymap(lambda x: np.nan if isinstance(x, (int, float)) and abs(x) > threshold else x)
    return data

# Preprocessing function
def preprocess_data(data, drop_columns=None, fill_value=0):
    if drop_columns:
        metadata = data[drop_columns]
        data = data.drop(columns=drop_columns, errors='ignore')
    else:
        metadata = pd.DataFrame()
    data = handle_large_values(data)
    return data.fillna(fill_value).select_dtypes(include=[np.number]), metadata

# Function to compute PCA loading scores
def compute_loading_scores(pca, feature_names):
    loadings = pca.components_.T
    return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

# Function to save reduced datasets
def save_reduced_dataset(threshold, reduced_data, metadata, file_name, scaler_name, output_folder):
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(reduced_data.shape[1])])
    # Re-add metadata
    reduced_df = pd.concat([metadata.reset_index(drop=True), reduced_df], axis=1)
    output_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_{int(threshold * 100)}pct.csv")
    reduced_df.to_csv(output_file, index=False)
    print(f"Saved reduced dataset for {threshold * 100}% variance to {output_file}")

# Function to create individual explained variance plot
def create_individual_explained_variance_plot(explained_variance_ratio, cumulative_variance, output_file):
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    plot_y = [val * 100 for val in explained_variance_ratio[:n_components_95]]
    plot_x = np.arange(1, len(plot_y) * 2, step=2)  # Increase spacing between bars

    plt.figure(figsize=(14, 8))
    bars = plt.bar(plot_x, plot_y, width=1.0, align="center", color="#1C3041", edgecolor="#000000", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom")
    plt.xlabel("Principal Component")
    plt.ylabel("Percentage of Explained Variance")
    plt.title("Variance Explained per Principal Component", loc="left", fontdict={"weight": "bold"}, y=1.06)
    plt.grid(axis="y")
    plt.xticks(plot_x, [f"PC{i}" for i in range(1, n_components_95 + 1)], rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Function to create cumulative variance plot
def create_cumulative_variance_plot(cumulative_variance, thresholds, output_file):
    n_components_thresholds = {threshold: np.argmax(cumulative_variance >= threshold) + 1 for threshold in thresholds}

    plt.figure(figsize=(12, 8))
    for threshold, n_components in n_components_thresholds.items():
        plt.axhline(y=threshold, linestyle="--", label=f"{int(threshold * 100)}% Variance", alpha=0.7)
        plt.scatter([n_components], [threshold], color="red", zorder=5)
        plt.text(n_components + 10, threshold - 0.02, f"{n_components} PCs", color="red")

    plt.plot(cumulative_variance, label="Cumulative Explained Variance", linewidth=2, color="blue")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance", loc="left", fontdict={"weight": "bold"}, y=1.06)
    plt.grid(axis="x")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Function to save explained variance per component as CSV
def save_individual_variance_csv(explained_variance_ratio, file_name, scaler_name, output_folder):
    explained_variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance_ratio))],
        "Explained Variance (%)": [val * 100 for val in explained_variance_ratio]
    })
    variance_csv_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_explained_variance.csv")
    explained_variance_df.to_csv(variance_csv_file, index=False)
    print(f"Saved explained variance as CSV to {variance_csv_file}")

# Function to save general variance report
def save_general_variance_report(general_results, output_folder):
    variance_report = []

    for file_name, scalers in general_results.items():
        for scaler_name, results in scalers.items():
            for threshold, components in results.items():
                variance_report.append([file_name, scaler_name, threshold, components])

    df_report = pd.DataFrame(variance_report, columns=["Dataset", "Scaler", "Threshold", "Components"])
    output_file = os.path.join(output_folder, "general_variance_report.csv")
    df_report.to_csv(output_file, index=False)
    print(f"Saved general variance report to {output_file}")


# Function to process a single file
def process_file(file_path, scalers, thresholds, output_folder, drop_columns=None, fill_value=0):
    print(f"\nProcessing file: {file_path}")
    data = pd.read_csv(file_path, low_memory=False)
    numeric_data, metadata = preprocess_data(data, drop_columns, fill_value)
    results = {}
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    for scaler_name, scaler in scalers.items():
        print(f"Using {scaler_name}")
        scaled_data = scaler.fit_transform(numeric_data)
        scaler_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_scaler.pkl")
        joblib.dump(scaler, scaler_file)
        print(f"Saved scaler to {scaler_file}")

        pca = PCA()
        pca.fit(scaled_data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        explained_variance_ratio = pca.explained_variance_ratio_

        # Save PCA loading scores
        loading_scores = compute_loading_scores(pca, numeric_data.columns)
        loading_scores_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_pca_loadings.csv")
        loading_scores.to_csv(loading_scores_file)
        print(f"Saved PCA loading scores to {loading_scores_file}")

        # Save explained variance as CSV
        save_individual_variance_csv(explained_variance_ratio, file_name, scaler_name, output_folder)

        # Generate plots
        cumulative_plot_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_cumulative_variance.png")
        create_cumulative_variance_plot(cumulative_variance, thresholds, cumulative_plot_file)

        individual_plot_file = os.path.join(output_folder, f"{file_name}_{scaler_name}_individual_variance.png")
        create_individual_explained_variance_plot(explained_variance_ratio, cumulative_variance, individual_plot_file)

        # Save reduced datasets for each threshold
        components_needed = {threshold: np.argmax(cumulative_variance >= threshold) + 1 for threshold in thresholds}
        for threshold, n_components in components_needed.items():
            pca_reduced = PCA(n_components=n_components)
            reduced_data = pca_reduced.fit_transform(scaled_data)
            save_reduced_dataset(threshold, reduced_data, metadata, file_name, scaler_name, output_folder)

        results[scaler_name] = components_needed

    return results

# Function to create mosaic plot
def create_mosaic_plot(general_results, thresholds, output_file):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, threshold in enumerate(thresholds):
        dataset_components = []
        for file_name, scalers in general_results.items():
            for scaler_name, results in scalers.items():
                if threshold in results:
                    dataset_components.append((file_name, results[threshold]))

        if dataset_components:
            datasets, components = zip(*dataset_components)
        else:
            datasets, components = [], []

        axes[i].barh(datasets, components, color="#4CAF50")
        axes[i].set_title(f"Components Needed for {int(threshold * 100)}% Variance")
        axes[i].set_xlabel("Number of Components")
        axes[i].set_ylabel("Dataset")
        axes[i].grid(axis="x")
        axes[i].set_xlim(0, 200)  # Set x-axis range for all subplots

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved mosaic plot to {output_file}")

# Main analysis loop remains the same
def analyze_folder(folder_path, output_folder, selected_scalers=None, drop_columns=None, fill_value=0):
    os.makedirs(output_folder, exist_ok=True)
    thresholds = [0.8, 0.85, 0.9, 0.95]
    available_scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "MaxAbsScaler": MaxAbsScaler(),
        "Normalizer": Normalizer(),
        "QuantileTransformer": QuantileTransformer(),
        "PowerTransformer": PowerTransformer()
    }
    scalers = {name: available_scalers[name] for name in selected_scalers} if selected_scalers else available_scalers
    general_results = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            results = process_file(file_path, scalers, thresholds, output_folder, drop_columns, fill_value)
            general_results[file_name] = results

    # Save general variance report
    save_general_variance_report(general_results, output_folder)

    # Create mosaic plot
    mosaic_plot_file = os.path.join(output_folder, "general_mosaic_plot_components_needed.png")
    create_mosaic_plot(general_results, thresholds, mosaic_plot_file)


# Specify the folder containing the files and the output folder
input_folder = '/home/leonardo/LAB/pca_analysis_datasets/all_cell_lines'
output_folder = '/home/leonardo/LAB/pca_analysis_datasets/all_cell_lines/output_v10'

# Columns to drop and fill value
columns_to_drop = ['Molecule ChEMBL ID', 'Smiles', 'class']
fill_value = 0
selected_scalers = ["StandardScaler"]

analyze_folder(input_folder, output_folder, selected_scalers, drop_columns=columns_to_drop, fill_value=fill_value)
