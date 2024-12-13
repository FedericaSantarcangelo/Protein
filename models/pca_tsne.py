import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
)
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
from argparse import Namespace, ArgumentParser

from utils.args import reducer_args
from utils.data_handling import elbow, silhouette

def get_parser_args():
    parser = ArgumentParser(description='QSAR Pilot Study')
    reducer_args(parser)
    return parser

class DimensionalityReducer():
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.pca=PCA()
        self.result_dir = self.args.path_pca_tsne
        os.makedirs(self.result_dir, exist_ok=True)
        self.best_scaler = None
    
    def comupute_similarity(self):
        self.similarity_matrix = self.similarity(self.scaled_data)
        self._plot_similarity_matrix(self.similarity_matrix, self.args.similarities + '_matrix.png')
        similarity_df = pd.DataFrame(self.similarity_matrix)
        similarity_path = os.path.join(self.result_dir, self.args.similarities + '_matrix.csv')
        similarity_df.to_csv(similarity_path, index=False)
        return self.similarity_matrix
    
    def _plot_similarity_matrix(self, matrix, filename):
        plt.figure(figsize=(10,10))
        sns.heatmap(matrix, cmap='coolwarm', annot=False)
        plt.savefig(os.path.join(self.result_dir, filename), bbox_inches='tight')
        plt.close()

    def coumpute_loading_scores(self, pca, feature_names):
        loadings = pca.components_.T
        return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
    
    def select_best_scaler(self, data):
        available_scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler(),
            "Normalizer": Normalizer(),
            "QuantileTransformer": QuantileTransformer(),
            "PowerTransformer": PowerTransformer()
        }
        scalers = {name: available_scalers[name] for name in self.args.scaler} if self.args.scaler else available_scalers

        best_variance = 0
        best_scaler_name = None

        for scaler_name, scaler in scalers.items():
            scaled_data = scaler.fit_transform(data)
            self.pca.fit(scaled_data)
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            
            n_components = np.argmax(cumulative_variance >= 0.80) + 1
            if cumulative_variance[n_components - 1] > best_variance:
                best_variance = cumulative_variance[n_components - 1]
                best_scaler_name = scaler_name

        self.best_scaler = scalers[best_scaler_name]

    def fit_transform(self, data):
        results = {}
        if self.best_scaler is None:
            self.select_best_scaler(data)

        self.scaled_data = self.best_scaler.fit_transform(data)
        self.comupute_similarity()
        self.pca.fit(self.scaled_data)

        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        explained_variance = self.pca.explained_variance_ratio_

        loading_scores = self.coumpute_loading_scores(self.pca, data.columns)
        loading_scores_file = os.path.join(self.result_dir, f'best_scaler_loading_scores.csv')
        loading_scores.to_csv(loading_scores_file, index=True)

        self.create_comulative_variance_plot(cumulative_variance, self.result_dir, 'best_scaler')
        self.create_individual_variance_plot(explained_variance, self.result_dir, 'best_scaler')

        components_needed = {threshold: np.argmax(cumulative_variance > threshold) + 1 for threshold in [0.7,0.75,0.8]}
        for threshold, components in components_needed.items():
            pca_reduced = PCA(n_components=components)
            reduced_data = pca_reduced.fit_transform(self.scaled_data)
            self.save_reduced_data(reduced_data, data, 'best_scaler', threshold)
            results[threshold] = reduced_data
        return results

    def create_comulative_variance_plot(self,comulative_variance, result_dir, scaler_name):
        plt.figure(figsize=(10,10))
        plt.plot(comulative_variance, marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title(f'Cumulative explained variance for {scaler_name}')
        plt.savefig(os.path.join(result_dir, f'{scaler_name}_cumulative_variance.png'), bbox_inches='tight')
        plt.close()

    def create_individual_variance_plot(self,explained_variance, result_dir, scaler_name):
        plt.figure(figsize=(10,10))
        plt.plot(explained_variance, marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.title(f'Explained variance for {scaler_name}')
        plt.savefig(os.path.join(result_dir, f'{scaler_name}_explained_variance.png'), bbox_inches='tight')
        plt.close()

    def save_reduced_data(self, reduced_data, data,scaler_name, threshold):
        reduced_df = pd.DataFrame(reduced_data,columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df = pd.concat([data.reset_index(drop=True),reduced_df], axis=1)
        reduced_data_path = os.path.join(self.result_dir, f'{scaler_name}_reduced_data_{threshold}.csv')
        reduced_df.to_csv(reduced_data_path, index=False)
    