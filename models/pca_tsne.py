""" 
@Author: Federica Santarcangelo
"""
import os
import pandas as pd
import numpy as np

from argparse import Namespace, ArgumentParser
from utils.args import reducer_args
from models.plot import plot_tsne, plot_kmeans_clusters, create_cumulative_variance_plot, create_individual_variance_plot, plot_similarity_matrix
from models.plot import save_loading_scores, elbow, silhouette,save_cluster_labels
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_handling import select_optimal_clusters
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features

def get_parser_args():
    parser = ArgumentParser(description='QSAR Pilot Study')
    reducer_args(parser)
    return parser

class DimensionalityReducer():
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.pca = PCA(n_components=11)
        self.result_dir = self.args.path_pca_tsne
        os.makedirs(self.result_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.similarity_matrix = None

    def compute_similarity(self):
        self.similarity_matrix = self.similarity(self.scaled_data)
        plot_similarity_matrix(self.similarity_matrix, self.args.similarities + '_matrix.png', self.result_dir)
        similarity_df = pd.DataFrame(self.similarity_matrix)
        similarity_path = os.path.join(self.result_dir, self.args.similarities + '_matrix.csv')
        similarity_df.to_csv(similarity_path, index=False)
        return self.similarity_matrix

    def compute_loading_scores(self, pca, feature_names):
        loadings = pca.components_.T
        return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    def fit_transform(self, data, log):
        data['ID'] = np.arange(len(data))
        results = {}

        self.scaled_data = self.scaler.fit_transform(data)
        feature_names = data.columns[:-1]
        self.scaled_data, feature_names = remove_zero_variance_features(self.scaled_data, feature_names)
        self.scaled_data, feature_names = remove_highly_correlated_features(self.scaled_data, feature_names)
        self.compute_similarity()

        self.pca.fit(self.scaled_data)
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        loading_scores = self.compute_loading_scores(self.pca, feature_names)
        save_loading_scores(loading_scores, 'standard_scaler_loading_scores.csv',self.result_dir)
        create_cumulative_variance_plot(cumulative_variance, self.result_dir, 'standard_scaler')
        create_individual_variance_plot(explained_variance, self.result_dir, 'standard_scaler')

        reduced_data = self.pca.transform(self.scaled_data)
        selected_components = self.analyze_pca_correlation(reduced_data, log)
        reduced_data = reduced_data[:, selected_components]
        self.save_reduced_data(reduced_data, data, 'selected_pca_components')
        results['reduced_data'] = reduced_data

        inertia = elbow(reduced_data, 10, self.result_dir, self.args.seed)
        silhouette_scores = silhouette(reduced_data, 10, self.result_dir, self.args.seed)
        optimal_clusters = select_optimal_clusters(inertia, silhouette_scores)
        labels = self.perform_kmeans(reduced_data, optimal_clusters)

        self.perform_tsne(reduced_data, labels)
        save_cluster_labels(data, reduced_data, labels,self.result_dir)

        return results
    
    def analyze_pca_correlation(self, reduced_data, log_activity, threshold=0.25):
        """
        Analyze the correlation between the PCA components and the target activity values.
        Select components with a correlation magnitude above the threshold.
        """
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df['-log_activity'] = log_activity
        correlation_matrix = reduced_df.corr()
        corr_path = os.path.join(self.result_dir, 'pca_correlation_with_log_activity.csv')
        correlation_matrix.to_csv(corr_path)
        target_corr = correlation_matrix['-log_activity'][:-1]
        selected_components = target_corr[abs(target_corr) > threshold].index
        selected_indices = [int(col[2:]) - 1 for col in selected_components] 
        selected_path = os.path.join(self.result_dir, 'selected_pca_components.csv')
        target_corr[selected_components].to_csv(selected_path)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlazione tra componenti PCA e -log dell'attività")
        plt.savefig(os.path.join(self.result_dir, 'pca_correlation_with_log_activity.png'), bbox_inches='tight')
        plt.close()
        return selected_indices

    def save_reduced_data(self, reduced_data, data, scaler_name):
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df = pd.concat([data.reset_index(drop=True), reduced_df], axis=1)
        reduced_data_path = os.path.join(self.result_dir, f'{scaler_name}_reduced_data.csv')
        reduced_df.to_csv(reduced_data_path, index=False)

    def perform_kmeans(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        cluster_label_path = os.path.join(self.result_dir, f'cluster_labels.csv')
        pd.DataFrame(labels, columns=['Cluster']).to_csv(cluster_label_path, index=False)
        centers_path = os.path.join(self.result_dir, f'cluster_centers.csv')
        pd.DataFrame(centers, columns=[f'PC{i+1}' for i in range(centers.shape[1])]).to_csv(centers_path, index=False)
        plot_kmeans_clusters(data, labels, centers, self.result_dir)
        return labels

    def perform_tsne(self, data, labels):
        n_components = min(self.args.n_components_tsne, data.shape[1])
        tsne = TSNE(n_components=n_components, perplexity=self.args.perplexity,
                    learning_rate=self.args.lr_tsne, max_iter=self.args.n_iter, random_state=42)
        tsne_results = tsne.fit_transform(data)
        tsne_df = pd.DataFrame(tsne_results, columns=[f'TSNE{i+1}' for i in range(tsne_results.shape[1])])
        tsne_path = os.path.join(self.result_dir, 'tsne_results.csv')
        tsne_df.to_csv(tsne_path, index=False)
        plot_tsne(tsne_results, labels, self.result_dir)