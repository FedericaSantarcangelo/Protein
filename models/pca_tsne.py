import os
import pandas as pd
import numpy as np

from argparse import Namespace, ArgumentParser
from utils.args import reducer_args

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_handling import select_optimal_clusters
from dataset.processing import remove_highly_correlated_features,remove_zero_variance_features

def get_parser_args():
    parser = ArgumentParser(description='QSAR Pilot Study')
    reducer_args(parser)
    return parser

class DimensionalityReducer():
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.pca = PCA(n_components=3)
        self.result_dir = self.args.path_pca_tsne
        os.makedirs(self.result_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.similarity_matrix = None

    def compute_similarity(self):
        self.similarity_matrix = self.similarity(self.scaled_data)
        self._plot_similarity_matrix(self.similarity_matrix, self.args.similarities + '_matrix.png')
        similarity_df = pd.DataFrame(self.similarity_matrix)
        similarity_path = os.path.join(self.result_dir, self.args.similarities + '_matrix.csv')
        similarity_df.to_csv(similarity_path, index=False)
        return self.similarity_matrix

    def _plot_similarity_matrix(self, matrix, filename):
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, cmap='coolwarm', annot=False)
        plt.savefig(os.path.join(self.result_dir, filename), bbox_inches='tight')
        plt.close()

    def compute_loading_scores(self, pca, feature_names):
        loadings = pca.components_.T
        return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    def fit_transform(self, data,log):
        data['ID'] = np.arange(len(data))
        results = {}

        self.scaled_data = self.scaler.fit_transform(data)

        feature_names = data.columns[:-1]  # Exclude 'ID' column

        self.scaled_data, feature_names = remove_zero_variance_features(self.scaled_data, feature_names)
        self.scaled_data, feature_names = remove_highly_correlated_features(self.scaled_data, feature_names)

        self.compute_similarity()

        self.pca.fit(self.scaled_data)
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        loading_scores = self.compute_loading_scores(self.pca, feature_names)
        loading_scores_file = os.path.join(self.result_dir, 'standard_scaler_loading_scores.csv')
        loading_scores.to_csv(loading_scores_file, index=True)

        self.create_cumulative_variance_plot(cumulative_variance, self.result_dir, 'standard_scaler')
        self.create_individual_variance_plot(explained_variance, self.result_dir, 'standard_scaler')

        components_needed = len(explained_variance)
        pca_reduced = PCA(n_components=components_needed)
        reduced_data = pca_reduced.fit_transform(self.scaled_data)

        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df['ID'] = data['ID'].values

        self.analyze_pca_correlation(reduced_data, log)

        self.save_reduced_data(reduced_data, data, 'standard_scaler')
        results['reduced_data'] = reduced_data

        inertia = self.elbow(reduced_data)
        silhouette_scores = self.silhouette(reduced_data)
        optimal_clusters = select_optimal_clusters(inertia, silhouette_scores)

        labels = self.perform_kmeans(reduced_data, optimal_clusters)
        self.perform_tsne(reduced_data, labels)

        id_cluster_df = pd.DataFrame({'ID': reduced_df['ID'], 'Cluster': labels})
        id_cluster_path = os.path.join(self.result_dir, 'original_data_with_labels.csv')
        id_cluster_df.to_csv(id_cluster_path, index=False)

        return results
    
    def analyze_pca_correlation(self, reduced_data, log_activity, threshold=3):
        """
            Analyze the correlation between the PCA components and the activity values
        """

        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df['-log_activity'] = log_activity
        
        correlation_matrix = reduced_df.corr()
        corr_path = os.path.join(self.result_dir, 'pca_correlation_with_log_activity.csv')
        correlation_matrix.to_csv(corr_path)

        residuals = reduced_df['-log_activity'] - reduced_df.drop(columns=['-log_activity']).dot(
            correlation_matrix.loc['-log_activity'].iloc[:-1]
        )
        residuals_zscore = (residuals - residuals.mean()) / residuals.std()
        outliers = residuals_zscore.abs() > threshold

        outlier_df = reduced_df[outliers]
        outlier_path = os.path.join(self.result_dir, 'pca_outliers_based_on_log_activity.csv')
        outlier_df.to_csv(outlier_path, index=False)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlazione tra componenti PCA e -log dell'attività")
        plt.savefig(os.path.join(self.result_dir, 'pca_correlation_with_log_activity.png'), bbox_inches='tight')
        plt.close()

    def elbow(self, data, max_k=10):
        inertia = []
        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k, random_state=self.args.seed)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
        print(f"Elbow Method: {inertia}")
        plt.plot(range(1, max_k), inertia, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.savefig(os.path.join(self.result_dir, 'elbow.png'), bbox_inches='tight')
        plt.close()
        return inertia

    def silhouette(self, data, max_k=10):
        silhouette_scores = []
        for k in range(2, max_k):
            kmeans = KMeans(n_clusters=k, random_state=self.args.seed)
            kmeans.fit(data)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        print(f"Silhouette Score: {silhouette_scores}")
        plt.plot(range(2, max_k), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.savefig(os.path.join(self.result_dir, 'silhouette.png'), bbox_inches='tight')
        plt.close()
        return silhouette_scores

    def create_cumulative_variance_plot(self, cumulative_variance, result_dir, scaler_name):
        plt.figure(figsize=(10, 10))
        plt.plot(cumulative_variance, marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title(f'Cumulative explained variance for {scaler_name}')
        plt.savefig(os.path.join(result_dir, f'{scaler_name}_cumulative_variance.png'), bbox_inches='tight')
        plt.close()

    def create_individual_variance_plot(self, explained_variance, result_dir, scaler_name):
        plt.figure(figsize=(10, 10))
        plt.plot(explained_variance, marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.title(f'Explained variance for {scaler_name}')
        plt.savefig(os.path.join(result_dir, f'{scaler_name}_explained_variance.png'), bbox_inches='tight')
        plt.close()

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

        self.plot_kmeans_clusters(data, labels, centers)
        return labels

    def plot_kmeans_clusters(self, data, labels, centers):
        plt.figure(figsize=(10, 10))
        if data.shape[1] > 1:
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        else:
            plt.scatter(data[:, 0], np.zeros_like(data[:, 0]), c=labels, cmap='viridis')
            plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), c='red', marker='x')
            plt.xlabel('PC1')
            plt.ylabel('Constant')
        plt.title(f'KMeans Clusters )')
        plt.savefig(os.path.join(self.result_dir, f'kmeans_clusters.png'))
        plt.close()

    def perform_tsne(self, data, labels):
        n_components = min(self.args.n_components_tsne, data.shape[1])
        tsne = TSNE(n_components=n_components, perplexity=self.args.perplexity,
                    learning_rate=self.args.lr_tsne, max_iter=self.args.n_iter, random_state=42)
        tsne_results = tsne.fit_transform(data)

        tsne_df = pd.DataFrame(tsne_results, columns=[f'TSNE{i+1}' for i in range(tsne_results.shape[1])])
        tsne_path = os.path.join(self.result_dir, 'tsne_results.csv')
        tsne_df.to_csv(tsne_path, index=False)

        self.plot_tsne(tsne_results, labels)

    def plot_tsne(self, tsne_results, labels):
        plt.figure(figsize=(10, 10))
        if tsne_results.shape[1] > 1:
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10)
            plt.xlabel('TSNE1')
            plt.ylabel('TSNE2')
        else:
            scatter = plt.scatter(tsne_results[:, 0], np.zeros_like(tsne_results[:, 0]), c=labels, cmap='viridis', s=10)
            plt.xlabel('TSNE1')
            plt.ylabel('Constant')
        plt.title(f't-SNE Results)')
        plt.colorbar(scatter)
        plt.savefig(os.path.join(self.result_dir,'tsne_results.png'))
        plt.close()