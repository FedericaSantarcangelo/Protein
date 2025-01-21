""" 
@Author: Federica Santarcangelo
"""
import os
import pandas as pd
import numpy as np
from argparse import Namespace, ArgumentParser
from utils.args import reducer_args
from models.plot import plot_tsne, plot_kmeans_clusters, create_cumulative_variance_plot, create_individual_variance_plot, plot_similarity_matrix
from models.plot import save_loading_scores, elbow, silhouette, save_cluster_labels, plot_pls_results
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_handling import select_optimal_clusters
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


def get_parser_args():
    parser = ArgumentParser(description='QSAR Pilot Study')
    reducer_args(parser)
    return parser

class DimensionalityReducer():
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
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
    
    def perform_pls_analysis(self, data, target):
        data = self.scaler.fit_transform(data)
        target = self.scaler.fit_transform(target.to_numpy().reshape(-1, 1)).ravel()
        max_components = min(data.shape[1], len(data) - 1)
        r2_scores = []
        q2_scores = []
        components_range = range(1, max_components + 1)

        for n_components in components_range:
            pls = PLSRegression(n_components=n_components)
            pls.fit(data, target)
            r2 = pls.score(data, target)
            r2_scores.append(r2)

            loo = LeaveOneOut()
            y_pred = np.zeros(target.shape)

            for train_index, test_index in loo.split(data):
                pls.fit(data[train_index], target[train_index])
                y_pred[test_index] = pls.predict(data[test_index]).ravel()

            q2 = r2_score(target, y_pred)
            q2_scores.append(q2)

        plot_pls_results(components_range, r2_scores, q2_scores, self.result_dir)

        optimal_components = self.select_optimal_components(r2_scores, q2_scores, max_components)
        print(f'Optimal number of components: {optimal_components}')
        pls_optimal = PLSRegression(n_components=optimal_components)
        pls_optimal.fit(data, target)
        reduced_data = pls_optimal.transform(data)

        return reduced_data

    def select_optimal_components(self, r2_scores, q2_scores, max_components):
        r2_threshold = 0.95
        q2_threshold_stable = -0.1

        optimal_range = []
        for i in range(1, len(r2_scores) + 1):
            if r2_scores[i-1] >= r2_threshold and (q2_scores[i-1] >= q2_threshold_stable or (i > 1 and q2_scores[i-1] >= q2_scores[i-2])):
                optimal_range.append(i)

        if not optimal_range:
            optimal_range = [i for i in range(1, len(r2_scores) + 1) if r2_scores[i-1] >= r2_threshold]

        optimal_component = optimal_range[-1] if optimal_range else 1
        return min(optimal_component, max_components)

    
    def fit_transform(self, data, log):
        data['ID'] = np.arange(len(data))
        results = {}

        self.scaled_data = self.scaler.fit_transform(data)
        feature_names = data.columns[:-1]
        self.scaled_data, feature_names = remove_zero_variance_features(self.scaled_data, feature_names)
        self.scaled_data, feature_names = remove_highly_correlated_features(self.scaled_data, feature_names)

        self.compute_similarity()
        reduced_data = self.perform_pls_analysis(self.scaled_data, log)
        optimal_components = reduced_data.shape[1]
        self.pca = PCA(n_components=optimal_components)
        self.pca.fit(self.scaled_data)

        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        loading_scores = self.compute_loading_scores(self.pca, feature_names)
        save_loading_scores(loading_scores, 'standard_scaler_loading_scores.csv', self.result_dir)
        create_cumulative_variance_plot(cumulative_variance, self.result_dir, 'standard_scaler')
        create_individual_variance_plot(explained_variance, self.result_dir, 'standard_scaler')

        results['reduced_data'] = reduced_data

        inertia = elbow(reduced_data, 10, self.result_dir, self.args.seed)
        silhouette_scores = silhouette(reduced_data, 10, self.result_dir, self.args.seed)
        optimal_clusters = select_optimal_clusters(inertia, silhouette_scores)

        labels = self.perform_kmeans(reduced_data, optimal_clusters)
        self.perform_tsne(reduced_data, labels)

        save_cluster_labels(data, reduced_data, labels, self.result_dir)

        return results

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