import os
import pandas as pd
import numpy as np
from argparse import Namespace
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.data_handling import select_optimal_clusters
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features
from models.qsar_models import QSARModelTrainer 
from models.plot import plot_similarity_matrix, save_loading_scores, create_cumulative_variance_plot, create_individual_variance_plot
from models.plot import elbow, silhouette, plot_kmeans_clusters, plot_tsne, save_cluster_labels, find_intersection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReducer:
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.result_dir = self.args.path_pca_tsne
        os.makedirs(self.result_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.similarity_matrix = None

    def compute_similarity(self):
        """Compute and plot the similarity matrix."""
        self.similarity_matrix = self.similarity(self.scaled_data)
        plot_similarity_matrix(self.similarity_matrix, self.args.similarities + '_matrix.png', self.result_dir)
        similarity_df = pd.DataFrame(self.similarity_matrix)
        similarity_path = os.path.join(self.result_dir, self.args.similarities + '_matrix.csv')
        similarity_df.to_csv(similarity_path, index=False)
        return self.similarity_matrix

    def compute_loading_scores(self, pca, feature_names):
        """Compute the loading scores for PCA components."""
        loadings = pca.components_.T
        return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    def perform_regression(self, data, target, component):
        """Perform regression using multiple models and return the reduced data."""
        trainer = QSARModelTrainer(self.args)
        trainer.train_and_evaluate(data, target, component)
    
    def fit_transform(self, data, log):
        """Fit and transform the data using PCA, regression, clustering, and t-SNE."""
        data['ID'] = np.arange(len(data))

        self.scaled_data = self.scaler.fit_transform(data)
        feature_names = data.columns[:-1]
        self.scaled_data, feature_names = remove_zero_variance_features(self.scaled_data, feature_names)
        self.scaled_data, feature_names = remove_highly_correlated_features(self.scaled_data, feature_names)
        self.compute_similarity()

        initial_PCA_components = min(self.scaled_data.shape[1], len(self.scaled_data) - 1)
        self.pca = PCA(n_components=initial_PCA_components)
        pca_transformed = self.pca.fit_transform(self.scaled_data)

        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        optimal_pca_components = np.argmax(cumulative_variance > 0.75) + 1
        pca_transformed = pca_transformed[:, :optimal_pca_components]
        loading_scores = self.compute_loading_scores(self.pca, feature_names)

        save_loading_scores(loading_scores, 'standard_scaler_loading_scores.csv', self.result_dir)
        create_cumulative_variance_plot(cumulative_variance, self.result_dir, 'standard_scaler')
        create_individual_variance_plot(explained_variance, self.result_dir, 'standard_scaler')

        inertia = elbow(pca_transformed, optimal_pca_components, self.result_dir, self.args.seed)
        silhouette_scores = silhouette(pca_transformed, optimal_pca_components, self.result_dir, self.args.seed)
        optimal_clusters = select_optimal_clusters(inertia, silhouette_scores)
        labels = self.perform_kmeans(pca_transformed, optimal_clusters)
        self.perform_tsne(pca_transformed, labels)
        save_cluster_labels(data, pca_transformed, labels, self.result_dir)

        for component in range(1,optimal_pca_components+1):
            reduced_data = pca_transformed[:, :component]
            self.perform_regression(reduced_data, log, component)


    def save_reduced_data(self, reduced_data, data, scaler_name):
        """Save the reduced data to a CSV file."""
        reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
        reduced_df = pd.concat([data.reset_index(drop=True), reduced_df], axis=1)
        reduced_data_path = os.path.join(self.result_dir, f'{scaler_name}_reduced_data.csv')
        reduced_df.to_csv(reduced_data_path, index=False)

    def perform_kmeans(self, data, n_clusters):
        """Perform KMeans clustering and save the results."""
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
        """Perform t-SNE and save the results."""
        n_components = min(self.args.n_components_tsne, data.shape[1])
        tsne = TSNE(n_components=n_components, perplexity=self.args.perplexity,
                    learning_rate=self.args.lr_tsne, max_iter=self.args.n_iter, random_state=self.args.seed)
        tsne_results = tsne.fit_transform(data)
        tsne_df = pd.DataFrame(tsne_results, columns=[f'TSNE{i+1}' for i in range(tsne_results.shape[1])])
        tsne_path = os.path.join(self.result_dir, 'tsne_results.csv')
        tsne_df.to_csv(tsne_path, index=False)
        plot_tsne(tsne_results, labels, self.result_dir)
