import os
import pandas as pd
import numpy as np
from argparse import Namespace
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.data_handling import select_optimal_clusters
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features
from models.qsar_models import QSARModelTrainer 
from sklearn.model_selection import train_test_split
from models.plot import plot_similarity_matrix, save_loading_scores, create_cumulative_variance_plot, create_individual_variance_plot
from models.plot import elbow, silhouette, plot_kmeans_clusters, plot_tsne, save_cluster_labels, find_intersection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

class DimensionalityReducer:
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.result_dir = self.args.path_pca_tsne
        os.makedirs(self.result_dir, exist_ok=True)
        self.scaler = QuantileTransformer()
        self.similarity_matrix = None

    def compute_similarity(self, scaled_data):
        """Compute and plot the similarity matrix."""
        self.similarity_matrix = self.similarity(scaled_data)
        plot_similarity_matrix(self.similarity_matrix, self.args.similarities + '_matrix.png', self.result_dir)
        similarity_df = pd.DataFrame(self.similarity_matrix)
        similarity_path = os.path.join(self.result_dir, self.args.similarities + '_matrix.csv')
        similarity_df.to_csv(similarity_path, index=False)
        return self.similarity_matrix

    def compute_loading_scores(self, pca, feature_names):
        """Compute the loading scores for PCA components."""
        loadings = pca.components_.T
        return pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    def fit_transform(self, X_train, y_train, X_test, y_test):
        """Fit and transform the data using PCA, regression, clustering, and t-SNE."""

        X_train_f, Y_train_f, X_test_f, Y_test_f = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train_f['ID'] = np.arange(len(X_train_f))
        Y_train_f['ID'] = np.arange(len(Y_train_f))
        scaled_X_train_f = self.scaler.fit_transform(X_train_f)
        scaled_Y_train_f = self.scaler.transform(Y_train_f)
        #print("Train 41- Media", np.mean(scaled_X_train_f), "Deviazione standard", np.std(scaled_X_train_f))
        #print("Test 11 train - Media", np.mean(scaled_Y_train_f), "Deviazione standard", np.std(scaled_Y_train_f))
        feature_X_train_f = X_train_f.columns[:-1] 
        feature_Y_train_f = Y_train_f.columns[:-1]  
        #scaled_X_train_f, feature_X_train_f = remove_zero_variance_features(scaled_X_train_f, feature_X_train_f)
        #scaled_Y_train_f, feature_Y_train_f = remove_zero_variance_features(scaled_Y_train_f, feature_Y_train_f)
        #common_features = np.intersect1d(feature_X_train_f, feature_Y_train_f)
        #feature_mask_X_train_f = np.isin(feature_X_train_f, common_features)
        #feature_mask_Y_train_f = np.isin(feature_Y_train_f, common_features)
        #scaled_X_train_f = scaled_X_train_f[:, feature_mask_X_train_f]
        #scaled_Y_train_f = scaled_Y_train_f[:, feature_mask_Y_train_f]
        #feature_X_train_f = feature_X_train_f[feature_mask_X_train_f]
        #feature_Y_train_f = feature_Y_train_f[feature_mask_Y_train_f]
        #scaled_X_train_f, feature_X_train_f = remove_highly_correlated_features(scaled_X_train_f, feature_X_train_f)
        #scaled_Y_train_f, feature_Y_train_f = remove_highly_correlated_features(scaled_Y_train_f, feature_Y_train_f)
        #common_features = np.intersect1d(feature_X_train_f, feature_Y_train_f)
        #feature_mask_X_train_f = np.isin(feature_X_train_f, common_features)
        #feature_mask_Y_train_f = np.isin(feature_Y_train_f, common_features)
        #scaled_X_train_f = scaled_X_train_f[:, feature_mask_X_train_f]
        #scaled_Y_train_f = scaled_Y_train_f[:, feature_mask_Y_train_f]
        #feature_X_train_f = np.array(feature_X_train_f)[feature_mask_X_train_f]
        #feature_Y_train_f = np.array(feature_Y_train_f)[feature_mask_Y_train_f]
        np.save("/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results/selected_features.npy", feature_X_train_f)
        self.compute_similarity(scaled_X_train_f)        
        #initial_PCA_components = min(scaled_X_train_f.shape[1], len(scaled_X_train_f) - 1)
        #self.pca = PCA(n_components=initial_PCA_components)
        #pca_scaled_X_train_f = self.pca.fit_transform(scaled_X_train_f)
        #pca_scaled_Y_train_f = self.pca.transform(scaled_Y_train_f)
        #explained_variance_ratio = self.pca.explained_variance_ratio_
        #cumulative_variance = np.cumsum(explained_variance_ratio)
        #optimal_pca_components = np.argmax(cumulative_variance > 0.99) + 1
        #pca_scaled_X_train_f = pca_scaled_X_train_f[:, :optimal_pca_components]
        #pca_scaled_Y_train_f = pca_scaled_Y_train_f[:, :optimal_pca_components]
        #loading_score = self.compute_loading_scores(self.pca, feature_X_train_f)
        #save_loading_scores(loading_score, 'pca_scaled_X_train_f_loading_score.csv' ,self.result_dir)
        #create_cumulative_variance_plot(cumulative_variance, 'cumulative_variance.png', self.result_dir)
        #create_individual_variance_plot(explained_variance_ratio, 'individual_variance.png', self.result_dir)
        #inertia = elbow(pca_scaled_X_train_f, optimal_pca_components ,self.result_dir, self.args.seed)
        #silhouette_score = silhouette(pca_scaled_X_train_f, optimal_pca_components, self.result_dir, self.args.seed)
        #optimal_cluster = select_optimal_clusters(inertia, silhouette_score)
        #labels = self.perform_kmeans(pca_scaled_X_train_f, optimal_cluster)
        #self.perform_tsne(pca_scaled_X_train_f, labels)
        #save_cluster_labels(X_train_f, pca_scaled_X_train_f, labels, self.result_dir)
        trainer = QSARModelTrainer(self.args)
        #for component in range(1, optimal_pca_components + 1):
            #reduced_data_X_train_f = pca_scaled_X_train_f[:, :component]
            #reduced_data_Y_train_f = pca_scaled_Y_train_f[:, :component]
            #trainer.train_and_evaluate(reduced_data_X_train_f, X_test_f, reduced_data_Y_train_f, Y_test_f, component)
        #trainer.select_best_model()
        scaled_X, scaled_y = self.allign(X_train, X_test)

        #trainer.retrain_best_model(pca_scaled_X, y_train, pca_scaled_y, y_test)
        #trainer.test_model(pca_scaled_y, y_test)
        trainer.train_and_evaluate(scaled_X_train_f, X_test_f, scaled_Y_train_f, Y_test_f)
        trainer.select_best_model()
        trainer.retrain_best_model(scaled_X, y_train, scaled_y, y_test)
        trainer.test_model(scaled_y, y_test)

    def allign(self,X_train, X_test):
        """ Allign train and test set for retrain models"""
        
        #selected_features = np.load("/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results/selected_features.npy")
        X_train_s = X_train.copy()
        X_test_s = X_test.copy()
        X_train_s['ID'] = np.arange(len(X_train_s))
        X_test_s['ID'] = np.arange(len(X_test_s))
        scaled_X_train = self.scaler.transform(X_train_s)
        scaled_X_test = self.scaler.transform(X_test_s)

        #print("Train 52- Media", np.mean(scaled_X_train), "Deviazione standard", np.std(scaled_X_train))
        #print("Test 41 - Media", np.mean(scaled_X_test), "Deviazione standard", np.std(scaled_X_test))

        #feature_names = np.array(X_train_s.columns)
        #feature_mask = np.isin(feature_names, selected_features) 
        #scaled_X_train = scaled_X_train[:, feature_mask]
        #scaled_X_test = scaled_X_test[:, feature_mask]
        #pca_scaled_X_train = self.pca.transform(scaled_X_train)
        #pca_scaled_X_test = self.pca.transform(scaled_X_test)    
        return scaled_X_train, scaled_X_test

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
