import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
from argparse import Namespace, ArgumentParser
from utils.args import reducer_args
from sklearn.metrics import silhouette_score

def get_parser_args():
    parser = ArgumentParser(description='QSAR Pilot Study')
    reducer_args(parser)
    return parser

class DimensionalityReducer():
    def __init__(self, args: Namespace):
        self.args = args
        self.similarity = cosine_similarity if self.args.similarities == 'cosine' else euclidean_distances
        self.pca = PCA(n_components=self.args.n_components_pca, svd_solver='auto', random_state=self.args.seed)
        self.tsne = TSNE(n_components=self.args.n_components_tsne,
                        perplexity=self.args.perplexity, learning_rate=self.args.lr_tsne,
                        max_iter=self.args.n_iter,
                        random_state=self.args.seed)
        self.result_dir = self.args.path_pca_tsne
        n_clusters = self.args.n_clusters if hasattr(self.args, 'n_clusters') else 5
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed)
        os.makedirs(self.result_dir, exist_ok=True)

    def fit_transform(self, data):
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data)

        self.pca_data = self.pca.fit_transform(self.scaled_data)
        self.explained_variance = self.pca.explained_variance_ratio_

        self.tsne_data = self.tsne.fit_transform(self.scaled_data)
        self.kmeans_data = self.kmeans.fit_predict(self.scaled_data)
        return self.pca_data, self.tsne_data, self.kmeans_data
    
    def comupute_similarity(self, data):
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

    def plot_pca_variance(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.explained_variance) + 1), self.explained_variance, alpha=0.7, align='center')
        plt.title('Explained Variance by PCA Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.savefig(os.path.join(self.result_dir, 'pca_variance.png'), bbox_inches='tight')
        plt.close()

    def plot_results(self):
        if self.pca_data is None or self.tsne_data is None :
            raise ValueError('Data not fitted yet!')
        
        cluster_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'yellow', 4: 'red'}
        cluster_labels = self.kmeans_data

        results_df = pd.DataFrame({
        'PCA1': self.pca_data[:, 0],
        'PCA2': self.pca_data[:, 1],
        't-SNE1': self.tsne_data[:, 0],
        't-SNE2': self.tsne_data[:, 1]
        })
        results_path = os.path.join(self.result_dir, 'dimensionality_reduction_results.csv')
        results_df.to_csv(results_path, index=False)
        
        plt.figure(figsize=(16,14))
        plt.scatter(self.pca_data[:,0], self.pca_data[:,1], alpha=0.7, 
                                  c=[cluster_colors[label] for label in cluster_labels], s=50)
        plt.title('PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for color in cluster_colors.values()]
        labels = [f'Cluster {i}' for i in cluster_colors.keys()]
        plt.legend(handles, labels, title="Clusters", loc='upper left')
        plt.savefig(os.path.join(self.result_dir, 'pca.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(16,14))
        plt.scatter(self.tsne_data[:,0], self.tsne_data[:,1], alpha=0.7, 
                                  c=[cluster_colors[label] for label in cluster_labels], s=50)
        plt.title('t-SNE')
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.savefig(os.path.join(self.result_dir, 'tsne.png'), bbox_inches='tight')
        plt.close()

    def save_results(self, filename="dimensionality_results.csv"):
        
        if not hasattr(self, 'kmeans_data') or not hasattr(self, 'pca_data') or not hasattr(self, 'tsne_data'):
            raise ValueError("I dati non sono stati calcolati. Assicurati di aver eseguito fit_transform prima di salvare i risultati.")
        
        results_df = pd.DataFrame(
            {
                'Cluster': self.kmeans_data,
                'PCA1': self.pca_data[:, 0],
                'PCA2': self.pca_data[:, 1],
                't-SNE1': self.tsne_data[:, 0],
                't-SNE2': self.tsne_data[:, 1]
            }
        )

        results_df['Silhouette_Score_Original'] = silhouette_score(self.scaled_data, self.kmeans_data)
        results_df['Silhouette_Score_PCA'] = silhouette_score(self.pca_data, self.kmeans_data)
        results_df['Silhouette_Score_tSNE'] = silhouette_score(self.tsne_data, self.kmeans_data)


        results_path = os.path.join(self.result_dir, filename)
        if not os.path.isdir(self.result_dir):
            raise FileNotFoundError(f"La directory {self.result_dir} non esiste")

        results_df.to_csv(results_path, index=False)
