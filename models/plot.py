"""
@Author: Federica Santarcangelo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_tsne(tsne_results, labels, result_dir):
    plt.figure(figsize=(10, 10))
    if tsne_results.shape[1] > 1:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10)
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
    else:
        scatter = plt.scatter(tsne_results[:, 0], np.zeros_like(tsne_results[:, 0]), c=labels, cmap='viridis', s=10)
        plt.xlabel('TSNE1')
        plt.ylabel('Constant')
    plt.title('t-SNE Results')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'tsne_results.png'))
    plt.close()

def plot_kmeans_clusters(data, labels, centers, result_dir):
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
    plt.title('KMeans Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'kmeans_clusters.png'))
    plt.close()

def elbow(data, max_k, result_dir, seed):
    inertia = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    print(f"Elbow Method: {inertia}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'elbow.png'), bbox_inches='tight')
    plt.close()
    return inertia

def silhouette(data, max_k, result_dir, seed):
    silhouette_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    print(f"Silhouette Score: {silhouette_scores}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'silhouette.png'), bbox_inches='tight')
    plt.close()
    return silhouette_scores

def plot_similarity_matrix(matrix, filename, result_dir):
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, cmap='coolwarm', annot=False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), bbox_inches='tight')
    plt.close()

def save_cluster_labels(data, reduced_data: np.ndarray, labels: np.ndarray, result_dir: str):
    reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(reduced_data.shape[1])])
    reduced_df['ID'] = data['ID'].values
    id_cluster_df = pd.DataFrame({'ID': reduced_df['ID'], 'Cluster': labels})
    id_cluster_path = os.path.join(result_dir, 'original_data_with_labels.csv')
    id_cluster_df.to_csv(id_cluster_path, index=False)

def save_loading_scores(loading_scores: pd.DataFrame, filename: str, result_dir: str):
    loading_scores_file = os.path.join(result_dir, filename)
    loading_scores.to_csv(loading_scores_file, index=True)

def create_cumulative_variance_plot(cumulative_variance, result_dir, scaler_name):
    plt.figure(figsize=(10, 10))
    plt.plot(cumulative_variance, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title(f'Cumulative explained variance for {scaler_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{scaler_name}_cumulative_variance.png'), bbox_inches='tight')
    plt.close()

def create_individual_variance_plot(explained_variance, result_dir, scaler_name):
    plt.figure(figsize=(10, 10))
    plt.plot(explained_variance, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title(f'Explained variance for {scaler_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{scaler_name}_explained_variance.png'), bbox_inches='tight')
    plt.close()
    
def plot_results(results_df, result_dir, filename='model_scores.png'):
    """
    Plot dei risultati (R² e Q²) per ciascun modello e salva l'immagine.
    """
    results_df = results_df.sort_values(by='R2', ascending=False)
    models = results_df['Model']
    r2_scores = results_df['R2']
    q2_scores = results_df['Q2']

    valid_indices = ~q2_scores.isnull()
    models = models[valid_indices]
    r2_scores = r2_scores[valid_indices]
    q2_scores = q2_scores[valid_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(models, r2_scores, color='blue', alpha=0.7, label='R²')
    plt.barh(models, q2_scores, color='orange', alpha=0.7, label='Q²')
    plt.xlabel('Score')
    plt.ylabel('Models')
    plt.title('R² e Q² per Modello')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), bbox_inches='tight')
    plt.close()