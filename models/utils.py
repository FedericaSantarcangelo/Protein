"""Script with utility functions for data handling.
@Author: Federica Santarcangelo
"""
import os
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs 
from sklearn.cluster import DBSCAN
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features
from sklearn.model_selection import train_test_split

models = {
    'Random Forest': (
        RandomForestRegressor(random_state=42),
        {
            'n_estimators': [50, 100],
            'max_depth': [3, 5,],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'max_samples': [0.8, None],
            'criterion': ['squared_error']
        }
    ), 
    'AdaBoost': (
        AdaBoostRegressor(random_state=42),
        {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1]
        }
    ),
    'Gradient Boosting': (
        GradientBoostingRegressor(random_state=42),
        {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8],
            'min_samples_split': [5, 10],  
            'min_samples_leaf': [2, 4]
        }
    ),  
    'MLP': (MLPRegressor(random_state=42),{'hidden_layer_sizes': [(50,), (100,), (100, 100), (200, 100)],'alpha': [0.0001, 0.001, 0.01],'learning_rate_init': [0.001, 0.01]}),
    'SVR': (SVR(),{'C': [0.1, 1, 10, 100],'gamma': ['scale', 'auto', 0.1, 1],'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsRegressor(),{'n_neighbors': [3, 5, 7, 10, 15],'weights': ['uniform'],'metric': ['euclidean', 'manhattan', 'chebyshev']}),
    'XGBoost': (
        XGBRegressor(random_state=42),
        {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    )
}

def allign(self,X_train, X_test):
    """ 
    Allign train and test set for retrain models
    """
    selected_features = np.load("/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results/selected_features.npy")
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s['ID'] = np.arange(len(X_train_s))
    X_test_s['ID'] = np.arange(len(X_test_s))
    scaled_X_train = self.scaler.transform(X_train_s)
    scaled_X_test = self.scaler.transform(X_test_s)
    feature_names = np.array(X_train_s.columns)
    feature_mask = np.isin(feature_names, selected_features) 
    scaled_X_train = scaled_X_train[:, feature_mask]
    scaled_X_test = scaled_X_test[:, feature_mask]
    pca_scaled_X_train = self.pca.transform(scaled_X_train)
    pca_scaled_X_test = self.pca.transform(scaled_X_test)    
    return pca_scaled_X_train, pca_scaled_X_test

def select_best_model(dir='/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results'):
    """
    Select the best model based on the highest Q2 score
    """
    best_results = []
    for file in os.listdir(dir):
        if file.endswith('_results.csv'):
            results_df = pd.read_csv(os.path.join(dir, file))
            filtered_results_df = results_df[results_df['R2'] > results_df['Q2']]
            filtered_results_df = filtered_results_df[filtered_results_df['R2'] < 1]
            if not filtered_results_df.empty:
                filtered_results_df['R2+Q2'] = filtered_results_df['R2'] + filtered_results_df['Q2']
                filtered_results_df = filtered_results_df.sort_values('R2+Q2', ascending=False)
                filtered_results_df['Vote sum'] = range(1, len(filtered_results_df) + 1)
                filtered_results_df['R2-Q2'] = filtered_results_df['R2'] - filtered_results_df['Q2']
                filtered_results_df = filtered_results_df.sort_values('R2-Q2', ascending=True)
                filtered_results_df['Vote diff'] = range(1, len(filtered_results_df) + 1)
                filtered_results_df['Vote'] = filtered_results_df['Vote sum'] + filtered_results_df['Vote diff']
                filtered_results_df = filtered_results_df.sort_values('Vote', ascending=True)
                filtered_results_df['(R2+Q2)-(R2-Q2)'] = filtered_results_df['R2+Q2'] - filtered_results_df['R2-Q2']
                filtered_results_df = filtered_results_df.sort_values('(R2+Q2)-(R2-Q2)', ascending=False)
                index = filtered_results_df.index[0]
                filtered_results_df.to_csv(os.path.join(dir, file), index=False)
                best_results.append(filtered_results_df.loc[[index]])
    best_results_df = pd.concat(best_results, ignore_index=True)
    best_results_df.to_csv(os.path.join(dir, 'best_model.csv'), index=False)
    return best_results_df

def preprocess_and_pca(scaled_X_train_f, feature_X_train_f, scaled_Y_train_f, feature_Y_train_f):
    """
    Preprocess the data to select the most relevant features and apply PCA
    """
    scaled_X_train_f, feature_X_train_f = remove_zero_variance_features(scaled_X_train_f, feature_X_train_f)
    scaled_Y_train_f, feature_Y_train_f = remove_zero_variance_features(scaled_Y_train_f, feature_Y_train_f)
    common_features = np.intersect1d(feature_X_train_f, feature_Y_train_f)
    feature_mask_X_train_f = np.isin(feature_X_train_f, common_features)
    feature_mask_Y_train_f = np.isin(feature_Y_train_f, common_features)
    scaled_X_train_f = scaled_X_train_f[:, feature_mask_X_train_f]
    scaled_Y_train_f = scaled_Y_train_f[:, feature_mask_Y_train_f]
    feature_X_train_f = feature_X_train_f[feature_mask_X_train_f]
    feature_Y_train_f = feature_Y_train_f[feature_mask_Y_train_f]
    scaled_X_train_f, feature_X_train_f = remove_highly_correlated_features(scaled_X_train_f, feature_X_train_f)
    scaled_Y_train_f, feature_Y_train_f = remove_highly_correlated_features(scaled_Y_train_f, feature_Y_train_f)
    common_features = np.intersect1d(feature_X_train_f, feature_Y_train_f)
    feature_mask_X_train_f = np.isin(feature_X_train_f, common_features)
    feature_mask_Y_train_f = np.isin(feature_Y_train_f, common_features)
    scaled_X_train_f = scaled_X_train_f[:, feature_mask_X_train_f]
    scaled_Y_train_f = scaled_Y_train_f[:, feature_mask_Y_train_f]
    feature_X_train_f = np.array(feature_X_train_f)[feature_mask_X_train_f]
    feature_Y_train_f = np.array(feature_Y_train_f)[feature_mask_Y_train_f]
    np.save("/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results/selected_features.npy", feature_X_train_f)
    return scaled_X_train_f, feature_X_train_f, scaled_Y_train_f, feature_Y_train_f

def smiles_to_ecfp4(smiles_list, radius=2, n_bits=1024):
    """Convert SMILES to ECFP4 fingerprints."""
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros((n_bits,), dtype=np.int8))  # Placeholder for invalid SMILES
    return np.array(fingerprints)

def bin_random_split(data: pd.DataFrame, eps=0.5, min_samples=5, train_ratio=0.6):
    """Split data into train and test sets based on DBSCAN clustering and activity stratification."""

    fingerprints = smiles_to_ecfp4(data['Smiles'].tolist())
    scaler = StandardScaler()
    data['standard_value_scaled'] = scaler.fit_transform(data[['Standard Value']])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    data['cluster'] = dbscan.fit_predict(fingerprints)
    train_idx, test_idx = [], []
    for cluster in np.unique(data['cluster']):
        cluster_data = data[data['cluster'] == cluster]
        if len(cluster_data) > 1: 
            try:
                train_sub, test_sub = train_test_split(
                    cluster_data, train_size=train_ratio, random_state=42, stratify=cluster_data[['standard_value_scaled']]
                )
            except ValueError: 
                train_sub, test_sub = train_test_split(
                    cluster_data, train_size=train_ratio, random_state=42
                )
            train_idx.extend(train_sub.index.tolist())
            test_idx.extend(test_sub.index.tolist())
        else:
            train_idx.extend(cluster_data.index.tolist())

    train = data.loc[train_idx].drop(columns=['cluster', 'standard_value_scaled'])
    test = data.loc[test_idx].drop(columns=['cluster', 'standard_value_scaled'])
    
    return train, test