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
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split 
from dataset.processing import remove_highly_correlated_features, remove_zero_variance_features


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
            'colsample_bytree': [0.8],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }
    )
}

def allign(self,X_train, X_test):
    """ 
    Allign train and test set for retrain models
    """
    selected_features = np.load("/home/federica/LAB2/chembl1865/egfr_qsar/qsar_results/selected_features.npy")
    #selected_features = [feature for feature in selected_features if feature in to_keep]
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

def calculate_tanimoto_similarity(smiles_list1, smiles_list2):
    """
    Calcola la similarità di Tanimoto tra due liste di SMILES.
    """
    fps1 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) for s in smiles_list1 if Chem.MolFromSmiles(s)]
    fps2 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) for s in smiles_list2 if Chem.MolFromSmiles(s)]
    
    similarities = []
    for fp1 in fps1:
        for fp2 in fps2:
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return avg_similarity
def split_train_test(df_x, df_y):
    df_total = pd.concat([df_x, df_y]).reset_index(drop=True)

    df_total['Log_Bins'] = pd.qcut(df_total['Log Standard Value'], q=4, labels=False, duplicates='drop')
    print(df_total['Log_Bins'].value_counts())
    df_total['Log_Bins'] = pd.cut(df_total['Log Standard Value'], bins=5, labels=False)
    df_x, df_y = train_test_split(df_total, test_size=0.3, stratify=df_total['Log_Bins'], random_state=42)
    df_x = df_x.drop(columns=['Log_Bins'])
    df_y = df_y.drop(columns=['Log_Bins'])
    return df_x, df_y

def check_train_test_similarity(train_set, test_set):
    """
    Verifica la similarità media tra train e test set.
    """
    smiles_train = train_set['Smiles'].dropna().tolist()
    smiles_test = test_set['Smiles'].dropna().tolist()
    similarity = calculate_tanimoto_similarity(smiles_train, smiles_test)
    print(f"Similarità media Train-Test (Tanimoto): {similarity:.4f}")
    return similarity