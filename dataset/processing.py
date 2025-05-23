"""
@Author: Federica Santarcangelo
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem.SaltRemover import SaltRemover
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np
import os
exclude_descriptors = ['rdkit_Ipc','BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
mordred_calculator = Calculator(descriptors, ignore_3D=False)
rdkit_descriptor_names = [name for name, func in Descriptors._descList if name not in exclude_descriptors]
descriptor_functions_3d = {name: func for name, func in Descriptors3D.__dict__.items() if callable(func) and not name.startswith('_')}
def prepare_molecule(smiles): 
    """
    Prepare a molecule for descriptor calculation
    :return: the molecule
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    remover = SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)
    mol = Chem.AddHs(mol)
    try:
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
    except Chem.KekulizeException:
        return None
    return mol

def calculate_rdkit_descriptors(mol):
    """
    Compute RDKit descriptors for a molecule
    :return: the list of RDKit descriptors
    """
    try:
        return [getattr(Descriptors, name)(mol) for name in rdkit_descriptor_names]
    except Exception as e:
        return [np.nan] * len(rdkit_descriptor_names)

def calculate_rdkit_3d_descriptors(mol):
    """
    Compute 3D RDKit descriptors for a molecule
    :return: the list of 3D RDKit descriptors
    """
    if mol and mol.GetNumConformers() > 0:
        return [func(mol) for func in descriptor_functions_3d.values()]
    return [np.nan] * len(descriptor_functions_3d)

def calculate_mordred_descriptors(mol):
    """
    Compute Mordred descriptors for a molecule
    :return: the list of Mordred descriptors
    """
    if mol:
        mordred_desc = mordred_calculator(mol)
        return list(mordred_desc.fill_missing(np.nan).values())
    return [np.nan] * len(mordred_calculator.descriptors)

def process_molecule(mol_id, smiles):
    """
    Process a molecule and calculate its descriptors
    :return: the molecule ID and the list of descriptors
    """
    try:
        mol = prepare_molecule(smiles)
        if mol is None:
            raise ValueError("Molecola non valida")
        rdkit_desc = calculate_rdkit_descriptors(mol)
        rdkit_3d_desc = calculate_rdkit_3d_descriptors(mol)
        mordred_desc = calculate_mordred_descriptors(mol)
        combined_descriptors = rdkit_desc + rdkit_3d_desc + mordred_desc
        return mol_id, combined_descriptors
    except Exception as e:
        print(f"Error with molecule {mol_id}: {e}")
        total_descriptor_count = len(rdkit_descriptor_names) + len(descriptor_functions_3d) + len(mordred_calculator.descriptors)
        return mol_id, [np.nan] * total_descriptor_count

def process_molecules_and_calculate_descriptors(df):
    """
    Process molecules and calculate their descriptors
    :return: the dataframe with the descriptors
    """
    smiles_dict = df.set_index('Molecule ChEMBL ID')['Smiles'].to_dict()
    results = []
    for mol_id, smiles in smiles_dict.items():
        results.append(process_molecule(mol_id, smiles))
    rdkit_descriptor_cols = ['rdkit_' + name for name in rdkit_descriptor_names]
    rdkit_3d_descriptor_cols = ['rdkit_3d_' + name for name in descriptor_functions_3d.keys()]
    mordred_descriptor_cols = ['mordred_' + str(d) for d in mordred_calculator.descriptors]
    descriptor_cols = rdkit_descriptor_cols + rdkit_3d_descriptor_cols + mordred_descriptor_cols
    descriptors_df = pd.DataFrame.from_records(results, columns=['Molecule ChEMBL ID', 'Descriptors'])
    descriptors_df = pd.concat([descriptors_df.drop(['Descriptors'], axis=1), descriptors_df['Descriptors'].apply(pd.Series)], axis=1)
    descriptors_df.columns = ['Molecule ChEMBL ID'] + descriptor_cols
    merged_df = pd.merge(df, descriptors_df, on='Molecule ChEMBL ID', how='left')
    merged_df.to_csv('/home/luca/LAB/LAB_federica/egfr_qsar/final_df.csv', index=False, encoding='utf-8')
    return merged_df

def remove_zero_variance_features(data, feature_names):
    """
    Remove features with zero variance
    :return: the dataframe without zero variance features
    """
    if data.shape[1] != len(feature_names):
        data = data[:, :len(feature_names)]
    variance = np.var(data, axis=0)
    non_zero_variance = variance > 0
    return data[:, non_zero_variance], feature_names[non_zero_variance]

def remove_highly_correlated_features(data, feature_names, threshold=0.95):
    """
    Remove highly correlated features
    :return: the dataframe without highly correlated features
    """
    correlation_matrix = np.corrcoef(data, rowvar=False)
    to_drop = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            if abs(correlation_matrix[i, j]) > threshold:
                to_drop.add(j)
    return np.delete(data, list(to_drop), axis=1), [name for i, name in enumerate(feature_names) if i not in to_drop]

def delete_feature(path, loading_score,feature_names):
    """
    Delete features higly correlated with the absolute error
    :return: the dataframe without highly correlated features
    """
    best_model_info_df = pd.read_csv(os.path.join(path, 'best_model.csv'))
    to_keep = []
    for _, best_model_info in best_model_info_df.iterrows():
        model_name = best_model_info['Model']
        component = best_model_info['PC']
        pred_path = os.path.join(path, f'predictions/{model_name}_{component}_predictions_train.csv')
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions not found at {pred_path}")
        pred_df = pd.read_csv(pred_path)
        if 'y_pred' not in pred_df.columns and 'y_test' not in pred_df.columns:
            raise ValueError("Predictions file must contain 'y_pred' and 'y_test' column")
        y_test = pred_df['y_test'].values
        y_pred = pred_df['y_pred'].values
        abs_error = np.abs(y_test - y_pred)
        pca_scores = loading_score.iloc[:, 1:].values  
        correlation = {}
        if len(abs_error) <= pca_scores.shape[0]:  # Ensure pca_scores has enough rows
            pca_scores = pca_scores[:len(abs_error), :]  # Select the first len(abs_error) rows
            for i, feature in range(1, len(abs_error)):
                corr, _ = pearsonr(abs_error, pca_scores[:, i])
                correlation[feature] = corr
            to_keep = [feature for feature, corr in correlation.items() if abs(corr) < 0.5]
        else:
            raise ValueError("Length of abs_error is greater than the number of rows in pca_scores")
        return to_keep