
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem.SaltRemover import SaltRemover
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np

exclude_descriptors = ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
mordred_calculator = Calculator(descriptors, ignore_3D=False)

rdkit_descriptor_names = [name for name, func in Descriptors._descList if name not in exclude_descriptors]
descriptor_functions_3d = {name: func for name, func in Descriptors3D.__dict__.items() if callable(func) and not name.startswith('_')}


def prepare_molecule(smiles): 
    """
    Prepara una molecola per il calcolo dei descrittori
    :param smiles: la stringa SMILES della molecola
    :return: un oggetto molecolare preparato
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
    except:
        return None
    return mol
        

def calculate_rdkit_descriptors(mol):
    """
    Calcola i descrittori 2D di RDKit
    :param mol: la molecola
    :return: una lista di descrittori
    """
    try:
        return [getattr(Descriptors, name)(mol) for name in rdkit_descriptor_names]
    except Exception as e:
        return [np.nan] * len(rdkit_descriptor_names)
        
    
def calculate_rdkit_3d_descriptors(mol):
    """
    Calcola i descrittori 3D di RDKit
    :param mol: la molecola
    :return: una lista di descrittori
    """
    if mol and mol.GetNumConformers() > 0:
        return [func(mol) if mol.GetNumConformers() > 0 else np.nan for func in descriptor_functions_3d.values()]
    return [np.nan] * len(descriptor_functions_3d)

def calculate_mordred_descriptors(mol):
    """
    Compute Mordred descriptors for a molecule
    :param mol: the molecule
    return: the list of Mordred descriptors
    """
    if mol:
        mordred_desc = mordred_calculator(mol)
        return list(mordred_desc.fill_missing(np.nan).values())
    return [np.nan] * len(mordred_calculator.descriptors)

def process_molecule(mol_id, smiles):
    """
    Processa una molecola e calcola tutti i descrittori
    :param mol_id: identificativo della molecola
    :param smiles: rappresentazione SMILES
    :return: identificativo della molecola e lista dei descrittori
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
        print(f"Errore con la molecola {mol_id}: {e}")
        total_descriptor_count = len(rdkit_descriptor_names) + len(descriptor_functions_3d) + len(mordred_calculator.descriptors)
        return mol_id, [np.nan] * total_descriptor_count

def process_molecules_and_calculate_descriptors(df):
    """
    Processa ogni molecola in un DataFrame e calcola i descrittori
    :param df: il DataFrame con le molecole
    :return: il DataFrame con i descrittori calcolati
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

    merged_df.to_csv('/home/federica/LAB2/egfr_qsar/final_df.csv', index=False, encoding='utf-8')
    return merged_df