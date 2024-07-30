
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np
import os

# Initialize the RDKit and Mordred descriptor calculators globally
exclude_descriptors = ['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
descriptor_names = [desc[0] for desc in Descriptors._descList if desc[0] not in exclude_descriptors]
#rdkit_calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in filtered_descriptors])
#descriptor_names = [x[0] for x in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
path = '/home/luca/LAB/LAB_federica/data'
mordred_calculator = Calculator(descriptors, ignore_3D=False)

# Initialize Descriptors3D
descriptor_functions = {name: func for name, func in Descriptors3D.__dict__.items() if callable(func) and name[0] != '_'}

def prepare_molecule(smiles): #singolo smiles e non multiplo 
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)

        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
        return mol
        

def calculate_rdkit_descriptors(mol):
    
    try:
        confs = AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=0)
        if len(confs)==0:
            return  [None]*len(descriptor_names)
    except RuntimeError:
        return [None]*len(descriptor_names)
        
    AllChem.ComputeGasteigerCharges(mol)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props:
        optResults = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000)
    else:
        optResults = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)

    if not any(optResults[0] == 0 for optResults in optResults):
        return [None]*len(descriptor_names)
        
    for optResult,confid in optResults:
        if optResult == 0: 
            descriptors = calculator.CalcDescriptors(mol, confId=confid)
            return descriptors
    return [None]*len(descriptor_names)
    

def calculate_rdkit_3d_descriptors(mol):
    if mol and mol.GetNumConformers() > 0:
        descriptors = {}
        for name, func in descriptor_functions.items():
            try:
                descriptors[name] = func(mol)
            except Exception as e:
                descriptors[name] = np.nan
        return [descriptors[name] for name in sorted(descriptor_functions)]
    return [np.nan] * len(descriptor_functions)

def calculate_mordred_descriptors(mol):
    if mol:
        mordred_desc = mordred_calculator(mol)
        return list(mordred_desc.fill_missing(np.nan).values())
    return [np.nan] * len(mordred_calculator.descriptors)

def process_molecule_with_logging(mol_id, smiles):
 
    try:
        mol2 = prepare_molecule(smiles)
        if mol2 is None:
            raise ValueError("Molecule preparation failed.")

        rdkit_desc = calculate_rdkit_descriptors(mol2)
        rdkit_3d_desc = calculate_rdkit_3d_descriptors(mol2)
        mordred_desc = calculate_mordred_descriptors(mol2)

        combined_descriptors = list(rdkit_desc) + list(rdkit_3d_desc) + list(mordred_desc)
        return mol_id, combined_descriptors
    except Exception as e:
        print(f"Error processing molecule {mol_id}: {e}")
        total_descriptor_count = len(descriptor_names) + len(descriptor_functions) + len(mordred_calculator.descriptors)
        return mol_id, [np.nan] * total_descriptor_count

def process_molecules_and_calculate_descriptors(df):
    smiles_dict = df.set_index('Molecule ChEMBL ID')['Smiles'].to_dict()

    results = []
    for mol_id, smiles in smiles_dict.items():
        results.append(process_molecule_with_logging(mol_id, smiles))

    
    # Add prefixes to descriptor names
    rdkit_descriptor_cols = ['rdkit_' + name for name in descriptor_names]
    rdkit_3d_descriptor_cols = ['rdkit_3d_' + name for name in sorted(descriptor_functions)]
    mordred_descriptor_cols = ['mordred_' + str(d) for d in mordred_calculator.descriptors]

    # Combine all descriptor column names
    descriptor_cols = rdkit_descriptor_cols + rdkit_3d_descriptor_cols + mordred_descriptor_cols

    descriptors_df = pd.DataFrame.from_records(results, columns=['Molecule ChEMBL ID', 'Descriptors'])
    descriptors_df = pd.concat([descriptors_df.drop(['Descriptors'], axis=1), descriptors_df['Descriptors'].apply(pd.Series)], axis=1)
    descriptors_df.columns = ['Molecule ChEMBL ID'] + descriptor_cols

    merged_df = pd.merge(df, descriptors_df, on='Molecule ChEMBL ID', how='left')

    # Remove the 'CalcMolDescriptors3D' column if it exists
    merged_df.drop(columns=['rdkit_3d_CalcMolDescriptors3D'], axis=1, inplace=True, errors='ignore')
    merged_df.to_csv(os.path.join(path, 'final_df.csv'),index=False, encoding='utf-8')
    return merged_df