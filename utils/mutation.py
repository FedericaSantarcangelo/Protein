import os 
import pandas as pd

def marge_data(organism: pd.DataFrame, mapping: pd.DataFrame, uniprot: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two dataframes on a specific column
    :param organism: the first dataframe
    :param mapping: the second dataframe
    :param on: the column to merge the dataframes
    :return: the merged dataframe
    """
    organism.rename(columns={'Entry': 'Accession Code'}, inplace=True)
    mapping.rename(columns={'UniProtID': 'Accession Code','Target_ChEMBLID':'ChEMBL DB'}, inplace=True)
    merged_df = organism.merge(mapping, on='Accession Code', how='left').merge(uniprot, on=['Accession Code','ChEMBL DB'], how='left')
    return merged_df[['Accession Code', 'ChEMBL DB', 'Known mutations']]

def save_mutation_target(args, data: pd.DataFrame, flag, f_path: str = 'mutation_target',id_column: str='Target ChEMBL ID') -> None:
    """
    Save the mutation target
    :param data: the data to be saved
    :param path: the path where to save the data
    """
    from dataset.preparation import Cleaner
    try:
        if id_column not in data.columns:
            raise ValueError(f"{id_column} not in the columns of the dataframe")
        
        full_path = os.path.join(args.path_output, f_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)

        cleaner = Cleaner(args)

        if flag != '1':
            drop_dupicates = data.drop_duplicates()
        else:
            drop_dupicates = cleaner.remove_duplicate(data)

        grouped = drop_dupicates.groupby(id_column)

        for name, group in grouped:
            output_path = os.path.join(full_path, f"{name}_{flag}.csv")
            if os.path.exists(output_path):
                try:
                    existing_data = pd.read_csv(output_path)
                    group = pd.concat([existing_data, group], ignore_index=True).drop_duplicates()
                except Exception as e:
                    print(f"Error during the reading of the file {output_path}: {e}")
            try:
                group.to_csv(output_path, index=False) #save or upodate the file
            except Exception as e:
                print(f"Error during the saving of the file {output_path}: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error I/O: {e}")
    
    return drop_dupicates

def population(data:pd.DataFrame):
    """
    Count the number of the document ChEMBL ID and assign the population to the dataframe
    :param data: the dataframe to be populated
    :return: the populated dataframe
    """
    data.sort_values(by='Document ChEMBL ID', inplace=True)
    counts=data['Document ChEMBL ID'].value_counts()
    data.loc[:, 'Population'] = data['Document ChEMBL ID'].map(lambda x: 'Plus' if counts[x] >= 3 else 'Less')
    return data

def find_mixed(mut: pd.DataFrame, no_mut:pd.DataFrame):
    """
    Find wrong mutation in the dataframe mut and move them in the dataframe no_mut with mixed label
    :param mut: the dataframe with the mutation
    :param no_mut: the dataframe without the mutation
    :return: no_mut with update
    """

    wrong_mut = mut.loc[mut['shifted_mutation'].str.contains('wrong')]
    mut = mut.drop(wrong_mut.index)
    wrong_mut.loc[:, 'mutation'] = False
    wrong_mut.loc[:, 'mutant_known'] = ''
    wrong_mut.loc[:, 'mutant'] = 'mixed'
    wrong_mut.loc[:, 'shifted_mutation'] = ''
    wrong_mut.loc[:, 'Accession Code'] = ''
    return wrong_mut
