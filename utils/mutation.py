import os 
import pandas as pd

def marge_data(organism: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two dataframes on a specific column
    :param organism: the first dataframe
    :param mapping: the second dataframe
    :param on: the column to merge the dataframes
    :return: the merged dataframe
    """
    merged_df = pd.merge(mapping, organism, left_on='UniProtID', right_on='Entry')
    return merged_df[['UniProtID', 'Target_ChEMBLID', 'Mutations']]

def save_mutation_target(args, data: pd.DataFrame, id_column: str='Target ChEMBL ID') -> None:
    """
    Save the mutation target
    :param data: the data to be saved
    :param path: the path where to save the data
    """
    from dataset.preparation import Cleaner
    try:
        if id_column not in data.columns:
            raise ValueError(f"{id_column} not in the columns of the dataframe")
        
        full_path = os.path.join(args.path_output, 'mutation_target')
        
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
        
        cleaner = Cleaner(args)
        drop_dupicates = cleaner.remove_duplicate(data)

        grouped = drop_dupicates.groupby(id_column)

        for name, group in grouped:
            output_path = os.path.join(full_path, f"{name}.csv")
            group.to_csv(output_path, index=False)
    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error I/O: {e}")