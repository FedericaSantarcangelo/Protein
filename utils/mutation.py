import os 
import pandas as pd


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
            output_path = os.path.join(args.path_output, f"{name}.csv")
            group.to_csv(output_path, index=False)
    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error I/O: {e}")