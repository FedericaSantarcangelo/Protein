"""
@Author: Federica Santarcangelo
"""

from argparse import ArgumentParser

def data_cleaning_args(parser : ArgumentParser) -> None:
    """
    Add arguments for data cleaning 
    :param parser: the parser istance
    """
    # Add arguments for data cleaning based on the fields of interest
    parser.add_argument('--assay_type', type = str, default = 'None', help = 'Specify the assay type to filter the data')
    parser.add_argument('--assay_organism', type = str, default = 'None', help = 'Specify the assay organism to filter the data')
    parser.add_argument('--BAO_Label', type = str, default = 'None', help = 'Specify the BAO label to filter the data')
    parser.add_argument('--target_type', type = str, default = 'None', help = 'Specify the target type to filter the data')

    # Add arguments for data cleaning based on the standard type 
    parser.add_argument('--standard_type_perc',type = str, default = 'None', nargs = '+', help = 'Specify the standard_type_perc to filter the data')
    parser.add_argument('--standard_type_log',type = str, default = 'None',nargs = '+', help = 'Specify the standard_type_log to filter the data')
    parser.add_argument('--standard_type_act', type = str, default = 'None', nargs = '+', help = 'Specify the standard_type_act to filter the data')

    parser.add_argument('--assay_description_perc', type=str, default='None',nargs='+', help = 'Specify the standard value to filter the data')

    # Add arguments to distinguish between active and inactive compounds
    parser.add_argument('--thr_perc', type = float, default = 50, help = 'Specify the threshold for the active/inactive compounds')
    parser.add_argument('--thr_act', type = float, default = 1000, help = 'Specify the threshold for the active/inactive compounds')

    # Add arguments for the priority of the relations, standard types and sources
    parser.add_argument('--rel_pri', type=str, default="{'=':1, '=<':2, '>=':3, '>':4,'<':5}",
                        help="relation priority")
    parser.add_argument('--sty_pri', type=str, default="{'IC50':1, 'Ki':2, 'Kd':3, 'EC50':4, 'Potency':5,'Inhibition':6, 'INH':7, 'Inhibition (at 100uM)':8, 'Enzyme Inhibition':9, 'Activity':10,'Enzyme Activity':11}",
                        help="standard type priority")
    parser.add_argument('--src_pri', type=str, default="{'Scientific Literature':1, 'BindingDB Database':2, 'Fraunhofer HDAC6':3, 'PubChem':4}",
                        help="source priority")
    
def file_args(parser: ArgumentParser) -> None:
    """ 
    Add arguments for file paths
    :param parser: the parser istance
    """
    parser.add_argument('--path_uniprot', type = str, default = '/home/federica/LAB/df_uniprot_details.tsv', 
                        help = 'Specify the path of uniprot detailes file')
    parser.add_argument('--path_mapping', type = str, default= '/home/federica/LAB2/chembl_uniprot_mapping.txt', 
                        help = 'Specify the path of the mapping file between chembl and uniprot')
    parser.add_argument('--path_organism', type = str, default= '/home/federica/LAB2/uniprotkb_AND_model_organism_9606_2024_07_19_mutations.tsv',
                        help = 'Specify the path of the uniptot and model organism file')
    parser.add_argument('--path_output', type = str, default = '/home/luca/LAB/LAB_federica/', 
                        help = 'Specify the path where to save the output file')
    parser.add_argument('--path_assay',type=str, default='/home/federica/LAB2/assays.csv', help='Specify the name of the assay file with confidence informations')
    parser.add_argument('--path_proteinfamily', type=str, default='/home/federica/LAB2/family_protein_with_formatted.csv', 
                        help='Specify the path of the protein family classification file')
    # Add arguments for mutational analysis
    parser.add_argument('--mutation', type = bool, default = False, help = 'Specify if the mutation is needed')
        
        
def reducer_args(parser: ArgumentParser) -> None:
    """
    Add arguments for PCA and t-SNE analysis
    :param parser: the parser istance
    """
    parser.add_argument('--path_pca_tsne', type=str, default='/home/luca/LAB/LAB_federica/chembl1865/egfr_qsar/pca_tsne/', 
                        help='Specify the path of the directory where to save the PCA and t-SNE results')
    parser.add_argument('--n_clusters', type=int, default=5, help='Specify the number of clusters for the KMeans algorithm')
    parser.add_argument('--n_components_tsne', type=int, default=2, help='Specify the number of components for the t-SNE algorithms')
    parser.add_argument('--perplexity', type=int, default=30, help='Specify the perplexity for the t-SNE algorithm')
    parser.add_argument('--lr_tsne', type=int, default=200, help='Specify the learning rate for the t-SNE algorithm')
    parser.add_argument('--n_iter', type=int, default=1000, help='Specify the number of iterations for the t-SNE algorithm')
    parser.add_argument('--similarities', type=str, default='cosine', help='Specify the type of similarity to use for the clustering')
    parser.add_argument('--scaler', type=str, choices=['StandardScaler', 'MinMaxScaler', 'RobustScaler',
    'MaxAbsScaler', 'Normalizer', 'QuantileTransformer', 'PowerTransformer'], nargs='+', help='Specify the type of scaler to use for the data')

def qsar_args(parser: ArgumentParser) -> None:
    """
    Add arguments for QSAR analysis
    :param parser: the parser istance
    """
    parser.add_argument('--path_qsar', type=str, default='/home/luca/LAB/LAB_federica/chembl1865/egfr_qsar/qsar_results/', 
                        help='Specify the path of the directory where to save the QSAR results')
    parser.add_argument('--model', type=str, choices=['rf_regressor','svr_regressor','all'], required=True, help='Type of model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees for the Random Forest model')
    parser.add_argument('--max_depth', type=int, default=10 ,help='Maximum depth of the trees for the Random Forest model')
    
    parser.add_argument('--save_predictions', action='store_true', help='Save model predictions to output files')