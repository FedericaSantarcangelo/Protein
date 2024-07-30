from argparse import ArgumentParser

def data_cleaning_args(parser : ArgumentParser) -> None:
    """ Add arguments for data cleaning 
        :param parser: the parser istance
    """
    parser.add_argument('--assay_type', type = str, default = 'None', help = 'Specify the assay type to filter the data')
    parser.add_argument('--assay_organism', type = str, default = 'None', help = 'Specify the assay organism to filter the data')
    parser.add_argument('--BAO_Label', type = str, default = 'None', help = 'Specify the BAO label to filter the data')
    parser.add_argument('--target_type', type = str, default = 'None', help = 'Specify the target type to filter the data')

    parser.add_argument('--standard_type_perc',type = str, default = 'None', nargs = '+', help = 'Specify the standard_type_perc to filter the data')
    parser.add_argument('--standard_type_log',type = str, default = 'None',nargs = '+', help = 'Specify the standard_type_log to filter the data')
    parser.add_argument('--standard_type_act', type = str, default = 'None', nargs = '+', help = 'Specify the standard_type_act to filter the data')

    parser.add_argument('--assay_description_perc', type=str, default='None',nargs='+', help = 'Specify the standard value to filter the data')

    parser.add_argument('--thr_perc', type = float, default = 50, help = 'Specify the threshold for the active/inactive compounds')
    parser.add_argument('--thr_act', type = float, default = 1000, help = 'Specify the threshold for the active/inactive compounds')

    parser.add_argument('--rel_pri', type=str, default="{'=':1, '=<':2, '>=':3, '>':4,'<':5}",
                        help="relation priority")
    parser.add_argument('--sty_pri', type=str, default="{'IC50':1, 'Ki':2, 'Kd':3, 'EC50':4, 'Inhibition':5, 'Activity':6}",
                        help="standard type priority")
    parser.add_argument('--src_pri', type=str, default="{'Scientific Literature':1, 'BindingDB Database':2, 'Fraunhofer HDAC6':3}",
                        help="source priority")
    parser.add_argument('--path_uniprot', type = str, default = '/home/federica/df_uniprot_details.tsv', help = 'Specify the path of uniprot file')
    
    parser.add_argument('--mutation', type = bool, default = False, help = 'Specify if the mutation is needed')
        
        
    def model_args(parser: ArgumentParser) -> None:
        """ Add arguments for model training
        :param parser: the parser istance
        """
        parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'logistic', 'svm', 'rf'],
                        help='Specify the type of model to use for training')
    
        parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Specify the learning rate for model training')
    
        parser.add_argument('--n_estimators', type=int, default=100,
                        help='Specify the number of estimators for ensemble models like Random Forest')
    
        parser.add_argument('--max_depth', type=int, default=None,
                        help='Specify the maximum depth of the tree (used for decision tree-based models)')
    
        parser.add_argument('--batch_size', type=int, default=32,
                        help='Specify the batch size for training')
    
        parser.add_argument('--epochs', type=int, default=10,
                        help='Specify the number of epochs for training')
    
        parser.add_argument('--regularization', type=float, default=0.0,
                        help='Specify the regularization parameter to prevent overfitting')
    
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                        help='Specify the optimizer to use for training')
    
        parser.add_argument('--loss_function', type=str, default='mse', choices=['mse', 'cross_entropy'],
                        help='Specify the loss function to use for training')
    
        parser.add_argument('--save_model_path', type=str, default='./model.pkl',
                        help='Specify the path to save the trained model')
