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

    #parser.add_argument('--rel_priority', type=str, default='None', nargs = ' + ' ,help = 'Specify the relation priority ')
    #parser.add_argument('--sty_priority',type=str, default='None', nargs = ' + ', help = 'Specify the standard type priority')
    #parser.add_argument('--src_priority', type=str, default='None', nargs = ' + ', help = 'Specify the source priority')