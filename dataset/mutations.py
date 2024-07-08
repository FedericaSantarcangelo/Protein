from argparse import Namespace, ArgumentParser
import os
import pandas as pd
import sys
import numpy as np
from utils.args import mutation


def parser_args():
    """ Get arguments parser """
    parser = ArgumentParser()
    mutation(parser)
    return parser

class Mutation():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data


