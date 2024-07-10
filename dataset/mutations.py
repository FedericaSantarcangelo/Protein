from argparse import Namespace, ArgumentParser
import os
import pandas as pd
import sys
import numpy as np
from utils.args import data_cleaning_args

class Mutation():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data

    def load_data(self):
        """Load data uniprot
        :return: data
        """
        if not os.path.exists(self.args.path_uniprot):
            print(f"Path {self.args.path_uniprot} does not exist")
            sys.exit(1)
        else:
            try:
                uniprot = pd.read_csv(self.args.path_uniprot, sep='\t', dtype='str',low_memory=False)
                return uniprot
            except pd.errors.ParserError as e:
                print(f"ParserError: {e}")
        return None
    
    def get_mutations(self, data: pd.DataFrame):
        """Get mutations main function
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: mutations
        """
        uniprot = self.load_data()
        single = self.single_mutation(uniprot, data)
        double = self.double_mutation(uniprot, data)
        triple = self.triple_mutation(uniprot, data)
        wild = self.wild_type(uniprot, data)
        mixed = self.mixed_type(uniprot, data)
        return data


    def single_mutation(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: mutations
        """
        uniprot=uniprot.dropna(subset=['Known mutations'])
        for mutations in uniprot['Known mutations']:
        # Dividi le mutazioni su ';' per gestire più mutazioni
            split_mutations = mutations.split(';')
            for mutation in split_mutations:
                mutation = mutation.strip()  # Rimuovi spazi bianchi extra
                for assay_description in data['Assay Description']:
                    if mutation in assay_description:
                        print(data['Molecule ChEMBL ID'])
        return data
    
    def double_mutation(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get double mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: double mutations
        """
        return data
    
    def triple_mutation(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get triple mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: triple mutations
        """
        return data
    
    def wild_type(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get wild type mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: wild type mutations
        """
        return data
    
    def mixed_type(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get mixed type mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: mixed type mutations
        """
        return data
    
    


#le mutazioni le riconosco nell'assay description perchè sono formattate come lettera(singolo char)+numero(da 1 a 4 cifre)+lettera(singolo char) 
#1.le key words che devo considerare sono wt, wild type e wild-type quindi devo cercare queste key words nel campo assay description e tenere questi record
#2. cerco il tipo di mutazione e vedo se ci sono altri record con la stessa mutazione 
#3. distinzione tra mutazione singole doppie e triple
#gli args possono essere gli stessi del data cleaning perchè mi interessano fondamentalmente i dizionari delle priorità
#devo capire se implementare anche una funzione per leggere i file uniprot nel caso siano presenti e quindi le mutazioni siano già state annotate e le prendo direttamente da li
#potrebbe essere utile creare il file di tutte le mutazioni che vengono trovate che viene sempre aggiornato in base ai nuovi dati che vengono analizzati,
#creando un db di mutazioni che possono essere usate per fare analisi statistiche 
