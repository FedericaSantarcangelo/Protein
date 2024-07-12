from argparse import Namespace, ArgumentParser
import os
import pandas as pd
import sys
import re
from collections import defaultdict


class Mutation():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data
        self.pattern = re.compile(r'^[A-Z]\d{1,4}[A-Z]$') #devo formattare diversi pattern di mutazioni

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
        knonw_mutations = self.format_uniprot(uniprot.copy())
        no_mut,mut=self.split_data(data.copy())
        single = self.single_mutation(knonw_mutations, mut)
        double = self.double_mutation(knonw_mutations, mut)
        triple = self.triple_mutation(knonw_mutations, mut)
        wild = self.wild_type(knonw_mutations, mut)
        mixed = self.mixed_type(knonw_mutations, mut)
        final = self.format_output(single, double, triple, wild, mixed)
        return final
    
    def split_data(self, data: pd.DataFrame):
        """Split data considering mutations: if there are no mutations, the row is added to no_mut, otherwise to mut
        :param data: data dataframe
        :return: no_mut, mut
        """
         # Aggiungi la colonna 'mutation'
        data['mutation'] = data[data['Assay Description']].apply(
            lambda x: bool(self.pattern.search(x))  )
    
    # Filtra i dati
        mut = data[data['mutation'] == True]
        no_mut = data[data['mutation'] == False]
        return no_mut, mut

    def format_uniprot(self, uniprot: pd.DataFrame):
        """Format uniprot dataframe
        :param uniprot: uniprot dataframe
        :return: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        """
        #first remove all the rows with NaN values in the column 
        uniprot = uniprot.dropna(subset=['Known mutations'])

        #then I create a dictionary with the keys (Accession Code, CheMBL ID) and the values are the mutations
        #clean and standardize the mutations
        uniprot['Known mutations'] = uniprot['Known mutations'].str.replace(r';+', ';', regex=True) #remove multiple ;
        uniprot['Known mutations'] = uniprot['Known mutations'].str.replace(r'^\s*;\s*|\s*;\s*$', '', regex=True) #remove ; 
        uniprot['Known mutations'] = uniprot['Known mutations'].str.replace(r'\s*;\s*', ';', regex=True) #remove spaces before and after ;
        uniprot['Known mutations'] = uniprot['Known mutations'].str.replace(r'"', '', regex=True) # Rimuovi virgolette
        uniprot['Known mutations'] = uniprot['Known mutations'].str.split(';') #split the mutations
        #check if the mutations are in the correct format

        #create the dictionary
        mutation_dict = {}
        for _, row in uniprot.iterrows():
            key = (row['Accession Code'], row['ChEMBL DB'])
            mutations = [mutation for mutation in row['Known mutations'] if mutation if self.pattern.match(mutation)]
            if key not in mutation_dict:
                mutation_dict[key] = []
            mutation_dict[key].extend(mutations)

        return mutation_dict

    def single_mutation(self, uniprot, data: pd.DataFrame):
        """Get mutations
        :mutation_dict: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        :param data: data dataframe with mutations to be found
        :return: mutations
        """

        return
    
    def shift_mutation(self, mutation: str, shift: int):
        """Shift mutation
        :param mutation: mutation
        :param shift: shift
        :return: shifted mutation
        """
        match = re.match(r"([A-Z])(\d{1,4})([A-Z])", mutation)
        if not match:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
    
        letter, number, last_letter = match.groups()
        # Applica lo shift al numero
        shifted_number = int(number) + shift
        # Riassembla la mutazione
        shifted_mutation = f"{letter}{shifted_number}{last_letter}"
    
        return shifted_mutation
    
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
    
    def format_output(self, single, double, triple, wild, mixed):
        """Format output
        :param single: single mutations
        :param double: double mutations
        :param triple: triple mutations
        :param wild: wild type mutations
        :param mixed: mixed type mutations
        :return: mutations
        """
        return 
