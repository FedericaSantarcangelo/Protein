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
        self.pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b|mutant|wild type| wt|wilde_type', re.IGNORECASE) #devo formattare diversi pattern di mutazioni

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
        single_k, single_u = self.single_mutation(knonw_mutations, mut)
        double = self.double_mutation(knonw_mutations, mut)
        triple = self.triple_mutation(knonw_mutations, mut)
        wild = self.wild_type(knonw_mutations, mut)
        mixed = self.mixed_type(knonw_mutations, mut)
        final = self.format_output(single_k, single_u, double, triple, wild, mixed)
        return final
    
    def split_data(self, data: pd.DataFrame):
        """Split data considering mutations: if there are no mutations, the row is added to no_mut, otherwise to mut
        :param data: data dataframe
        :return: no_mut, mut
        """
         # Aggiungi la colonna 'mutation'
        data['mutation'] = data['Assay Description'].apply(
            lambda x: bool(self.pattern.search(x)))
    
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

        #devo aggiungere l emutazioni con .. come le gestisco???

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
        all_mutations = set()
        for mutations in uniprot.values():
            for mutation in mutations:
                if isinstance(mutation, tuple):
                    all_mutations.update(mutation)  # Aggiunge ogni elemento della tupla
                else:
                    all_mutations.add(mutation)  # Aggiunge la mutazione se non è una tupla
        #cerco single mutant in uniprot(dizionario conosciute) e in data (dataframe con mutazioni da classificare)
        pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b', re.IGNORECASE)
        found_mut = []
        not_found = []
        #itero sul df
        for _,row in data.iterrows():
            #itero sulle mutazioni
            mutation = pattern.search(row['Assay Description'])
            if mutation:
                mutation = mutation.group()
                if mutation in all_mutations:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation))
                else:
                        for shift in [-2,-1,1,2]:
                            shifted_mutation = self.shift_mutation(mutation, shift)
                            if shifted_mutation in all_mutations:
                                found_mut.append((row['Molecule ChEMBL ID'], shifted_mutation))
                            else:
                                not_found.append((row['Molecule ChEMBL ID'], mutation))
                                break
        return found_mut,not_found
    
    def shift_mutation(self, mutation: str, shift: int):
        """Shift mutation
        :param mutation: mutation
        :param shift: shift
        :return: shifted mutation
        """
        match = re.match(r"\b[A-Z]\d{1,4}[A-Z]\b", mutation)
        if not match:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
    
        letter, number, last_letter = mutation[0], mutation[1:-1], mutation[-1]
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
        #cerco double mutant e combinazioni dei singoli mutanti
        pattern = re.compile(r'\b(?:[A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/[A-Z]\d+[A-Z]|-[A-Z]\d+[A-Z])?\b')
        found_mut = []
        not_found = []
        
        return data
    
    def triple_mutation(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get triple mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: triple mutations
        """
        #cerco triple mutant e combinazioni dei singoli e double mutanti

        return data
    
    def wild_type(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get wild type mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: wild type mutations
        """
        #cerco se ci sono in quelle conosciute
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
