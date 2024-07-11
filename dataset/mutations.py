from argparse import Namespace, ArgumentParser
import os
import pandas as pd
import sys
import numpy as np
from utils.args import data_cleaning_args
import re
from collections import defaultdict


class Mutation():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data
        self.pattern = re.compile(r'[A-Z]\d{1,4}[A-Z](?:\w+)?') #devo formattare diversi pattern di mutazioni

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
        single = self.single_mutation(knonw_mutations, data)
        double = self.double_mutation(knonw_mutations, data)
        triple = self.triple_mutation(knonw_mutations, data)
        wild = self.wild_type(knonw_mutations, data)
        mixed = self.mixed_type(knonw_mutations, data)
        final = self.format_output(single, double, triple, wild, mixed)
        return final


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

        #create the dictionary
        mutation_dict = {}
        for _, row in uniprot.iterrows():
            key = (row['Accession Code'], row['ChEMBL DB'])
            mutations = [mutation for mutation in row['Known mutations'] if mutation]
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
        mutation_match = defaultdict(list)
        for(accession_code, chembl_id), mutations in uniprot.items():
            for mutation in mutations:
                mutation_key = ()
                #compute the shift for each mutation
                for shift in range(-2,3):
                    shifted_mutation = self.shift_mutation(mutation, shift)
                    mutation_key += (shifted_mutation,)
                #search the mutation in the data 
                for mut in mutation_key:
                    matches = data[data['Assay Description'].str.contains(mut, case=False, na=False)]['Molecule ChEMBL ID'].unique() 
                    mutation_match[mutation_key].extend(matches)

        if not mutation_matches[mutation_key]:
            del mutation_matches[mutation_key]   

        mutation_matches = {key: value for key, value in mutation_matches.items() if value}        
        return mutation_match
    
    def shift_mutation(self, mutation: str, shift: int):
        """Shift mutation
        :param mutation: mutation
        :param shift: shift
        :return: shifted mutation
        """
        if not self.pattern.match(mutation):
            raise ValueError(f"Mutation {mutation} is not in the correct format")
        
        #spilt the mutation in the letter and the number
        letter = mutation[0]
        number = mutation[1:-1]
        last_letter = mutation[-1]

        try:
            number = int(number)
        except ValueError:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
        
        #shift the number
        new_num = str(number + shift)

        if len(new_num) > 4 or int(new_num) <= 0:
            return None
        mutation = f"{letter}{new_num}{last_letter}"
        return mutation
    
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


#le mutazioni le riconosco nell'assay description perchè sono formattate come lettera(singolo char)+numero(da 1 a 4 cifre)+lettera(singolo char) 
#1.le key words che devo considerare sono wt, wild type e wild-type quindi devo cercare queste key words nel campo assay description e tenere questi record
#2. cerco il tipo di mutazione e vedo se ci sono altri record con la stessa mutazione 
#3. distinzione tra mutazione singole doppie e triple
#gli args possono essere gli stessi del data cleaning perchè mi interessano fondamentalmente i dizionari delle priorità
#devo capire se implementare anche una funzione per leggere i file uniprot nel caso siano presenti e quindi le mutazioni siano già state annotate e le prendo direttamente da li
#potrebbe essere utile creare il file di tutte le mutazioni che vengono trovate che viene sempre aggiornato in base ai nuovi dati che vengono analizzati,
#creando un db di mutazioni che possono essere usate per fare analisi statistiche 
