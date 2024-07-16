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
        self.shift=[-2,-1,1,2]

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
        knonw_mutations,all = self.format_uniprot(uniprot.copy())
        no_mut,mut=self.split_data(data.copy())
        single_k, single_u = self.single_mutation(all, mut) #forse aggiungere anche il dict delle mutazioni conosciute per la return formattata in un altro modo
        double_k,double_u = self.double_mutation(all, mut)
        triple_k,triple_u = self.triple_mutation(all, mut)
        wild_k, wild_u = self.wild_type(all, mut)
        final = self.format_output(single_k, single_u, double_k,double_u, triple_k,triple_u, wild_k, wild_u)
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

        all_mutations = set()
        for mutations in mutation_dict.values():
            for mutation in mutations:
                all_mutations.add(mutation)  # Aggiunge la mutazione se non è una tupla

        return mutation_dict,all_mutations
    
        
    def shift_mutation(self, mutation: str, shift: list):
        """Shift mutation
        :param mutation: mutation
        :param shift: shift
        :return: shifted mutation
        """
        match = re.match(r"\b[A-Z]\d{1,4}[A-Z]\b", mutation)
        if not match:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
    
        letter, number, last_letter = mutation[0], mutation[1:-1], mutation[-1]
        shifted_num=[]
        shifted_mutation=[]
        # Applica lo shift al numero
        for s in shift:
            shifted_num = str(int(number) + s)
            m = f"{letter}{shifted_num}{last_letter}"
            shifted_mutation.append(m)    
        return shifted_mutation
    
    def find_or_shift(self, mutation, uniprot_set):
        """Cerca la mutazione in uniprot o prova gli shift."""
        if mutation in uniprot_set:
            return [mutation]
        else:
            return self.shift_mutation(mutation, self.shift)

    def single_mutation(self, uniprot, data: pd.DataFrame):
        """Get mutations
        :mutation_dict: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        :param data: data dataframe with mutations to be found
        :return: mutations
        """
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
                if mutation in uniprot:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation))
                else:
                    shifted_mutation = self.shift_mutation(mutation, self.shift)
                    found = (any(s in uniprot for s in shifted_mutation))
                    if found:
                        #salvo ID, mutazione del chembl che non c'è in uniprot e shift presente nel uniprot
                        found_mut.append((row['Molecule ChEMBL ID'], mutation, found))  
                    else:
                    #salvo ID, mutazione del chembl che non c'è in uniprot e tutti gli shift che comunque non sono presenti
                        not_found.append((row['Molecule ChEMBL ID'], mutation, shifted_mutation))
                        
        return found_mut,not_found

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

        for _, row in data.iterrows():
            mutation_match = pattern.search(row['Assay Description'])
            if mutation_match:
                mutation = mutation_match.group()
                split_mutations = re.split(r'\/|-', mutation)
                if len(split_mutations) != 2:
                    continue
                split1, split2 = split_mutations[0], split_mutations[1]
                mutations_shifted = [self.find_or_shift(split1, uniprot), self.find_or_shift(split2, uniprot)]
                if len(mutations_shifted[0]) == 1 and len(mutations_shifted[1]) == 1:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation))
                else:
                    not_found.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted))

        return found_mut,not_found
    
    def triple_mutation(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get triple mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: triple mutations
        """
        pattern = re.compile(r'\b([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)){2}\b')
        found_mut = []
        not_found = []
        #modificare logica for, troppo lungo. modificare chiamata a shift_mutation
        for _,row in data.iterrows():
            mutation_match = pattern.search(row['Assay Description'])
            if mutation_match:
                mutation = mutation_match.group()
                split=re.split(r'\/|-',mutation)
                if len(split) != 3:
                    continue
                split1,split2,split3=split[0],split[1],split[2]
                mutations_shifted=[self.find_or_shift(split1,uniprot),self.find_or_shift(split2,uniprot),self.find_or_shift(split3,uniprot)]
                if len(mutations_shifted[0])==1 and len(mutations_shifted[1])==1 and len(mutations_shifted[2])==1:
                    found_mut.append((row['Molecule ChEMBL ID'],mutation))
                else:
                    not_found.append((row['Molecule ChEMBL ID'],mutation,mutations_shifted))
        return found_mut,not_found
    
    def wild_type(self, uniprot: pd.DataFrame, data: pd.DataFrame):
        """Get wild type mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: wild type mutations
        """
        pattern = re.compile(r'wild type| wt|wilde_type', re.IGNORECASE)
        pattern_combined = re.compile(
            r'\b[A-Z]\d{1,4}[A-Z]\b|'  # Cattura single mutations
            r'\b(?:[A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/[A-Z]\d+[A-Z]|-[A-Z]\d+[A-Z])?\b|'  # Cattura double mutations
            r'\b([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)){2}\b',  # Cattura triple mutations
            re.IGNORECASE)
        found = []
        not_found = []

        for _, row in data.iterrows():
            wt = pattern.search(row['Assay Description'])
            if wt:
                mut_match=pattern_combined.search(row['Assay Description'])
                if mut_match:
                    mutation = mut_match.group()
                    split = re.split(r'\/|-', mutation)
                    if len(split) == 1:
                        if split[0] in uniprot:
                            found.append((row['Molecule ChEMBL ID'], mutation))
                        else:
                            shifted_mutation = self.shift_mutation(split[0], self.shift)
                            found_shifted = any(s in uniprot for s in shifted_mutation)
                            if found_shifted:
                                found.append((row['Molecule ChEMBL ID'], mutation, found_shifted))
                            else:
                                not_found.append((row['Molecule ChEMBL ID'], mutation, shifted_mutation))
                    elif len(split) == 2:
                        split1, split2 = split[0], split[1]
                        mutations_shifted = [self.find_or_shift(split1, uniprot), self.find_or_shift(split2, uniprot)]
                        if len(mutations_shifted[0]) == 1 and len(mutations_shifted[1]) == 1:
                            found.append((row['Molecule ChEMBL ID'], mutation))
                        else:
                            not_found.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted))
                    elif len(split) == 3:
                        split1, split2, split3 = split[0], split[1], split[2]
                        mutations_shifted = [self.find_or_shift(split1, uniprot), self.find_or_shift(split2, uniprot),
                                             self.find_or_shift(split3, uniprot)]
                        if len(mutations_shifted[0]) == 1 and len(mutations_shifted[1]) == 1 and len(mutations_shifted[2]) == 1:
                            found.append((row['Molecule ChEMBL ID'], mutation))
                        else:
                            not_found.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted))
        return found, not_found
    
    
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
