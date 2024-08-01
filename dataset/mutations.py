from argparse import Namespace
import os
import sys
import re
import pandas as pd
from utils.file_utils import load_uniprot_data
from utils.mutation import save_mutation_target

class Mutation():
    def __init__(self, args: Namespace):
        self.args = args
        self.pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b|mutant|wild type|wild_type') 
        self.shift=[-2,-1,1,2]

        
    def get_mutations(self, data: pd.DataFrame):
        """
        Get mutations main function
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: final dataframe with mutations and no mutations, mutation_report dataframe with mutations found
        """
        uniprot = load_uniprot_data(self.args.path_uniprot)
        knonw_mutations,all_mut = self.format_uniprot(uniprot.copy())
        no_mut,mut=self.split_data(data.copy())
        mutant = self.find_mutant(mut,all_mut)
        final, mutation_report = self.format_output(no_mut,mutant,knonw_mutations)
        save_mutation_target(self.args, mutation_report)
        return final
    
    def split_data(self, data: pd.DataFrame):
        """
        Split data considering mutations: if there are no mutations, 
        the row is added to no_mut, otherwise to mut
        :param data: data dataframe
        :return: no_mut, mut
        """
        required_columns = ['mutation','mutant_known', 'mutant', 'shifted_mutation','Accession Code']
        for column in required_columns:
            if column not in data.columns:
                data[column] = ''

        data['mutation'] = data['Assay Description'].apply(
            lambda x: bool(self.pattern.search(x)))

        mut = data[data['mutation'] == True].copy()
        no_mut = data[data['mutation'] == False].copy()
        return no_mut, mut

    def format_uniprot(self, uniprot: pd.DataFrame):
        """
        Format uniprot dataframe
        :param uniprot: uniprot dataframe
        :return: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        """ 
        uniprot = uniprot.dropna(subset=['Known mutations'])

        uniprot.loc[:, 'Known mutations'] = uniprot['Known mutations'].str.replace(r';+', ';', regex=True)  # remove multiple ;
        uniprot.loc[:,'Known mutations'] = uniprot['Known mutations'].str.replace(r'^\s*;\s*|\s*;\s*$', '', regex=True) #remove ; 
        uniprot.loc[:,'Known mutations'] = uniprot['Known mutations'].str.replace(r'\s*;\s*', ';', regex=True) #remove spaces before and after ;
        uniprot.loc[:,'Known mutations'] = uniprot['Known mutations'].str.replace(r'"', '', regex=True) # Rimuovi virgolette
        uniprot.loc[:,'Known mutations'] = uniprot['Known mutations'].str.split(';') #split the mutations
        
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
                all_mutations.add(mutation)

        return mutation_dict,all_mutations
    
        
    def shift_mutation(self, mutation: str, shift: list):
        """
        Shift mutation
        :param mutation: mutation
        :param shift: shift
        :return: shifted mutation
        """
        if 'del' in mutation or 'ins' in mutation or 'Del' in mutation or 'd' in mutation:
            return mutation
        
        shifted_mutation = []
        if re.match(r'\b[A-Z]\d{1,4}[A-Z]\b', mutation):
            let_s,num,let_e = mutation[0],mutation[1:-1],mutation[-1]
            for s in shift:
                shifted_num = int(num) + s
                if shifted_num >= 0:
                    shifted_num = str(shifted_num)
                    shifted_mutation.append(f"{let_s}{shifted_num}{let_e}")
            return ','.join(shifted_mutation)
        
    # Special case (e.g., Sins.)
        elif mutation.lower() == 'sins.':
            return mutation
        else:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
        
    def find_and_shift(self, mutation, uniprot_set):
        """
        Cerca la mutazione in uniprot o prova gli shift.
        param mutation: mutazione
        param uniprot_set: set di mutazioni conosciute
        return: True se la mutazione è presente in uniprot, False altrimenti
        e la mutazione shiftata"""
        if mutation in uniprot_set:
            return True, self.shift_mutation(mutation, self.shift)
        else:
            return False, self.shift_mutation(mutation, self.shift)

    def find_mutant(self, mut: pd.DataFrame, all_mut: set):
        """
        Find mutations of any type in the assay description (single, double, triple, wild type)
        :param mut: dataframe with mutations to be found
        :param all_mut: set of known mutations
        :return: dataframe with mutations
        """
        patterns = [
        r'\b[A-Z]\d{1,4}[A-Z]\b',  # Mutazione singola, e.g., L747S
        r'\b[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z]del)?(?:/[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z]del)?|-[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z]del)?|)[A-Z]?\b',  # Mutazione doppia, e.g., L747S-T751del o L747S/T751del
        r'\b[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z]del)?(?:/[A-Z]\d+[A-Z](-[A-Z]\d+[A-Z]del)?){1,2}\b',  # Mutazione tripla, e.g., L747S-T751del/M752del
        r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}del\b',  # Mutazione d'intervallo, e.g., L747-T751del
        r'\b[A-Z]\d{1,4}-[A-Z]\d{1,4}\s*ins', # Mutazione di inserzione, e.g., D770-N771ins
        r'\bDel\s*\d{1,4}\b',  # Delezione, e.g., Del19
        r'\bSins\.\b',  # Inserzione, e.g., Sins.
        r'\b[A-Z]\d{1,4}_[A-Z]\d{1,4}\s*ins',  # Inserzione tra due amminoacidi, e.g., A763_Y764ins
        r'\bDel [A-Z]\d{1,4}/[A-Z]\d{1,4}\b',  # Delezione tra due amminoacidi con separatore di barra, e.g., Del E746/A750
        r'\bex\d{1,2}del\b',  # Delezione con notazione esone, e.g., ex19del
        r'\bdel\s*(\d{1,4} to \d{1,4}\s*)\b',  # Delezione con intervallo numerico tra parentesi, e.g., del (746 to 750)
        r'\bd(\d{1,4}-\d{1,4})\/([A-Z]\d{1,4}[A-Z])\b',  # Delezione con intervallo numerico e mutazione, e.g., d746-750/L858R
        ]
        combined_pattern = re.compile('|'.join(patterns))

        mutant = mut.copy()
        for index, row in mutant.iterrows():
            assay_description = row['Assay Description']
            match_mut = combined_pattern.finditer(assay_description)
            if match_mut:
                mutations_found = []
                shifted_mutations = []
                known_flags = []
                for match in match_mut:
                    mutation = match.group()
                    is_known, shifted = self.find_and_shift(mutation, all_mut)
                    mutations_found.append(mutation)
                    shifted_mutations.append(shifted)
                    known_flags.append(str(is_known))

            # Process found mutations
                if mutations_found:
                    mutant.loc[index, 'mutant_known'] = '/'.join(known_flags)
                    mutant.loc[index, 'mutant'] = '/'.join(mutations_found)
                    mutant.loc[index, 'shifted_mutation'] = '/'.join(shifted_mutations)

        return mutant
    
    def format_output(self,no_mut,mut,known_mutations):
        """Formatting final output
        param no_mut: dataframe with no mutations
        param mut: dataframe with mutations
        param known_mutations: dictionary with known mutations
        """

        mut = mut.sort_values(by='mutant')
        wt = re.compile(r'\b(wild type|wild_type)\b')

        for index, row in mut.iterrows():
            mutant_value = row['mutant']
            assay_description = row['Assay Description']
        
        # Set 'wild type' if 'mutation' is empty and 'Assay Description' contains 'wild type'
            if row['mutant'] == '' and wt.search(assay_description):
                mut.loc[index, 'mutant'] = 'wild type'

        # Check known mutations and assign accession code
            if not row.get('mutant_known', False):
                continue
        
            for key, mutations_list in known_mutations.items():
                if mutant_value in mutations_list:
                    accession_code = key[0]  # Assuming the first element of the key tuple is the Accession Code
                    mut.at[index, 'Accession Code'] = accession_code
                    break  # Stop searching once we find the Accession Code 
        no_mut.loc[:,'mutant'] = 'mixed'
        final= pd.concat([no_mut,mut],ignore_index=True)
        return final,mut
