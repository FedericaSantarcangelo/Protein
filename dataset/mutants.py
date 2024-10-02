from argparse import Namespace
import os
import sys
import re
import pandas as pd
from utils.file_utils import load_file, save_other_files
from utils.mutation import *

from utils.data_handling import patterns

aminoacid=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

class Mutation():
    def __init__(self, args: Namespace):
        self.args = args
        self.pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b|mutant|wild type|wild_type|\b[A-Z]\d{1,4}-[A-Z]\d{1,4}') 
        self.shift=[-2,-1,1,2]
        uniprot, mapping, organism = load_file(self.args.path_uniprot), load_file(self.args.path_mapping), load_file(self.args.path_organism)
        self.merged_uniprot = marge_data(self.args.path_output,organism, mapping, uniprot)

        
    def get_mutations(self, data: pd.DataFrame, flag='1'):
        """
        Get mutations main function
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: final dataframe with mutations and no mutations, mutation_report dataframe with mutations found
        """
        knonw_mutations,all_mut = self.format_uniprot(self.merged_uniprot)
        no_mut,mut=self.split_data(data.copy())
        mutant = self.find_mutant(mut,all_mut)
        mut, wild_type, mixed = self.format_output(no_mut,mutant,knonw_mutations, flag)
        return mut, wild_type, mixed
    
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
        uniprot.loc[:,'Known mutations'] = uniprot['Known mutations'].str.replace(r';+', ';', regex=True)  # remove multiple ;
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
        if re.match(r'\b[A-Z]\d{1,4}[A-Z]\b', mutation):
            shifted_mutation = []
            let_s,num,let_e = mutation[0],mutation[1:-1],mutation[-1]
            if let_s not in aminoacid or let_e not in aminoacid or let_e == let_s:
                return 'wrong'
            for s in shift:
                shifted_num = int(num) + s
                if shifted_num >= 0:
                    shifted_num = str(shifted_num)
                    shifted_mutation.append(f"{let_s}{shifted_num}{let_e}")
            return ','.join(shifted_mutation)

        else:
            return mutation
        
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
                if mutations_found:
                    if len(mutations_found) > 3: 
                        grouped_mutations = ["/".join(mutations_found[i:i+2]) for i in range(0, len(mutations_found), 2)]
                        grouped_shifted = ["/".join(shifted_mutations[i:i+2]) for i in range(0, len(shifted_mutations), 2)]
                        grouped_known = ["/".join(known_flags[i:i+2]) for i in range(0, len(known_flags), 2)]
                        
                        mutant.loc[index, 'mutant_known'] = '-'.join(grouped_known)
                        mutant.loc[index, 'mutant'] = '-'.join(grouped_mutations)
                        mutant.loc[index, 'shifted_mutation'] = '-'.join(grouped_shifted)
                    else:
                        mutant.loc[index, 'mutant_known'] = '/'.join(known_flags)
                        mutant.loc[index, 'mutant'] = '/'.join(mutations_found)
                        mutant.loc[index, 'shifted_mutation'] = '/'.join(shifted_mutations)
        return mutant
    
    def format_output(self,no_mut,mut,known_mutations,flag):
        """Formatting final output
        param no_mut: dataframe with no mutations
        param mut: dataframe with mutations
        param known_mutations: dictionary with known mutations
        """
        mut = mut.sort_values(by='mutant')
        wt = re.compile(r'\b(wild type|wild_type)\b')
        for index, row in mut.iterrows():
            assay_description = row['Assay Description']

            if row['mutant'] == '' and wt.search(assay_description):
                mut.loc[index, 'mutant'] = 'wild type'
            if not row.get('mutant_known', False):
                continue
            for key,_ in known_mutations.items():
                if row['Target ChEMBL ID'] == key[1]:
                    accession_code = key[0]
                    mut.loc[index, 'Accession Code'] = accession_code
                    break  # Stop searching once we find the Accession Code 
        wild_type = mut[mut['mutant'] == 'wild type'].copy()
        no_mut.loc[:,'mutant'] = 'mixed'
        mut = mut[mut['mutant'] != 'wild type']
        wrong,mut = find_mixed(mut)
        no_mut = pd.concat([no_mut, wrong], ignore_index=True)   
        mut = population(mut); no_mut = population(no_mut); wild_type = population(wild_type)
        save_other_files(no_mut, self.args.path_output ,'mixed', flag) 
        mut = save_mutation_target(self.args, mut, flag) 
        wild_type = save_mutation_target(self.args, wild_type ,flag, 'wild_type')
        return mut,wild_type,no_mut
