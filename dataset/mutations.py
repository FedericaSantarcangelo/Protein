from argparse import Namespace
import os
import sys
import re
import pandas as pd

class Mutation():
    def __init__(self, args: Namespace, data: pd.DataFrame):
        self.args = args
        self.data = data
        self.pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b|mutant|wild type| wt|wild_type',
                                   re.IGNORECASE) 
        self.shift=[-2,-1,1,2]

    def load_data(self):
        """
        Load data uniprot
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
        """
        Get mutations main function
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: mutations
        """
        uniprot = self.load_data()
        knonw_mutations,all_mut = self.format_uniprot(uniprot.copy())
        no_mut,mut=self.split_data(data.copy())
        single = self.single_mutation(all_mut, mut)
        double = self.double_mutation(all_mut, mut)
        triple = self.triple_mutation(all_mut, mut)
        wild = self.wild_type(all_mut, mut)
        final,mutation_report = self.format_output(no_mut,mut,
                                                knonw_mutations,single,
                                                double, triple,wild)
        return final, mutation_report
    
    def split_data(self, data: pd.DataFrame):
        """
        Split data considering mutations: if there are no mutations, 
        the row is added to no_mut, otherwise to mut
        :param data: data dataframe
        :return: no_mut, mut
        """
        data['mutation'] = data['Assay Description'].apply(
            lambda x: bool(self.pattern.search(x)))

        mut = data[data['mutation'] == True]
        no_mut = data[data['mutation'] == False]
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
        match = re.match(r"\b[A-Z]\d{1,4}[A-Z]\b", mutation)
        if not match:
            raise ValueError(f"Mutation {mutation} is not in the correct format")
    
        letter, number, last_letter = mutation[0], mutation[1:-1], mutation[-1]
        shifted_num=[]
        shifted_mutation=[]
        
        for s in shift:
            shifted_num = str(int(number) + s)
            m = f"{letter}{shifted_num}{last_letter}"
            shifted_mutation.append(m)
        
        shifted_mutation_str= ','.join(shifted_mutation)
        return shifted_mutation_str
    
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

    def single_mutation(self, uniprot, data: pd.DataFrame):
        """Get mutations
        :mutation_dict: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        :param data: data dataframe with mutations to be found
        :return: mutations
        """
        
        pattern = re.compile(r'\b[A-Z]\d{1,4}[A-Z]\b', re.IGNORECASE) 
        found_mut = []
        
        for index,row in data.iterrows():
            
            mutation = pattern.search(row['Assay Description'])
            if mutation:
                mutation = mutation.group()
                t,shifted_mutation = self.find_and_shift(mutation, uniprot)
                if t:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation,shifted_mutation))
                    data.loc[index, 'mutant_known'] = True
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation'] = shifted_mutation
                else:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation, shifted_mutation))
                    data.loc[index, 'mutant_known'] = False
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation'] = shifted_mutation
        return found_mut

    def double_mutation(self, uniprot, data: pd.DataFrame):
        """Get double mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: double mutations
        """
        pattern = re.compile(r'\b(?:[A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/[A-Z]\d+[A-Z]|-[A-Z]\d+[A-Z])?\b')
        found_mut = []
        for index, row in data.iterrows():
            mutation_match = pattern.search(row['Assay Description'])
            if mutation_match:
                mutation = mutation_match.group()
                split_mutations = re.split(r'\/|-', mutation)
                if len(split_mutations) != 2:
                    continue
                split1, split2 = split_mutations[0], split_mutations[1]
                t,mutations_shifted = self.find_and_shift(split1, uniprot)
                t_2,mutations_shifted_2 = self.find_and_shift(split2, uniprot)
                if t and t_2:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted,mutations_shifted_2))
                    data.loc[index, 'mutant_known'] = True
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+mutations_shifted_2
                else:
                    found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted,mutations_shifted_2))
                    data.loc[index, 'mutant_known'] = False
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+mutations_shifted_2
        return found_mut
    
    def triple_mutation(self, uniprot, data: pd.DataFrame):
        """Get triple mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: triple mutations
        """
        pattern = re.compile(r'\b([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)){2}\b')
        found_mut = []
        for index,row in data.iterrows():
            mutation_match = pattern.search(row['Assay Description'])
            if mutation_match:
                mutation = mutation_match.group()
                split=re.split(r'\/|-',mutation)
                if len(split) != 3:
                    continue
                split1,split2,split3=split[0],split[1],split[2]
                t,mutations_shifted= self.find_and_shift(split1,uniprot)
                t2,mutations_shifted_2=self.find_and_shift(split2,uniprot)
                t3,mutations_shifted_3=self.find_and_shift(split3,uniprot)
                if t and t2 and t3:
                    found_mut.append((row['Molecule ChEMBL ID'],mutation,mutations_shifted,mutations_shifted_2,mutations_shifted_3))
                    data.loc[index,'mutant_known']=True
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation']=mutations_shifted+'/'+mutations_shifted_2+'/'+mutations_shifted_3
                else:
                    found_mut.append((row['Molecule ChEMBL ID'],mutation,mutations_shifted,mutations_shifted_2,mutations_shifted_3))
                    data.loc[index,'mutant_known']=False
                    data.loc[index,'mutant'] = mutation
                    data.loc[index, 'shifted_mutation']=mutations_shifted+'/'+mutations_shifted_2+'/'+mutations_shifted_3
        return found_mut
    
    def wild_type(self, uniprot, data: pd.DataFrame):
        """Get wild type mutations
        :param uniprot: uniprot dataframe with mutations known
        :param data: data dataframe with mutations to be found
        :return: wild type mutations
        """
        pattern = re.compile(r'wild type| wt|wild_type', re.IGNORECASE)
        pattern_combined = re.compile(
            r'\b[A-Z]\d{1,4}[A-Z]\b|'  
            r'\b(?:[A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/[A-Z]\d+[A-Z]|-[A-Z]\d+[A-Z])?\b|'  
            r'\b([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)(?:\/([A-Z]\d+[A-Z](?:-[A-Z]\d+[A-Z]del)?)){2}\b', 
            re.IGNORECASE)
        found_mut = []

        for index, row in data.iterrows():
            wt = pattern.search(row['Assay Description'])
            if wt:
                mut_match=pattern_combined.search(row['Assay Description'])
                if mut_match:
                    mutation = mut_match.group()
                    split = re.split(r'\/|-', mutation)
                    if len(split) == 1:
                        t,shifted_mutation = self.find_and_shift(split[0], uniprot)
                        if t:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, shifted_mutation))
                            data.loc[index, 'mutant_known'] = True
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = shifted_mutation
                        else:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, shifted_mutation))
                            data.loc[index, 'mutant_known'] = False
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = shifted_mutation
                    elif len(split) == 2:
                        split1, split2 = split[0], split[1]
                        t,mutations_shifted = self.find_and_shift(split1, uniprot)
                        t2,mutations_shifted_2 = self.find_and_shift(split2, uniprot)
                        if t and t2:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted, mutations_shifted_2))
                            data.loc[index, 'mutant_known'] = True
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+ mutations_shifted_2
                        else:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted,mutations_shifted_2))
                            data.loc[index, 'mutant_known'] = False
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+mutations_shifted_2
                    elif len(split) == 3:
                        split1, split2, split3 = split[0], split[1], split[2]
                        t,mutations_shifted = self.find_and_shift(split1, uniprot)
                        t2,mutations_shifted_2=self.find_and_shift(split2, uniprot)
                        t3,mutations_shifted_3=self.find_and_shift(split3, uniprot)
                        if t and t2 and t3:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted, mutations_shifted_2, mutations_shifted_3))
                            data.loc[index, 'mutant_known'] = True
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+ mutations_shifted_2+'/'+ mutations_shifted_3
                        else:
                            found_mut.append((row['Molecule ChEMBL ID'], mutation, mutations_shifted, mutations_shifted_2, mutations_shifted_3))
                            data.loc[index, 'mutant_known'] = False
                            data.loc[index,'mutant'] = mutation
                            data.loc[index, 'shifted_mutation'] = mutations_shifted+'/'+mutations_shifted_2+'/'+mutations_shifted_3
        return found_mut
    
    
    def format_output(self, no_mut,mut,known_mutations,single,double,triple,wild):
        """Formatting final output
        param no_mut: dataframe with no mutations
        param mut: dataframe with mutations
        param knonw_mutations: dictionary with keys (Accession Code, CheMBL ID):[mutations]
        param single: single mutations list (Molecule ChEMBL ID, mutation, shifted mutation)
        param double: double mutations list (Molecule ChEMBL ID, mutation, shifted mutation)
        param triple: triple mutations list (Molecule ChEMBL ID, mutation, shifted mutation)
        param wild: wild type mutations list (Molecule ChEMBL ID, mutation, shifted mutation)
        """
        #create report
        row_to_move = mut[mut['shifted_mutation'].isna()].copy()
        row_to_move['mutation'] = False
        no_mut = pd.concat([no_mut, row_to_move], ignore_index=True)
        mut = mut[mut['shifted_mutation'].notna()]

        mut = mut.sort_values(by='mutant')
        mut['Accession Code'] = None

        for index, row in mut.iterrows():
            mutant_value = row['mutant']
            if not row['mutant_known']:
                continue 
            for key, mutations_list in known_mutations.items():
                if mutant_value in mutations_list:
                    accession_code = key[0]  # Assuming the first element of the key tuple is the Accession Code
                    mut.loc[index, 'Accession Code'] = accession_code
                    break  # Stop searching once we find the Accession Code 

        final= pd.concat([no_mut,mut],ignore_index=True)

        return final,mut
