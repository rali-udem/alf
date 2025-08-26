import math
import re
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import pickle
import numpy as np


# Helper function to extract the desired content from raw HTML-like string
def extract_content(raw: str) -> str:
    pattern = r"<span class='namingform'>(.*?)</span>"
    s = re.search(pattern, raw).group(1)
    s = s.replace('\xa0', '')  # Remove non-breaking spaces
    return s


# General function to load a dataset from a given index
def load_data(index: int) -> pd.DataFrame:
    dataset_paths = {
        1: "path/to/first_dataset.csv",
        2: "path/to/second_dataset.csv",
        4: "path/to/fourth_dataset.csv",
        6: "path/to/sixth_dataset.csv",
        8: "path/to/eighth_dataset.csv",
        10: "path/to/tenth_dataset.csv",
        11: "path/to/eleventh_dataset.csv",
        13: "path/to/thirteenth_dataset.csv",
        15: "path/to/fifteenth_dataset.csv",
        17: "path/to/seventeenth_dataset.csv",
        18: "path/to/eighteenth_dataset.csv",
    }

    if index not in dataset_paths:
        raise ValueError(f"Dataset index {index} not found!")

    df = pd.read_csv(dataset_paths[index], sep='\t')
    return df.fillna(0)  # Fill NaN with zeros


# General function to extract the word form from a given ID
def extract_word_form_by_id(identifier: str) -> str:
    df = load_data(1)
    raw = df.loc[df['id'] == identifier].iloc[0]['lexname']
    return extract_content(raw)


# Build a mapping between IDs and words for a given dataset
def build_id_to_word_mapping() -> dict:
    df = load_data(1)
    mapping = {}
    
    for i in range(len(df)):
        word = extract_content(df["lexname"].iloc[i])
        identifier = df["id"].iloc[i]
        mapping[identifier] = word
        
    return mapping


# Function to retrieve word forms based on a list of IDs
def get_word_forms(ids: list, id_to_word_mapping: dict) -> list:
    words = [id_to_word_mapping[id] for id in tqdm(ids)]
    return words


# Extracts the source and target word forms or IDs from the relationship data
def get_relation_endpoints(df: pd.DataFrame, index: int, use_word_forms: bool = True) -> tuple:
    source_id = df.iloc[index]['source']
    target_id = df.iloc[index]['target']
    
    if use_word_forms:
        source = extract_word_form_by_id(source_id)
        target = extract_word_form_by_id(target_id)
    else:
        source, target = source_id, target_id

    return source, target


# Retrieve the merged value for a given index in the relationship data
def get_merged_value(df: pd.DataFrame, index: int) -> int:
    return df.iloc[index]["merged"]


# Parse an XML file to extract relationships between lexical functions
def load_relation_data() -> dict:
    tree = ET.parse("path/to/relationship_model.xml")
    root = tree.getroot()

    relations = {}
    for family in root.findall('.//family'):
        family_id = family.attrib['id']
        family_name = family.attrib['name']

        for lexicalfunction in family.findall('.//lexicalfunction'):
            lf_id = lexicalfunction.attrib['id']
            lf_name = lexicalfunction.attrib['name']
            lf_linktype = lexicalfunction.attrib["linktype"]

            relations[lf_id] = (lf_name, family_name, lf_linktype)

    return relations


# Extract the relationship name and family for a given index
def get_relation_name_by_index(df: pd.DataFrame, index: int) -> tuple:
    relation_id = df.iloc[index]['lf']
    relations = load_relation_data()
    return relations[relation_id][:2]  # Return (name, family)


# Write relationship data to an output file
def write_relationships_to_file(df: pd.DataFrame, use_word_forms: bool = True):
    N = len(df)
    
    with open("relationships_output.txt", "w") as f:
        for i in tqdm(range(N), desc="Writing relationships", ascii=True):
            source, target = get_relation_endpoints(df, i, use_word_forms)
            relation, family = get_relation_name_by_index(df, i)
            merged = get_merged_value(df, i)
            
            line = f"{relation} ({source}) = {target} [{family}, {merged}]\n"
            f.write(line)


# Count frequency of words based on their form in a dataset
def count_word_frequency(df: pd.DataFrame):
    names = [extract_content(df['lexname'][i]) for i in range(len(df))]
    freq_dict = Counter(len(name.split()) for name in names)

    print(freq_dict)
    
    num_words = list(freq_dict.keys())
    frequency = list(freq_dict.values())
    
    plt.bar(num_words, frequency, color='skyblue')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution')
    plt.show()

    return freq_dict


# Extract target words based on source ID and relation ID
def get_related_targets(source_id: str, relation_id: str) -> list:
    df = load_data(15)
    targets = []

    for i in range(len(df)):
        if df.iloc[i]["source"] == source_id and df.iloc[i]["lf"] == relation_id:
            targets.append((df.iloc[i]["target"], df.iloc[i]["form"], df.iloc[i]["separator"],
                            df.iloc[i]["merged"], df.iloc[i]["syntacticframe"], df.iloc[i]["constraint"],
                            df.iloc[i]["position"]))
    
    return targets


# Translate list of target words into their corresponding word forms
def translate_target_list(target_list: list) -> list:
    translated = []
    for target in target_list:
        target_id, form, separator, merge, syntax, constraint, position = target
        word = extract_word_form_by_id(target_id)
        translated.append((word, form, separator, merge, syntax, constraint, position))
    
    return translated


# Translate the relationship dictionary from IDs to word forms
def translate_relation_dict(id_to_word_mapping: dict) -> dict:
    df = load_data(15)
    relations = load_relation_data()
    
    translated_dict = {}
    for key in relations:
        name, family = relations[key][:2]

        for source_pair in df[df['lf'] == key].itertuples():
            source_word = extract_word_form_by_id(source_pair.source)
            target_list = translate_target_list(get_related_targets(source_pair.source, key))
            
            if name not in translated_dict:
                translated_dict[name] = []

            translated_dict[name].append((source_word, target_list))
    
    return translated_dict


# Helper function to remove duplicates from a dictionary of lists
def remove_duplicates_from_dict(my_dict: dict) -> dict:
    def make_hashable(tpl):
        """ Convert lists to tuples recursively to make them hashable """
        if isinstance(tpl, list):
            return tuple(make_hashable(item) for item in tpl)
        elif isinstance(tpl, tuple):
            return tuple(make_hashable(item) for item in tpl)
        else:
            return tpl

    def remove_duplicates(lst):
        seen = set()
        result = []
        for tpl in lst:
            tpl_hashable = make_hashable(tpl)
            if tpl_hashable not in seen:
                seen.add(tpl_hashable)
                result.append(tpl)
        return result

    for key in my_dict:
        my_dict[key] = remove_duplicates(my_dict[key])

    return my_dict


# Clean the dictionary by removing non-word information
def clean_synonyms_dict(my_dict: dict) -> dict:
    cleaned_dict = {}
    
    for relation in my_dict:
        source_pairs = []
        
        for source_element in my_dict[relation]:
            source, targets_list = source_element
            cleaned_targets = [target[0] for target in targets_list]
            source_pairs.append((source, cleaned_targets))
        
        cleaned_dict[relation] = source_pairs

    return cleaned_dict


# Translate synonym IDs via the ID to word mapping
def translate_synonym_ids(name_mapping: dict, relation_mapping: dict, synonyms: dict) -> dict:
    trans_synonyms = {}

    for rel in synonyms:
        new_source_pair_list = []

        for source_pair in synonyms[rel]:
            source_id, target_id_list = source_pair
            source_name = name_mapping[source_id]

            target_names = [name_mapping[target_id] for target_id in target_id_list]
            new_source_pair_list.append((source_name, target_names))

        rel_name = relation_mapping[rel][0]
        trans_synonyms[rel_name] = new_source_pair_list

    return remove_duplicates_from_dict(trans_synonyms)


# Get and return paradigmatic and syntagmatic relations
def get_paradigmatic_syntagmatic_relations() -> tuple:
    relations = load_relation_data()

    paradigmatic = []
    syntagmatic = []

    for key in relations:
        name, family, linktype = relations[key]
        if linktype == "paradigmatic":
            paradigmatic.append(name)
        elif linktype == "syntagmatic":
            syntagmatic.append(name)

    return paradigmatic, syntagmatic


# Main execution
if __name__ == "__main__":
    df = load_data(15)
    name_mapping = build_id_to_word_mapping()
    relation_mapping = load_relation_data()

    para, synta = get_paradigmatic_syntagmatic_relations()

    print(f"Number of paradigmatic relations: {len(para)}")
    print(f"Number of syntagmatic relations: {len(synta)}")
