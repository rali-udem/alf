

import re
import random

from lexnet import get_df
from lexnet import get_relation_dict
from lexnet import get_para_synta_relations
from lexnet import get_word_from_id_mapping


def get_quadruples(id_bool=False, merged_bool=False, custom_file=""):

    suffix = ""
    if id_bool:
        suffix = "_id"

    elif merged_bool:
        suffix = "_id_merged"

    if custom_file == "":
        with open(f"relations{suffix}.txt") as f:
            lines = f.readlines()
    else:
        with open(custom_file) as f:
            lines = f.readlines()


    quadruples_list = []

    # pattern = r'^(.*?) \((.*?)\) =  (.*?) \[.*\]$'
    pattern = r'^(.*?) \((.*?)\) = (.*?) \[(.*?), ([0-1])\]$'

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            relation, source, target, family, merged_value = match.groups()
            quadruples_list.append((source, relation, target, family, merged_value))

    return quadruples_list


def create_dict(quadruples_list, precise=False):
    relation_dict = {}

    # Iterate over the quadriples
    for source, relation, target, family in quadruples_list:
        # Check if the relation already exists in the dictionary

        if precise:     # if we want the precise relations, not the broader families
            if relation not in relation_dict:
                # If it doesn't exist, create a new list with the current pair
                relation_dict[relation] = [(source, target)]
            else:
                # If it exists, append the current pair to the existing list
                relation_dict[relation].append((source, target))

        else:           # if we want the broader family relations
            if family not in relation_dict:
                # If it doesn't exist, create a new list with the current pair
                relation_dict[family] = [(source, target)]
            else:
                # If it exists, append the current pair to the existing list
                relation_dict[family].append((source, target))

    return relation_dict


def create_dict_with_merge(quadruples_list, precise=False):
    relation_dict = {}

    # Iterate over the quadriples
    for source, relation, target, family, merged_value in quadruples_list:
        # Check if the relation already exists in the dictionary

        if precise:     # if we want the precise relations, not the broader families
            if relation not in relation_dict:
                # If it doesn't exist, create a new list with the current pair
                relation_dict[relation] = [(source, target, merged_value)]
            else:
                # If it exists, append the current pair to the existing list
                relation_dict[relation].append((source, target, merged_value))

        else:           # if we want the broader family relations
            if family not in relation_dict:
                # If it doesn't exist, create a new list with the current pair
                relation_dict[family] = [(source, target, '')]
            else:
                # If it exists, append the current pair to the existing list
                relation_dict[family].append((source, target, ''))

    return relation_dict


def filter(relation_dict, n=100):
    important_relations_dict = {}

    for key in relation_dict:
        # for nominal clone removal
        if len(list(set(relation_dict[key][1]))) >= n:
            important_relations_dict[key] = (relation_dict[key][0], relation_dict[key][1], relation_dict[key][2])

    return important_relations_dict


def get_count_dict(relation_dict):

    pair_count_dict = {}

    keys = list(relation_dict.keys())

    for key in keys:
        pair_count_dict[key] = len(relation_dict[key])

    sorted_pair_count_dict = dict(sorted(pair_count_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_pair_count_dict


def get_examples(relation_dict):
    example_pairs_dict = {}
    keys = list(relation_dict.keys())

    for key in keys:
        example1, example2 = random.sample(relation_dict[key], 2)
        example_pairs_dict[key] = (example1, example2)

    return example_pairs_dict


def extract_in_file(sampled_pairs_file, model_response_file, write_filename, mapping=None, synonyms_dict=None):
    with open(sampled_pairs_file) as f:
        content1 = f.read()

    with open(model_response_file) as f:
        content2 = f.read()

    relation_dict = eval(content1)
    response_dict = eval(content2)

    for rel in relation_dict:
        new_pair = (relation_dict[rel][0][10:], relation_dict[rel][1][10:], relation_dict[rel][2][10:])
        relation_dict[rel] = new_pair


    ### one way, which is directly with the names of the words (rather than ids)
    # for key in relation_dict:
    #     with open(f"{write_filename}/output/{key}_output.txt", "w") as g:
    #         for i, metapair in enumerate(relation_dict[key][1]):
    #             pair1, pair2 = metapair
    #             source1, target1 = pair1
    #             source2, target2 = pair2
    #
    #             g.write(f"IN: {source1}: {target1} :: {source2}: {target2} || «{response_dict[key][i]}»\n{100 * '-'}\n")

    ### another way, which is via the ids and then the mapping to the words
    for key in relation_dict:
        with open(f"{write_filename}/output_via_ids/{key}_output.txt", "w") as g:
            for i, metapair in enumerate(relation_dict[key][0]):
                pair1, pair2 = metapair
                source1, target1 = pair1
                source2, target2 = pair2

                g.write(f"IN: {source1}: {target1} :: {source2}: {target2} || «{response_dict[key][i]}»\n{100 * '-'}\n")


def extract_in_file2(sampled_pairs_file, model_response_file, write_filename, name_mapping=None, synonyms_dict=None):
    with open(sampled_pairs_file) as f:
        content1 = f.read()

    with open(model_response_file) as f:
        content2 = f.read()

    relation_dict = eval(content1)
    response_dict = eval(content2)

    for rel in relation_dict:
        new_pair = (relation_dict[rel][0][10:], relation_dict[rel][1][10:], relation_dict[rel][2][10:])
        relation_dict[rel] = new_pair

    ### one way, which is directly with the names of the words (rather than ids)
    # for key in relation_dict:
    #     with open(f"{write_filename}/output/{key}_output.txt", "w") as g:
    #         for i, metapair in enumerate(relation_dict[key][1]):
    #             pair1, pair2 = metapair
    #             source1, target1 = pair1
    #             source2, target2 = pair2
    #
    #             g.write(f"IN: {source1}: {target1} :: {source2}: {target2} || «{response_dict[key][i]}»\n{100 * '-'}\n")

    ### another way, which is via the ids and then the mapping to the words

    for key in relation_dict:
        with open(f"{write_filename}/{key}_output.txt", "w") as g:
            for i, metapair in enumerate(relation_dict[key][0]):
                pair1, pair2 = metapair
                source1, target1, merged = pair1
                source2, target2, merged = pair2



                try:
                    equivalent_targets = synonyms_dict[key][source2]
                except KeyError:
                    print(synonyms_dict[key])
                    print()
                    print(f"{key}\t {target2}")


                source1_new = name_mapping[source1]
                target1_new = name_mapping[target1]
                source2_new = name_mapping[source2]
                target2_new = name_mapping[target2]


                # equivalent_targets2 = []
                # for target_id in equivalent_targets:
                #     target_name = name_mapping[target_id]
                #     equivalent_targets2.append(target_name)

                g.write(
                    f"IN: {source1_new}: {target1_new} :: {source2_new}: {target2_new} || «{response_dict[key][i]}»\t {str(equivalent_targets)}\n{100 * '-'}\n")

def extract_in_file3(sampled_pairs_file, model_response_file, write_filename, name_mapping, synonyms_dict, tajp):
    with open(sampled_pairs_file) as f:
        content1 = f.read()

    with open(model_response_file) as f:
        content2 = f.read()

    relation_dict = eval(content1)
    response_dict = eval(content2)

    for rel in relation_dict:
        new_pair = (relation_dict[rel][0][10:], relation_dict[rel][1][10:], relation_dict[rel][2][10:])
        relation_dict[rel] = new_pair


    for key in relation_dict:

        if tajp == "synta1":
            pseudo_key = key + '*'

        else:
            pseudo_key = key

        with open(f"{write_filename}/{key}_output.txt", "w") as g:
            for i, metapair in enumerate(relation_dict[key][0]):
                pair1, pair2 = metapair
                source1, target1, merged = pair1
                source2, target2, merged = pair2


                try:
                    equivalent_targets = synonyms_dict[pseudo_key][source2]
                except KeyError:
                    print(synonyms_dict[key])
                    print()
                    print(f"{key}\t {target2}")


                source1_new = name_mapping[source1]
                target1_new = name_mapping[target1]
                source2_new = name_mapping[source2]
                target2_new = name_mapping[target2]



                g.write(
                    f"IN: {source1_new}: {target1_new} :: {source2_new}: {target2_new} || «{response_dict[key][i]}»\t {str(equivalent_targets)}\n{100 * '-'}\n")




def check_repeated_samples(sample_filename):
    clone_dict = {}

    with open(sample_filename) as f:
        content = f.read()
    sample_dict = eval(content)


    for key in sample_dict:
        count = 0
        big_count = 0
        for metapair in sample_dict[key]:
            pair1, pair2 = metapair
            source1, target1 = pair1
            source2, target2 = pair2

            if pair1 == pair2:
                count += 1
                print(f"yes for {key}")

            big_count += 1

        clone_dict[key] = count/big_count * 100

    return clone_dict


def get_relevant_arrays():
    LLAMA_SAMPLES = ["runs/llama3/prompt1/sample_dict.txt",
                     "runs/llama3/prompt2/sample_dict_llama3-p2.txt",
                    "runs/llama3/prompt3/sample_dict_llama3-p3.txt"]

    GEMMA2b_SAMPLES = ["runs/gemma2b/prompt1/sample_dict.txt",
                       "runs/gemma2b/prompt2/sample_dict_gemma2b-p2.txt",
                       "runs/gemma2b/prompt3/sample_dict_gemma2b-p3.txt"]

    GEMMA7b_SAMPLES = ["runs/gemma7b/prompt1/sample_dict_gemma7b-p1.txt",
                       "runs/gemma7b/prompt2/sample_dict_gemma7b-p2.txt",
                       "runs/gemma7b/prompt3/sample_dict_gemma7b-p3.txt"]

    SAMPLES = [LLAMA_SAMPLES, GEMMA2b_SAMPLES, GEMMA7b_SAMPLES]

    ##############################


    LLAMA_RESPONSES = ["runs/llama3/prompt1/model_response_dict.txt",
                       "runs/llama3/prompt2/model_response_dict_llama3-p2.txt",
                       "runs/llama3/prompt3/model_response_dict_llama3-p3.txt"]

    GEMMA2b_RESPONSES = ["runs/gemma2b/prompt1/model_response_dict.txt",
                         "runs/gemma2b/prompt2/model_response_dict_gemma2b-p2.txt",
                         "runs/gemma2b/prompt3/model_response_dict_gemma2b-p3.txt"]

    GEMMA7b_RESPONSES = ["runs/gemma7b/prompt1/model_response_dict_gemma7b-p1.txt",
                         "runs/gemma7b/prompt2/model_response_dict_gemma7b-p2.txt",
                         "runs/gemma7b/prompt3/model_response_dict_gemma7b-p3.txt"]

    RESPONSES = [LLAMA_RESPONSES, GEMMA2b_RESPONSES, GEMMA7b_RESPONSES]

    ############################################################


    LLAMA_LABELS = ["runs/llama3/prompt1/label_dict.txt",
                    "runs/llama3/prompt2/label_dict_llama3-p2.txt",
                    "runs/llama3/prompt3/label_dict_llama3-p3.txt"]


    GEMMA2b_LABELS = ["runs/gemma2b/prompt1/label_dict.txt",
                      "runs/gemma2b/prompt2/label_dict_gemma2b-p2.txt",
                      "runs/gemma2b/prompt3/label_dict_gemma2b-p3.txt"]


    GEMMA7b_LABELS = ["runs/gemma7b/prompt1/label_dict_gemma7b-p1.txt",
                      "runs/gemma7b/prompt2/label_dict_gemma7b-p2.txt",
                      "runs/gemma7b/prompt3/label_dict_gemma7b-p3.txt"]

    LABELS = [LLAMA_LABELS, GEMMA2b_LABELS, GEMMA7b_LABELS]

    ##############################


    LLAMA_OUTPUTS = ["runs/llama3/prompt1", "runs/llama3/prompt2", "runs/llama3/prompt3"]
    GEMMA2b_OUTPUTS = ["runs/gemma2b/prompt1", "runs/gemma2b/prompt2", "runs/gemma2b/prompt3"]
    GEMMA7b_OUTPUTS = ["runs/gemma7b/prompt1", "runs/gemma7b/prompt2", "runs/gemma7b/prompt3"]

    OUTPUTS = [LLAMA_OUTPUTS, GEMMA2b_OUTPUTS, GEMMA7b_OUTPUTS]


    return SAMPLES, RESPONSES, LABELS, OUTPUTS


# to revisit
def create_example_pairs():
    quadruples_id = get_quadruples(True)
    quadruples = get_quadruples(False)

    relation_dict_id = create_dict(quadruples_id, precise=True)
    relation_dict = create_dict(quadruples, precise=True)

    with open("example_dict_id") as f:
        example_dict_id = eval(f.read())


    example_pairs = {}
    for key in relation_dict_id:
        list_of_pairs = relation_dict_id[key]

        list_of_new_pairs = []
        for i, pair in enumerate(list_of_pairs):
            source_id, target_id = pair

            try:
                source_example = example_dict_id[source_id][0]
                target_example = example_dict_id[target_id][0]

            except KeyError:
                source_example = ""
                target_example = ""

            new_pair = (source_example, target_example)
            list_of_new_pairs.append(new_pair)

        example_pairs[key] = list_of_new_pairs

    return relation_dict_id, relation_dict, example_pairs

def create_mega_dict(id_dict, name_dict, sentence_example_dict):
    mega_dict = {}
    for key in id_dict:     # since, by assumption, we assume all dicts to have the same keys
        mega_pair = (id_dict[key], name_dict[key], sentence_example_dict[key])
        mega_dict[key] = mega_pair
    return mega_dict


def split_dict_in3(my_dict, para, synta):
    para_dict = {}

    for key in my_dict:
        if key in para:
            para_dict[key] = my_dict[key]


    synta_dict0 = {}
    synta_dict1 = {}

    for key in synta:
        if key == "Oper_4":     # I think there are no relations w this function in the data
            continue

        for item in my_dict[key]:
            source, target, merged_value = item

            if merged_value == '0':
                if key not in synta_dict0:
                    synta_dict0[key] = [item]
                else:
                    synta_dict0[key].append(item)

            elif merged_value == '1':
                if key not in synta_dict1:
                    synta_dict1[key] = [item]
                else:
                    synta_dict1[key].append(item)
            else:
                raise ValueError("neither pragmatic nor syntagmatic")

    return para_dict, synta_dict0, synta_dict1

def translate_id_dict_to_name_dict(id_dict, mapping):

    name_dict = {}
    for key in id_dict:
        new_item = []
        for item in id_dict[key]:
            source, target, merged_value = item
            new_source = mapping[source]
            new_target = mapping[target]
            new_item.append((new_source, new_target, merged_value))
        name_dict[key] = new_item
    return name_dict


def find_examples_for_id_dict(my_dict):
    with open("example_dict_id") as f:
        example_mapping = eval(f.read())

    example_dict = {}
    for key in my_dict:
        new_item = []
        for item in my_dict[key]:
            source, target, merged_value = item
            try:
                source_example = example_mapping[source]
                target_example = example_mapping[target]
            except KeyError:
                source_example = [""]
                target_example = [""]

            new_item.append((source_example, target_example, merged_value))

        example_dict[key] = new_item

    return example_dict

def mega_split_protocol():
    # Wednesday, July 17, 2024

    quadruples = get_quadruples(False, True)

    my_dict = create_dict_with_merge(quadruples, precise=True)

    para, synta = get_para_synta_relations()

    para_dict, synta_dict0, synta_dict1 = split_dict_in3(my_dict, para, synta)

    mapping = get_word_from_id_mapping()

    name_para_dict = translate_id_dict_to_name_dict(para_dict, mapping)
    name_synta_dict0 = translate_id_dict_to_name_dict(synta_dict0, mapping)
    name_synta_dict1 = translate_id_dict_to_name_dict(synta_dict1, mapping)

    para_example_dict = find_examples_for_id_dict(para_dict)
    synta_example_dict0 = find_examples_for_id_dict(synta_dict0)
    synta_example_dict1 = find_examples_for_id_dict(synta_dict1)

    def mega_split_dict():
        para_mega_dict = create_mega_dict(para_dict, name_para_dict, para_example_dict)
        synta_mega_dict0 = create_mega_dict(synta_dict0, name_synta_dict0, synta_example_dict0)
        synta_mega_dict1 = create_mega_dict(synta_dict1, name_synta_dict1, synta_example_dict1)

        # with open("para_mega_dict.txt", 'w') as f:
        #     f.write(str(para_mega_dict))
        # with open("synta_mega_dict0.txt", 'w') as f:
        #     f.write(str(synta_mega_dict0))
        # with open("synta_mega_dict1.txt", 'w') as f:
        #     f.write(str(synta_mega_dict1))

        return para_mega_dict, synta_mega_dict0, synta_mega_dict1

    return mega_split_dict()
    # Край на функцията



def filter_and_sample(data_dict, n):
    new_dict = {}

    for key, (list1, list2, list3) in data_dict.items():
        seen = set()
        unique_indices = []

        # Find indices of unique elements in the second list
        for idx, item in enumerate(list2):
            if item not in seen:
                seen.add(item)
                unique_indices.append(idx)

        # If n is larger than the number of unique items, adjust n
        # n = min(n, len(unique_indices))

        # Randomly sample n indices from the unique indices
        sampled_indices = random.sample(unique_indices, n)

        # Filter the lists based on sampled indices
        new_list1 = [list1[idx] for idx in sampled_indices]
        new_list2 = [list2[idx] for idx in sampled_indices]
        new_list3 = [list3[idx] for idx in sampled_indices]

        # Update the dictionary with the filtered and sampled lists
        new_dict[key] = (new_list1, new_list2, new_list3)

    return new_dict


def regroup_sample(data_dict, sample_size=220):
    n = sample_size - 20
    m = int(n / 2)




    regrouped_sample = {}

    for lf in data_dict:
        components = []
        for i in range(3):
            sample1 = data_dict[lf][i][:m]
            sample2 = data_dict[lf][i][m:n]
            sample3 = data_dict[lf][i][n:]

            kshota = sample3[:10]
            kshotb = sample3[10:]

            kshots = list(zip(kshota, kshotb))


            sampled_pairs = list(zip(sample1, sample2))
            sampled_pairs = kshots + sampled_pairs

            components.append(sampled_pairs)
        regrouped_sample[lf] = (components[0], components[1], components[2])

    return regrouped_sample



















































































