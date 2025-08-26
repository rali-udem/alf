def extract_in_file(sampled_pairs_file, model_response_file, write_filename, synonyms_dict):
    with open(sampled_pairs_file) as f:
        content1 = f.read()

    with open(model_response_file) as f:
        content2 = f.read()

    relation_dict = eval(content1)
    response_dict = eval(content2)

    # Process the relation dictionary if needed
    # for rel in relation_dict:
    #     new_pair = (relation_dict[rel][0] relation_dict[rel][1][10:], relation_dict[rel][2][10:])
    #     relation_dict[rel] = new_pair

    # Loop through the relations
    for key in relation_dict:
        # Writing the output for each relation
        with open(f"{write_filename}/{key}_output.txt", "w") as g:
            for i, metapair in enumerate(relation_dict[key]):
                pair1, pair2 = metapair
                source1, target1 = pair1
                source2, target2 = pair2

                try:
                    # Look up equivalent targets in the synonym map
                    equivalent_targets = synonyms_dict[key][source2]
                except KeyError:
                    print(synonyms_dict[key])
                    print(f"{key}\t {target2}")

                # Write the output in the specified format
                g.write(
                    f"IN: {source1} : {target1} :: {source2}: {target2} || «{response_dict[key][i]}»\t {str(equivalent_targets)}\n{'-'*100}\n")

# Placeholder paths
sample_file = "path/to/sample_data.txt"
response_file = "path/to/model_response.txt"
output_dir = "path/to/output_directory"

# Load synonym map (use a placeholder path)
with open("path/to/synonym_map.txt") as f:
    synonym_map = eval(f.read())

# Run the extraction function
extract_in_file(sample_file, response_file, output_dir, synonym_map)

print("done")
