import re
import os
import random
import openai
import time

# Load your OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)


def syn_map_creation():
    base_dirs = [
        "data/BATS/1_Inflectional_morphology",
        "data/BATS/2_Derivational_morphology",
        "data/BATS/3_Encyclopedic_semantics",
        "data/BATS/4_Lexicographic_semantics"
    ]

    BATS_syn_map = {}
    for base_dir in base_dirs:
        letter = base_dir.split("/")[-1][0]
        files = os.listdir(base_dir)
        sorted_files = sorted(files, key=lambda x: int(x.split("_")[0]))

        for i, file in enumerate(sorted_files):
            with open(f"{base_dir}/{file}") as f:
                content = f.read()
                pattern = r"(\w+)\t+([^\n]+)"
                matches = re.findall(pattern, content)
                for j, (left, right) in enumerate(matches):
                    right_list = right.split('/')
                    matches[j] = (left, right_list)

                BATS_syn_map[f"{letter}{i+1}"] = {match[0]: match[1] for match in matches}

    return BATS_syn_map


def bats_test_creation():
    with open("data/syn_map.txt") as f:
        syn_map = eval(f.read())

    bats_test = {}

    for key, pairs in syn_map.items():
        bats_og_pairs = [(left, right[0]) for left, right in pairs.items()]
        random.shuffle(bats_og_pairs)

        mid = len(bats_og_pairs) // 2
        first_half, second_half = bats_og_pairs[:mid], bats_og_pairs[mid:]

        bats_analogies = list(zip(first_half, second_half))
        bats_test[key] = bats_analogies

    with open("data/BATS/bats_test.txt", 'w') as f:
        f.write(str(bats_test))


def get_response(prompt_list):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_list,
        )
        return response.choices[0].message.content
    except openai.APIError:
        print("Error occurred.")
        return "Error"


def build_kshot_prompt(kshots, english, k):
    content = [
        f"{shot_a[0]} : {shot_a[1]} :: {shot_b[0]} : {shot_b[1]}"
        for shot_a, shot_b in kshots[:k]
    ]
    content.append("Now, use this pattern to solve the following analogy:" if english else
                   "Maintenant, utilise ce patron pour résoudre l'analogie suivante:")
    return [{"role": "user", "content": "\n".join(content)}]


def build_prompt(pair, simple, english):
    if simple:
        content = (f"{pair[0][0]} is to {pair[0][1]} as {pair[1][0]} is to" if english else
                   f"Résolvez l'analogie suivante. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : ")
    else:
        content = (
            f"Consider this first term: {pair[0][0]}. "
            f"Consider this second term: {pair[0][1]}. "
            f"Consider this third term: {pair[1][0]}. "
            f"With this in mind, solve the following analogy. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
            if english else
            f"Considérez ce premier terme: {pair[0][0]}. "
            f"Considérez ce deuxième terme: {pair[0][1]}. "
            f"Considérez ce troisième terme: {pair[1][0]}. "
            f"Sachant cela, résolvez l'analogie suivante. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
        )
    return [{"role": "user", "content": content}]


def prompt_set_up(sample_dict, use_kshots, simple, english, k):
    prompt_dict, label_dict, source_id_dict = {}, {}, {}

    for relation, pairs in sample_dict.items():
        kshot_prompt = build_kshot_prompt([], english, k) if use_kshots else []
        prompts = [
            kshot_prompt + build_prompt(pair, simple, english)
            for pair in pairs
        ]
        prompt_dict[relation] = prompts
        label_dict[relation] = [pair[1][1] for pair in pairs]
        source_id_dict[relation] = [pair[1][0] for pair in pairs]

    return prompt_dict, label_dict, source_id_dict


if __name__ == "__main__":
    use_history = input("Use conversation history? (yes/no): ").lower() == 'yes'
    confirmation = input("Start processing? This may overwrite files. (yes/no): ").lower()

    if confirmation in {"y", "yes"}:
        with open("data/BATS/bats_test.txt") as f:
            bats_test = eval(f.read())
        with open("data/syn_map.txt") as f:
            syn_map = eval(f.read())

        for simple in [True]:
            for english in [True]:
                for k in [0]:
                    prompt_dict, label_dict, source_id_dict = prompt_set_up(
                        bats_test, use_kshots=(k > 0), simple=simple, english=english, k=k)
                    print("Processing completed.")
