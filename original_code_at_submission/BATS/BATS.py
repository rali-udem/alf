import re
import os
import random
import time
import openai


# Define directories (generic placeholders)
def syn_map_creation():
    base_dirs = [
        "path/to/some_directory/1",
        "path/to/some_directory/2",
        "path/to/some_directory/3",
        "path/to/some_directory/4"
    ]

    for base_dir in base_dirs:
        letter = base_dir.split("/")[-1][0]  # Extract a generic letter (e.g., "1", "2", etc.)

        files = os.listdir(base_dir)
        sorted_files = sorted(files, key=lambda x: int(x[1:3]))

        synonym_map = {}
        for i, file in enumerate(sorted_files):
            with open(f"{base_dir}/{file}") as f:
                content = f.read()

                # Generic regex pattern
                pattern = r"(\w+)\t+([^\n]+)"
                matches = re.findall(pattern, content)

                for j, (left, right) in enumerate(matches):
                    right_list = right.split('/')
                    matches[j] = (left, right_list)

                synonym_map[f"{letter}{i+1}"] = {}

                for match in matches:
                    synonym_map[f"{letter}{i+1}"][match[0]] = match[1]

        # Save dictionary (if needed)
        # with open(f"path/to/output/{letter}_dict.txt", 'w') as f:
        #     f.write(str(synonym_map))


# Create BATS test data
def bats_test_creation():
    with open("path/to/syn_map.txt") as f:
        syn_map = eval(f.read())

    bats_test = {}

    for key in syn_map:
        bats_og_pairs = []
        for left in syn_map[key]:
            right = syn_map[key][left][0]
            bats_og_pairs.append((left, right))

        random.shuffle(bats_og_pairs)

        n = len(bats_og_pairs)
        first_half = bats_og_pairs[:n // 2]
        second_half = bats_og_pairs[n // 2:]

        bats_analogies = list(zip(first_half, second_half))

        bats_test[key] = bats_analogies

    # Save the bats test data (if needed)
    # with open("path/to/output/bats_test.txt", 'w') as f:
    #     f.write(str(bats_test))


#####################################

# OpenAI API integration
openai.api_key = "your-api-key"
client = openai.OpenAI(api_key=openai.api_key)

# Function to get a response from OpenAI's GPT
def get_response(prompt_list):
    try:
        print(len(prompt_list))

        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt_list,
        )

        response_text = response.choices[0].message.content

    except openai.APIError:
        print("Error in OpenAI API.")
        response_text = "Error occurred."

    return response_text


# Function to retrieve synonyms from the mapping
def get_synonyms(mapping, relation, label_id):
    return mapping[relation][label_id]


# Function to generate the k-shot content for the prompt
def get_kshot_content(kshots, english, k):
    content = []
    for i in range(k):
        shot_a, shot_b = kshots[i]
        content.append(f"{shot_a[0]} : {shot_a[1]} :: {shot_b[0]} : {shot_b[1]}")

    if english:
        content.append("Now, use this pattern to solve the following analogy:")
    else:
        content.append("Maintenant, utilise ce patron pour résoudre l'analogie suivante:")

    return "\n".join(content)


# Function to build the prompt for k-shots
def build_kshot_prompt(kshots, english, k):
    kshot_prompt = []
    kshot_prompt.append({"role": "user", "content": get_kshot_content(kshots, english, k)})
    return kshot_prompt


# Function to build the prompt for solving the analogy
def build_prompt(pair, simple, english):
    if english:
        if simple:
            return [
                {"role": "user", "content": f"Solve the following analogy. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "}
            ]
        else:
            return [
                {"role": "user", "content": (
                    f"Consider this first term: {pair[0][0]}. "
                    f"Consider this second term: {pair[0][1]}. "
                    f"Consider this third term: {pair[1][0]}. "
                    f"With this in mind, solve the following analogy. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
                )}
            ]
    else:
        if simple:
            return [
                {"role": "user", "content": f"Résolvez l'analogie suivante. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "}
            ]
        else:
            return [
                {"role": "user", "content": (
                    f"Considérez ce premier terme: {pair[0][0]}. "
                    f"Considérez ce deuxième terme: {pair[0][1]}. "
                    f"Considérez ce troisième terme: {pair[1][0]}. "
                    f"Sachant cela, résolvez l'analogie suivante. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
                )}
            ]


# Set up the prompts, labels, and source IDs
def prompt_set_up(sample_dict, use_kshots, simple, english, k):
    prompt_dict = {}
    label_dict = {}
    source_id_dict = {}

    for relation in sample_dict:
        prompts = []
        source_ids = []
        labels = []

        # kshots = sample_dict[relation][:5]
        kshots = []

        kshot_prompt = []
        if use_kshots:
            kshot_prompt = build_kshot_prompt(kshots, english, k)

        for i, pair in enumerate(sample_dict[relation]):
            prompt = build_prompt(pair, simple, english)
            if use_kshots:
                full_prompt = kshot_prompt + prompt
            else:
                full_prompt = prompt

            prompts.append(full_prompt)
            labels.append(pair[1][1])
            source_ids.append(pair[1][0])

        prompt_dict[relation] = prompts
        label_dict[relation] = labels
        source_id_dict[relation] = source_ids

    return prompt_dict, label_dict, source_id_dict


# Provide feedback for model response
def provide_feedback(model_response, english, relation, source_id_dict, syn_map, index, tajp, partial_prompt):
    corrects = get_synonyms(syn_map, relation, source_id_dict[relation][index])
    skip = False
    for correct in corrects:
        if correct.lower() in model_response.lower():
            praise_string = "Good job! This is one of the answers we were looking for." if english else "Bon travail! Ceci est une des réponses que l'on cherchait."
            praise_message = {"role": "user", "content": praise_string}
            partial_prompt.append(praise_message)

            response_to_feedback = get_response(partial_prompt)
            assistant_message = {"role": "assistant", "content": response_to_feedback}
            partial_prompt.append(assistant_message)

            skip = True
            break

    if not skip:
        strike_string = f"Wrong answer. Examples of correct answers we were looking for were: {str(corrects)}."
        strike_message = {"role": "user", "content": strike_string}
        partial_prompt.append(strike_message)

        response_to_feedback = get_response(partial_prompt)
        assistant_message = {"role": "assistant", "content": response_to_feedback}
        partial_prompt.append(assistant_message)


# Run the model session with prompts and provide feedback
def model_session(prompt_dict, english, source_id_dict, syn_map, feedback_mode, tajp):
    model_response_dict = {}
    partial_prompts_dict = {}

    for r, relation in enumerate(prompt_dict):
        model_responses = []
        partial_prompt = [{"role": "system", "content": "You are an expert on analogies. Respond in one lexeme." if english else "Vous êtes un expert des analogies. Répondez en une seule lexie."}]

        for i, prompt in enumerate(prompt_dict[relation]):
            partial_prompt += prompt
            model_response = get_response(partial_prompt)
            assistant_response = [{"role": "assistant", "content": model_response}]
            partial_prompt += assistant_response

            if feedback_mode:
                provide_feedback(model_response, english, relation, source_id_dict, syn_map, i, tajp, partial_prompt)

            model_responses.append(model_response)

        partial_prompts_dict[relation] = partial_prompt
        model_response_dict[relation] = model_responses

    return model_response_dict, partial_prompts_dict


if __name__ == "__main__":

    # Confirmation to start the process
    safety = False
    confirmation = input("Are you sure you want to start? This may overwrite files.\n")
    if confirmation.lower() == "y" or confirmation.lower() == "yes":
        safety = True

    if safety:
        # Load the sample data and synonym mapping
        with open("path/to/bats_test.txt") as f:
            bats_test = eval(f.read())

        with open("path/to/syn_map.txt") as f:
            syn_map = eval(f.read())

        i = 0
        simple_options = [True]
        english_options = [True]
        k_shots_options = [0]
        feedback_options = [True]

        for simple in simple_options:
            for english in english_options:
                for k in k_shots_options:
                    use_kshots = (k > 0)
                    for feedback_ness in feedback_options:
                        main_dir = "path/to/output/directory"
                        durations = []

                        start = time.time()
                        print(f"Processing configuration: is_simple={simple}, language={english}, k_shots={k}")

                        prompt_dict, label_dict, source_id_dict = prompt_set_up(bats_test, use_kshots, simple, english, k)
                        model_response_dict, chat_history = model_session(prompt_dict, english, source_id_dict, syn_map, feedback_ness, "")

                        with open(f"{main_dir}/model_response.txt", 'w') as f:
                            f.write(str(model_response_dict))

                        with open(f"{main_dir}/chat_history.txt", 'w') as g:
                            g.write(str(chat_history))

                        end = time.time()
                        duration = end - start
                        durations.append(duration)
                        print(f"Duration: {duration} seconds")

                        with open(f"{main_dir}/durations.txt", 'w') as f:
                            f.write(str(durations))

                        i += 1
