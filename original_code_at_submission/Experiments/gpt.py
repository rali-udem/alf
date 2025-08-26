import os
import logging
import openai
from dotenv import load_dotenv
from write_output_files_auto import auto_output
from itertools import product
import time

# Load environment variables
load_dotenv()

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)


def get_response(prompt_list):
    try:
        logging.info(f"Sending prompt with {len(prompt_list)} messages.")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_list,
        )
        response_text = response.choices[0].message.content
    except openai.APIError as e:
        logging.error(f"API error: {e}")
        response_text = "Error in API request"
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        response_text = "Unexpected error occurred"
    
    return response_text


def get_synonyms(mapping, rel, label_id):
    return mapping[rel][label_id]


def get_kshot_content(kshots, english, k):
    content = ["Consider the following examples"] if english else ["Considère les exemples suivants"]
    
    for i in range(k):
        shot_a, shot_b = kshots[i]
        content.append(f"{shot_a[0]} : {shot_a[1]} :: {shot_b[0]} : {shot_b[1]}")

    content.append("Now, use this pattern to solve the following analogy:" if english else 
                   "Maintenant, utilise ce patron pour résoudre l'analogie suivante:")
    
    return "\n".join(content)


def build_kshot_prompt(kshots, english, k):
    kshot_prompt = [{"role": "user", "content": get_kshot_content(kshots, english, k)}]
    return kshot_prompt


def build_prompt(pair, simple, english):
    base_content = f"{pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
    
    if english:
        if simple:
            return [{"role": "user", "content": f"Solve the following analogy. {base_content}"}]
        else:
            return [{"role": "user", "content": f"Consider this first term: {pair[0][0]}. "
                                                f"Consider this second term: {pair[0][1]}. "
                                                f"Consider this third term: {pair[1][0]}. "
                                                f"With this in mind, solve the following analogy. {base_content}"}]
    else:
        if simple:
            return [{"role": "user", "content": f"Résolvez l'analogie suivante. {base_content}"}]
        else:
            return [{"role": "user", "content": f"Considérez ce premier terme: {pair[0][0]}. "
                                                f"Considérez ce deuxième terme: {pair[0][1]}. "
                                                f"Considérez ce troisième terme: {pair[1][0]}. "
                                                f"Sachant cela, résolvez l'analogie suivante. {base_content}"}]


def prompt_set_up_no_hist(sample_dict, use_kshots, simple, english, k):
    prompt_dict = {}
    label_dict = {}
    source_id_dict = {}

    for relation in sample_dict:
        prompts = []
        source_ids = []
        labels = []

        kshots = sample_dict[relation][1][:10]

        kshot_prompt = []
        if use_kshots:
            kshot_prompt = build_kshot_prompt(kshots, english, k)

        for i, (id_pair, pair, _) in enumerate(zip(sample_dict[relation][0][10:], sample_dict[relation][1][10:], sample_dict[relation][2][10:])):
            prompt = build_prompt(pair, simple, english)
            if use_kshots:
                full_prompt = kshot_prompt + prompt
            else:
                full_prompt = prompt

            prompts.append(full_prompt)
            labels.append(pair[1][1])
            source_ids.append(id_pair[1][0])

        prompt_dict[relation] = prompts
        label_dict[relation] = labels
        source_id_dict[relation] = source_ids

    return prompt_dict, label_dict, source_id_dict


def prompt_set_up_hist(sample_dict, use_kshots, simple, english, k):
    prompt_dict = {}
    label_dict = {}
    source_id_dict = {}

    for relation in sample_dict:
        prompts = []
        source_ids = []
        labels = []

        kshots = sample_dict[relation][1][:10]

        kshot_prompt = []
        if use_kshots:
            kshot_prompt = build_kshot_prompt(kshots, english, k)

        for i, (id_pair, pair, _) in enumerate(zip(sample_dict[relation][0][10:], sample_dict[relation][1][10:], sample_dict[relation][2][10:])):
            prompt = build_prompt(pair, simple, english)
            if i == 0 and use_kshots:
                full_prompt = kshot_prompt + prompt
            else:
                full_prompt = prompt

            prompts.append(full_prompt)
            labels.append(pair[1][1])
            source_ids.append(id_pair[1][0])

        prompt_dict[relation] = prompts
        label_dict[relation] = labels
        source_id_dict[relation] = source_ids

    return prompt_dict, label_dict, source_id_dict


def provide_feedback(model_response, english, relation, source_id_dict, syn_map, index, tajp, partial_prompt):
    if tajp == "synta1":
        pseudo_relation = relation + '*'
    else:
        pseudo_relation = relation

    corrects = get_synonyms(syn_map, pseudo_relation, source_id_dict[relation][index])
    skip = False
    for correct in corrects:
        if correct.lower() in model_response.lower():
            praise_string = "Good job! This is one of the answers we were looking for." if english else \
                            "Bon travail! Ceci est une des réponses que l'on cherchait."
            praise_message = {"role": "user", "content": praise_string}
            partial_prompt.append(praise_message)
            skip = True
            break

    if not skip:
        strike_string = f"Wrong answer. Examples of correct answers we were looking for were: {str(corrects)}." if english else \
                        f"Mauvaise réponse. Des exemples de bonnes réponses que l'on cherchait étaient: {str(corrects)}."
        strike_message = {"role": "user", "content": strike_string}
        partial_prompt.append(strike_message)


def model_session_history(prompt_dict, english, source_id_dict, syn_map, feedback_mode, tajp):
    model_response_dict = {}
    partial_prompts_dict = {}

    for r, relation in enumerate(prompt_dict):
        model_responses = []
        partial_prompt = [{"role": "system", "content": "You are an expert on French analogies. You always respond in one French lexeme."}] if english else \
                         [{"role": "system", "content": "Vous êtes un expert des analogies françaises. Vous répondez toujours en une seule lexie française."}]

        for i, prompt in enumerate(prompt_dict[relation]):
            partial_prompt += prompt
            model_response = get_response(partial_prompt)
            assistant_response = [{"role": "assistant", "content": model_response}]
            partial_prompt += assistant_response
            model_responses.append(model_response)

            if feedback_mode:
                provide_feedback(model_response, english, relation, source_id_dict, syn_map, i, tajp, partial_prompt)

        partial_prompts_dict[relation] = partial_prompt
        model_response_dict[relation] = model_responses

    return model_response_dict, partial_prompts_dict


def model_session_no_hist(prompt_dict, english):
    model_response_dict = {}
    partial_prompts_dict = {}

    for r, relation in enumerate(prompt_dict):
        model_responses = []
        chat = []
        system_prompt = [{"role": "system", "content": "You are an expert on French analogies. You always respond in one French lexeme."}] if english else \
                        [{"role": "system", "content": "Vous êtes un expert des analogies françaises. Vous répondez toujours en une seule lexie française."}]

        for i, prompt in enumerate(prompt_dict[relation]):
            actual_prompt = system_prompt + prompt
            model_response = get_response(actual_prompt)
            model_responses.append(model_response)
            chat.append(actual_prompt + [{"role": "assistant", "content": model_response}])

        partial_prompts_dict[relation] = chat
        model_response_dict[relation] = model_responses

    return model_response_dict, partial_prompts_dict


if __name__ == "__main__":

    safety = False
    confirmation = input("Are you sure you want to start? This may overwrite files.\n")
    if confirmation.lower() in ["y", "yes"]:
        safety = True

    if safety:
        # Load sample data and synonym mapping from environment or relative paths
        sample_dir = os.getenv("SAMPLE_DIR", "samples")
        with open(os.path.join(sample_dir, "regrouped_para_sample.txt")) as f:
            regrouped_para_sample = eval(f.read())

        syn_map = { "relation1": {"term1": ["syn1", "syn2"]}, "relation2": {"term1": ["syn3", "syn4"]}}  # Example map

        prompt_dict, label_dict, source_id_dict = prompt_set_up_no_hist(regrouped_para_sample, use_kshots=True, simple=False, english=True, k=7)
        
        # Model session without history
        model_response_dict, partial_prompts_dict = model_session_no_hist(prompt_dict, english=True)

        # Print output for verification (or write to a file)
        auto_output(model_response_dict, "output.txt", safety=safety)

        logging.info("Session completed.")



