import json
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig)

import time
import os
from itertools import product

"""## Configuration"""

# HIST_MAX = 100
# print(f"HIST_MAX is {HIST_MAX}")

# Load config (e.g., containing HF token)
with open("indata/config.json") as f:
    config_data = json.load(f)
HF_TOKEN = config_data["HF_TOKEN"]

"""## Model Set-Up"""

model_nick_name = "mistral"
print(model_nick_name)

model_names = {
    "llama3.0_chat": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1_chat": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralMix": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralNemo": "mistralai/Mistral-Nemo-Instruct-2407"
}

model_name = model_names[model_nick_name]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

print("Model loaded successfully")

"""## Response Functions"""

def get_response_llama(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=64,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_response_qwen(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=64)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_response_mistral(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=64)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

# Choose response function depending on model
if "mistral" in model_nick_name:
    get_response = get_response_mistral
elif model_nick_name == "qwen":
    get_response = get_response_qwen
else:
    get_response = get_response_llama

"""## Prompt Utilities"""

def get_synonyms(mapping, rel, label_id):
    return mapping[rel][label_id]

def get_kshot_content(kshots, english, k):
    content = ["Consider the following examples"] if english else ["Considère les exemples suivants"]
    for i in range(k):
        shot_a, shot_b = kshots[i]
        content.append(f"{shot_a[0]} : {shot_a[1]} :: {shot_b[0]} : {shot_b[1]}")
    content.append("Now, use this pattern to solve the following analogy:" if english else "Maintenant, utilise ce patron pour résoudre l'analogie suivante:")
    return "\n".join(content)

def build_kshot_prompt(kshots, english, k):
    kshot_prompt = [{"role": "user", "content": get_kshot_content(kshots, english, k)}]
    if "mistral" in model_nick_name:
        kshot_prompt.append({"role": "assistant", "content": "ok"})
    return kshot_prompt

def build_prompt(pair, simple, english):
    if english:
        if simple:
            return [{"role": "user", "content": f"Solve the following analogy. {pair[0][0]}: {pair[0][1]}:: {pair[1][0]}: "}]
        else:
            return [{"role": "user", "content": (
                f"Consider this first term: {pair[0][0]}. "
                f"Consider this second term: {pair[0][1]}. "
                f"Consider this third term: {pair[1][0]}. "
                f"With this in mind, solve the following analogy. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
            )}]
    else:
        if simple:
            return [{"role": "user", "content": f"Résolvez l'analogie suivante. {pair[0][0]}: {pair[0][1]}:: {pair[1][0]}: "}]
        else:
            return [{"role": "user", "content": (
                f"Considérez ce premier terme: {pair[0][0]}. "
                f"Considérez ce deuxième terme: {pair[0][1]}. "
                f"Considérez ce troisième terme: {pair[1][0]}. "
                f"Sachant cela, résolvez l'analogie suivante. {pair[0][0]} : {pair[0][1]} :: {pair[1][0]} : "
            )}]

"""## Prompt Setup"""

def prompt_set_up_no_hist(sample_dict, use_kshots, simple, english, k):
    prompt_dict, label_dict, source_id_dict = {}, {}, {}
    for relation in sample_dict:
        prompts, labels, source_ids = [], [], []
        kshots = sample_dict[relation][1][:10]
        kshot_prompt = build_kshot_prompt(kshots, english, k) if use_kshots else []
        for id_pair, pair, _ in zip(sample_dict[relation][0][10:], sample_dict[relation][1][10:], sample_dict[relation][2][10:]):
            prompt = build_prompt(pair, simple, english)
            full_prompt = kshot_prompt + prompt if use_kshots else prompt
            prompts.append(full_prompt)
            labels.append(pair[1][1])
            source_ids.append(id_pair[1][0])
        prompt_dict[relation], label_dict[relation], source_id_dict[relation] = prompts, labels, source_ids
    return prompt_dict, label_dict, source_id_dict

def prompt_set_up_hist(sample_dict, use_kshots, simple, english, k):
    prompt_dict, label_dict, source_id_dict = {}, {}, {}
    for relation in sample_dict:
        prompts, labels, source_ids = [], [], []
        kshots = sample_dict[relation][1][:10]
        kshot_prompt = build_kshot_prompt(kshots, english, k) if use_kshots else []
        for i, (id_pair, pair, _) in enumerate(zip(sample_dict[relation][0][10:], sample_dict[relation][1][10:], sample_dict[relation][2][10:])):
            prompt = build_prompt(pair, simple, english)
            full_prompt = kshot_prompt + prompt if (i == 0 and use_kshots) else prompt
            prompts.append(full_prompt)
            labels.append(pair[1][1])
            source_ids.append(id_pair[1][0])
        prompt_dict[relation], label_dict[relation], source_id_dict[relation] = prompts, labels, source_ids
    return prompt_dict, label_dict, source_id_dict

"""## Feedback"""

def provide_feedback(model_response, english, relation, source_id_dict, syn_map, index, tajp, partial_prompt):
    if tajp == "synta1":
        pseudo_relation = relation + "*"
    else:
        pseudo_relation = relation
    corrects = get_synonyms(syn_map, pseudo_relation, source_id_dict[relation][index])
    matched = any(correct.lower() in model_response.lower() for correct in corrects)
    if matched:
        msg = "Good job! This is one of the answers we were looking for." if english else "Bon travail! Ceci est une des réponses que l'on cherchait."
    else:
        msg = (f"Wrong answer. Examples of correct answers: {str(corrects)}." if english
               else f"Mauvaise réponse. Exemples de bonnes réponses: {str(corrects)}.")
    partial_prompt.append({"role": "user", "content": msg})
    if "mistral" in model_nick_name:
        partial_prompt.append({"role": "assistant", "content": "ok"})
    print(msg)

"""## Model Sessions"""

def model_session_history(prompt_dict, english, source_id_dict, syn_map, feedback_mode, tajp):
    model_response_dict, partial_prompts_dict = {}, {}
    for r, relation in enumerate(prompt_dict):
        model_responses, partial_prompt = [], []
        system_msg = {"role": "system", "content": ("You are an expert on French analogies. Respond with one French lexeme."
                     if english else "Vous êtes un expert des analogies françaises. Répondez toujours en une seule lexie.")}
        partial_prompt.append(system_msg)
        for i, prompt in enumerate(prompt_dict[relation]):
            if i % 5 == 0:
                print(f"{int(i/5)+1} / 20")
            partial_prompt += prompt
            try:
                model_response = get_response(partial_prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise
            assistant_response = {"role": "assistant", "content": model_response}
            partial_prompt.append(assistant_response)
            print(model_response)
            if feedback_mode:
                provide_feedback(model_response, english, relation, source_id_dict, syn_map, i, tajp, partial_prompt)
            model_responses.append(model_response)
        partial_prompts_dict[relation], model_response_dict[relation] = partial_prompt, model_responses
    return model_response_dict, partial_prompts_dict

def model_session_no_hist(prompt_dict, english):
    model_response_dict, partial_prompts_dict = {}, {}
    for r, relation in enumerate(prompt_dict):
        model_responses, chat = [], []
        system_prompt = [{"role": "system", "content": ("You are an expert on French analogies. Respond with one French lexeme."
                       if english else "Vous êtes un expert des analogies françaises. Répondez toujours en une seule lexie.")}]
        for i, prompt in enumerate(prompt_dict[relation]):
            if i % 5 == 0:
                print(f"{int(i/5)+1} / 20")
            actual_prompt = system_prompt + prompt
            model_response = get_response(actual_prompt)
            print(model_response)
            model_responses.append(model_response)
            chat.append(actual_prompt + [{"role": "assistant", "content": model_response}])
        partial_prompts_dict[relation], model_response_dict[relation] = chat, model_responses
    return model_response_dict, partial_prompts_dict

"""## Main"""

# Load sample data and synonym mapping
with open("indata/regrouped_para_sample.txt") as f:
    regrouped_para_sample = eval(f.read())
with open("indata/regrouped_synta0_sample.txt") as f:
    regrouped_synta0_sample = eval(f.read())
with open("indata/regrouped_synta1_sample.txt") as f:
    regrouped_synta1_sample = eval(f.read())
with open("indata/synonym_mapping2.txt") as f:
    syn_map = eval(f.read())

options = [
    (True,),              # history options
    (True, False),        # simple options
    (True, False),        # english options
    (0, 1, 3, 5, 7, 10),  # k_shots options
    (True, False)         # feedback options
]

i = 0
for combination in product(*options):
    history, simple, english, k, feedback_mode = combination
    prompt_set_up = prompt_set_up_hist if history else prompt_set_up_no_hist
    use_kshots = (k > 0)
    if not history and feedback_mode:
        continue

    extra_dir = f"{'hist' if history else 'no_hist'}/"
    extra_dir += "simple/" if simple else "complex/"
    extra_dir += "English/" if english else "French/"
    extra_dir += f"{k}shot/"
    extra_dir += "feedback" if feedback_mode else "no_feedback"

    base_dir = f"outdata/{model_nick_name}"
    main_dir = f"{base_dir}/{extra_dir}"
    os.makedirs(main_dir, exist_ok=True)

    sample_dict = {"synta1": regrouped_synta1_sample,
                   "para": regrouped_para_sample,
                   "synta0": regrouped_synta0_sample}

    durations = []
    for tajp in sample_dict:
        if os.path.exists(f"{main_dir}/{tajp}_model_response.txt"):
            print(f"Skipping {main_dir}/{tajp}_model_response.txt")
            continue
        start = time.time()
        print(f"Processing configuration: simple={simple}, english={english}, k={k}, type={tajp}")
        prompt_dict, label_dict, source_id_dict = prompt_set_up(sample_dict[tajp], use_kshots, simple, english, k)
        if history:
            model_response_dict, chat_history = model_session_history(prompt_dict, english, source_id_dict, syn_map, feedback_mode, tajp)
        else:
            model_response_dict, chat_history = model_session_no_hist(prompt_dict, english)
        with open(f"{main_dir}/{tajp}_model_response.txt", "w") as f:
            f.write(str(model_response_dict))
        with open(f"{main_dir}/{tajp}_chat_history.txt", "w") as g:
            g.write(str(chat_history))
        durations.append(time.time() - start)
    with open(f"{main_dir}/durations.txt", "w") as f:
        f.write(str(durations))
    i += 1
