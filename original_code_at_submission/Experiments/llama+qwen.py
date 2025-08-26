import json
import os
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
from dotenv import load_dotenv

import time

# Load environment variables from .env file
load_dotenv()

# Safely retrieve the Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

"""## Model Set-Up"""

model_nick_name = "qwen"
model_names = {
    "llama3.0_chat": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1_chat": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-14B-Instruct"
}

model_name = model_names[model_nick_name]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

print("Model loaded successfully")

# Function for generating responses from the model
def get_response(messages, model_name):
    if model_name == "qwen":
        return get_response_qwen(messages)
    else:
        return get_response_llama(messages)

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
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def build_prompt(pair, simple, english):
    if english:
        if simple:
            return [
                {"role": "user", "content": f"Solve the following analogy. {pair[0][0]}: {pair[0][1]}:: {pair[1][0]}: "}
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
                {"role": "user", "content": f"Résolvez l'analogie suivante. {pair[0][0]}: {pair[0][1]}:: {pair[1][0]}: "}
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

# Set up k-shot prompts
def build_kshot_prompt(kshots, english, k):
    content = ["Consider the following examples"] if english else ["Considère les exemples suivants"]

    for i in range(k):
        shot_a, shot_b = kshots[i]
        content.append(f"{shot_a[0]} : {shot_a[1]} :: {shot_b[0]} : {shot_b[1]}")

    content.append("Now, use this pattern to solve the following analogy:" if english else "Maintenant, utilise ce patron pour résoudre l'analogie suivante:")

    return "\n".join(content)

# Setup for running the model and obtaining responses
def model_session_no_hist(prompt_dict, english):
    model_response_dict = {}
    partial_prompts_dict = {}

    for r, relation in enumerate(prompt_dict):
        model_responses = []
        chat = [{"role": "system", "content": "You are an expert on analogies. Always respond concisely."}]

        for i, prompt in enumerate(prompt_dict[relation]):
            if i % 5 == 0:
                print(f"{int(i / 5) + 1} / 20")

            actual_prompt = chat + prompt
            model_response = get_response(actual_prompt, model_nick_name)
            print(model_response)

            model_responses.append(model_response)
            chat.append(actual_prompt + [{"role": "assistant", "content": model_response}])

        partial_prompts_dict[relation] = chat
        model_response_dict[relation] = model_responses
        print(f"Processed relation {relation}")
    
    return model_response_dict, partial_prompts_dict

def main():
    # Load the data (assuming it's been preprocessed in a safe and reusable way)
    data_paths = {
        "regrouped_para_sample": "indata/regrouped_para_sample.txt",
        "regrouped_synta0_sample": "indata/regrouped_synta0_sample.txt",
        "regrouped_synta1_sample": "indata/regrouped_synta1_sample.txt",
        "synonym_mapping": "indata/synonym_mapping2.txt"
    }

    sample_data = {}
    for key, path in data_paths.items():
        with open(path) as f:
            sample_data[key] = eval(f.read())

    # Iterate over configurations
    options = [
        (True, False),  # history options
        (True, False),  # simple_options
        (False, True),  # english_options
        (0, 3, 5, 7, 10),         # k_shots_options
        (True, False)   # feedback_options
    ]

    for combination in product(*options):
        history, simple, english, k, feedback_ness = combination
        use_kshots = k > 0

        if not history and feedback_ness:
            continue

        print(f"Processing configuration: history={history}, simple={simple}, language={english}, k_shots={k}, feedback={feedback_ness}")

        prompt_dict, _, _ = prompt_set_up_no_hist(sample_data["regrouped_synta1_sample"], use_kshots, simple, english, k)
        model_response_dict, chat_history = model_session_no_hist(prompt_dict, english)

        output_dir = f"outdata/{model_nick_name}/{'hist' if history else 'no_hist'}/"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}model_responses.json", "w") as f:
            json.dump(model_response_dict, f, ensure_ascii=False, indent=4)

        with open(f"{output_dir}partial_prompts.json", "w") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
