import yaml
import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# =========================================================
# CONFIG
# =========================================================

DATASETS_DIR = "./data"
PROMPTS_DIR = "./prompts"
OUTPUTS_DIR = "./outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODELS = [
    "google/gemma-3-1b-it"
]

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.1

PROMPT_FILES = {
    "general": "general_prompt.yaml",
    #"rsa": "rsa_prompt.yaml",
    #"reasoning": "reasoning_prompt.yaml"
}

# =========================================================
# LOAD PROMPT
# =========================================================

def load_prompt_template(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
        print("\n--- RAW YAML ---\n", raw)

        data = yaml.safe_load(raw)
        print("\n--- PARSED YAML ---\n", data)

    if isinstance(data, dict) and "template" in data:
        return data["template"]

    if isinstance(data, str):
        return data

    raise ValueError(f"Invalid YAML format: {path}")

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df)

# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# =========================================================
# PROMPT BUILD
# =========================================================

def build_prompt(template, text):
    return template.format(text=text)

# =========================================================
# GENERATION
# =========================================================

def generate_predictions(
    pipe,
    tokenizer,
    dataset,
    template,
    model_name,
    prompt_type,
    dataset_name,
    output_path
):

    records = []

    with torch.no_grad():

        for i, record in enumerate(dataset, 1):

            print(f"\n[{prompt_type.upper()}] Record #{i}")

            text = record["text"]

            prompt = build_prompt(template, text)

            try:
                messages = [{"role": "user", "content": prompt}]

                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                result = pipe(
                    formatted_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True
                )

                generated_text = result[0]["generated_text"][len(formatted_prompt):]

                print(f"\nINPUT:\n{text}")
                print(f"\nOUTPUT:\n{generated_text}")

                records.append({
                    "id": record.get("id", i),
                    "model": model_name,
                    "dataset": dataset_name,
                    "prompt_type": prompt_type,
                    "text": text,
                    "label": record.get("label", ""),
                    "output": generated_text
                })

            except Exception as e:
                print(f"ERROR: {e}")

                records.append({
                    "id": record.get("id", i),
                    "model": model_name,
                    "dataset": dataset_name,
                    "prompt_type": prompt_type,
                    "text": text,
                    "label": record.get("label", ""),
                    "output": f"ERROR: {e}"
                })

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("Starting inference pipeline...")

    # 🔥 FLAT CSV STRUCTURE
    csv_files = [
        f for f in os.listdir(DATASETS_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print("No CSV files found in ./data")
        exit()

    for model_name in MODELS:

        print(f"\nLoading model: {model_name}")

        clean_model_name = model_name.split("/")[-1]

        model, tokenizer = load_model(model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        for csv_file in csv_files:

            dataset_path = os.path.join(DATASETS_DIR, csv_file)
            dataset_name = csv_file.replace(".csv", "")

            print(f"\nLoading dataset: {dataset_name}")

            dataset = load_dataset(dataset_path)

            for prompt_type, prompt_file in PROMPT_FILES.items():

                print(f"\nRunning prompt: {prompt_type}")

                prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
                template = load_prompt_template(prompt_path)

                output_dir = os.path.join(OUTPUTS_DIR, prompt_type)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(
                    output_dir,
                    f"{clean_model_name}_{dataset_name}.csv"
                )

                generate_predictions(
                    pipe=pipe,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    template=template,
                    model_name=clean_model_name,
                    prompt_type=prompt_type,
                    dataset_name=dataset_name,
                    output_path=output_file
                )

        del model
        del tokenizer
        del pipe

        gc.collect()
        torch.cuda.empty_cache()

    print("\nInference completed successfully.")