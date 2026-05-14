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
from util.tokenizer import build_prompt
from util.constants import (MODELS, PROMPT_FILES)

# =========================================================
# CONFIG
# =========================================================

DATASETS_DIR = "./data"
PROMPTS_DIR = "./prompts"
OUTPUTS_DIR = "./outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.1

CONDITION_MAP = {
    "Condition1B_context_richness_stimuli": "condition_1",
    "Condition2_common_ground_stimuli": "condition_2",
}

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
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
# GENERATION
# =========================================================

def generate_predictions(
    pipe,
    tokenizer,
    dataset,
    model_name,
    prompt_type,
    dataset_name,
    output_path
):

    records = []

    with torch.no_grad():

        for i, record in enumerate(dataset, 1):

            print(f"\n[{prompt_type.upper()}] Record #{i}")

            prompt = record["prompt"]

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

                print(f"\nINPUT:\n{prompt}")  
                print(f"\nOUTPUT:\n{generated_text}")

                records.append({
                    "id": record.get("id", i),
                    "model": model_name,
                    "dataset": dataset_name,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
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
                    "prompt": prompt,
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

for model_key, model_name in MODELS.items():  
    print(f"\nLoading model: {model_key} ({model_name})")
    clean_model_name = model_key  
    model, tokenizer = load_model(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    for csv_file in csv_files:

        dataset_path = os.path.join(DATASETS_DIR, csv_file)
        dataset_name = csv_file.replace(".csv", "")

        print(f"\nLoading dataset: {dataset_name}")

        dataset = load_dataset(dataset_path)

        for prompt_type, prompt_file in PROMPT_FILES.items():

            print(f"\nRunning prompt: {prompt_type}")

            dataset["prompt"] = dataset.apply(
                lambda row: build_prompt(row, CONDITION_MAP[dataset_name], prompt_file), axis=1
            )
            records_list = dataset.to_dict(orient="records") 

            csv_files = [
                f for f in os.listdir(DATASETS_DIR)
                if f.endswith(".csv")
            ]

            output_dir = os.path.join(
                OUTPUTS_DIR,
                prompt_type,
            )
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(
                output_dir,
                f"{clean_model_name}_{dataset_name}.csv"
            )

            # Run inference
            generate_predictions(
                pipe=pipe,
                tokenizer=tokenizer,
                dataset=records_list,
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