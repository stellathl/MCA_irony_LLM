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
#from huggingface_hub import login

# =========================================================
# CONFIG
# =========================================================

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# Directories
DATASETS_DIR = "./data"
PROMPTS_DIR = "./prompts"
OUTPUTS_DIR = "./outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Models
MODELS = [
    "google/gemma-3-1b-it"
]

# Generation settings
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.1

# Prompt templates
PROMPT_FILES = {
    "general": "general_prompt.txt",
    "rsa": "rsa_prompt.txt",
    "reasoning": "reasoning_prompt.txt"
}

# =========================================================
# LOAD PROMPT TEMPLATE
# =========================================================

def load_prompt_template(path):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    # optional shuffle
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
# BUILD PROMPT
# =========================================================

def build_prompt(template, text):

    return template.format(text=text)

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(
    pipe,
    tokenizer,
    dataset,
    template,
    model_name,
    prompt_type,
    dataset_category,
    output_path
):

    records = []

    with torch.no_grad():

        for i, record in enumerate(dataset, 1):

            print(f"\n[{prompt_type.upper()}] [{dataset_category}] Record #{i}")

            text = record["text"]

            # Build final prompt
            prompt = build_prompt(template, text)

            try:

                # Apply chat template
                messages = [
                    {"role": "user", "content": prompt}
                ]

                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Generate
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
                    "dataset_category": dataset_category,
                    "prompt_type": prompt_type,
                    "text": text,
                    "label": record.get("label", ""),
                    "raw_output": generated_text
                })

            except Exception as e:

                print(f"ERROR on record #{i}: {e}")

                records.append({
                    "id": record.get("id", i),
                    "model": model_name,
                    "dataset_category": dataset_category,
                    "prompt_type": prompt_type,
                    "text": text,
                    "label": record.get("label", ""),
                    "raw_output": f"ERROR: {e}"
                })

            # cleanup
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("Starting inference pipeline...")

    # Load dataset categories
    dataset_categories = os.listdir(DATASETS_DIR)

    for model_name in MODELS:

        print(f"\nLoading model: {model_name}")

        clean_model_name = model_name.split("/")[-1]

        # Load model
        model, tokenizer = load_model(model_name)

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # =============================================
        # LOOP THROUGH DATASET CATEGORIES
        # =============================================

        for category in dataset_categories:

            category_path = os.path.join(DATASETS_DIR, category)

            # skip non-folders
            if not os.path.isdir(category_path):
                continue

            print(f"\nDataset category: {category}")

            # =============================================
            # LOAD CSV FILES INSIDE CATEGORY
            # =============================================

            csv_files = [
                f for f in os.listdir(category_path)
                if f.endswith(".csv")
            ]

            for csv_file in csv_files:

                dataset_path = os.path.join(category_path, csv_file)

                dataset_name = csv_file.replace(".csv", "")

                print(f"\nLoading dataset: {dataset_name}")

                dataset = load_dataset(dataset_path)

                # =============================================
                # LOOP THROUGH PROMPT STRATEGIES
                # =============================================

                for prompt_type, prompt_file in PROMPT_FILES.items():

                    print(f"\nRunning prompt strategy: {prompt_type}")

                    prompt_path = os.path.join(
                        PROMPTS_DIR,
                        prompt_file
                    )

                    template = load_prompt_template(prompt_path)

                    # output folder
                    output_dir = os.path.join(
                        OUTPUTS_DIR,
                        prompt_type,
                        category
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
                        dataset=dataset,
                        template=template,
                        model_name=clean_model_name,
                        prompt_type=prompt_type,
                        dataset_category=category,
                        output_path=output_file
                    )

        # cleanup
        del model
        del tokenizer
        del pipe

        gc.collect()
        torch.cuda.empty_cache()

    print("\nInference completed successfully.")