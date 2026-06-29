import os
import gc
import random
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from util.parse import parse_response
from util.shuffle_options import build_run_splits, combine_results, format_options, get_correct_option_text, letter_to_pos, parse_options, pos_to_letter, save_combined
from util.tokenizer import build_prompt
from util.constants import (MODELS, PROMPT_FILES, SEEDS)
from util.metrics import (
    save_metrics
)

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
    "Condition1B_context_richness_stimuli": "condition",
}


# =========================================================
# LOAD & SHUFFLE DATASET  (seeded — reproducible)
# =========================================================

def load_and_shuffle_dataset(csv_path, model_key):
    """
    Load the master CSV, assign correct answer texts, then apply a
    seeded shuffle of both item order and answer-option positions.
    Returns a DataFrame ready to feed into the inference loop.
    
    ⭐ 
    : Stores the ORIGINAL position of each option
    """
    seed = SEEDS.get(model_key)
    if seed is None:
        raise ValueError(
            f"No seed defined for model '{model_key}'. "
            f"Add it to MODEL_SEEDS in the CONFIG section."
        )

    df = pd.read_csv(csv_path)
    df["Item_ID"]   = df["Item_ID"].str.strip()
    df["base_item"] = df["Item_ID"].str.extract(r"(C1_\d+)")

    print(f"\n{'='*50}")
    print(f"Dataset : {os.path.basename(csv_path)}")
    print(f"Model   : {model_key}  (seed={seed})")
    print(f"{'='*50}")
    print(f"Total rows       : {len(df)}")
    print(f"Unique base items: {df['base_item'].nunique()}")
    print(f"Conditions per item:\n{df.groupby('base_item').size().value_counts()}")

    # ── Identify correct answer BEFORE shuffling ──────────
    df["correct_option_text"] = df.apply(get_correct_option_text, axis=1)
    missing = df["correct_option_text"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have no correct option text — check irony_label values")

    # ── Shuffle item order (seeded) ───────────────────────
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["presentation_order"] = range(1, len(df) + 1)

    # ── Shuffle answer options per row (seeded) ───────────
    rng = random.Random(seed)
    shuffled_options           = []
    correct_option_pos         = []
    distractor_positions       = []
    original_option_mapping    = []  # ⭐ NEW: Orijinal pozisyon mapping

    for _, row in df.iterrows():
        options      = parse_options(row["answering_options"])
        correct_text = row["correct_option_text"]

        # ⭐ BEFORE shuffling: Store the original index of each option
        # options[0] = option 1, options[1] = option 2, vb.
        original_indices = list(range(len(options)))  # [0, 1, 2, 3] (1-indexed olarak [1, 2, 3, 4])
        
        # Aynı shuffle seed'i kullanarak optionleri ve indeksleri beraber karıştır
        combined = list(zip(options, original_indices))
        rng.shuffle(combined)
        shuffled_options_temp, shuffled_indices = zip(*combined)
        
        options = list(shuffled_options_temp)
        original_map = [pos_to_letter(idx + 1) for idx in shuffled_indices]  # e.g. ['a','c','d','b']

        new_pos = next(
            (pos_to_letter(i + 1) for i, opt in enumerate(options) if opt == correct_text),
            None
        )
        distractor_pos = [
            pos_to_letter(i + 1) for i, opt in enumerate(options) if opt != correct_text
        ]

        shuffled_options.append(format_options(options))
        correct_option_pos.append(new_pos)
        distractor_positions.append(str(distractor_pos))
        original_option_mapping.append(str(original_map))  # ⭐ Store the mapping: [1, 3, 4, 2] vb.

    df["answering_options"]        = shuffled_options
    df["correct_option_pos"]       = correct_option_pos   # ground truth for scoring
    df["distractor_positions"]     = distractor_positions
    df["original_option_mapping"]  = original_option_mapping  # ⭐ NEW 
    df["seed"]                     = seed

    print(f"\nSample after shuffle:")
    print(df[["Item_ID", "irony_label", "correct_option_pos", "original_option_mapping", "presentation_order"]].head(6))

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


def generate_predictions(
    pipe,
    tokenizer,
    dataset,
    model_name,
    prompt_type,
    dataset_name,
    output_path,
    metrics_path 
):
    """
    Generate predictions for dataset and compute metrics.
    
    Returns:
        pd.DataFrame: Results dataframe with predictions and metrics
    """
    records = []

    with torch.no_grad():
        for i, record in enumerate(dataset, 1):

            print(f"\n[{prompt_type.upper()}] Record #{i} / {len(dataset)}")

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

                print(f"INPUT :\n{prompt}")
                print(f"OUTPUT:\n{generated_text}")

            except Exception as e:
                print(f"ERROR on record #{i}: {e}")
                generated_text = f"ERROR: {e}"

            records.append({
                # ── Identifiers ──────────────────────────
                "Item_ID"            : record.get("Item_ID", ""),
                "base_item"          : record.get("base_item", ""),
                "presentation_order" : record.get("presentation_order", i),
                "seed"               : record.get("seed", ""),
                # ── Condition info ───────────────────────
                "context_level"      : record.get("context_level", ""),
                "irony_label"        : record.get("irony_label", ""),
                # ── Ground truth ─────────────────────────
                "correct_option_pos" : record.get("correct_option_pos", ""),
                "correct_option_text": record.get("correct_option_text", ""),
                # ── Run metadata ─────────────────────────
                "model"              : model_name,
                "dataset"            : dataset_name,
                "prompt_type"        : prompt_type,
                "prompt"             : prompt,
                # ── Raw model output ─────────────────────
                "output"             : generated_text,
                "original_option_mapping": record.get("original_option_mapping", ""),
                "run"                : record.get("run", ""),       
                "condition"          : record.get("condition", ""), 
            })

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # ── Build DataFrame ───────────────────────────────────
    df = pd.DataFrame(records)

    # ── Parse responses ───────────────────────────────────
    parsed = df["output"].apply(parse_response)
    df["chosen_option"] = [p[0] for p in parsed]  # Karıştırılmış haldeki selections (1, 2, 3, 4)
    df["reasoning"]     = [p[1] for p in parsed]

    # Convert shuffled selection → ORIGINAL selection
    def convert_to_original_option(row):
        """
        Convert the shuffled selection to the original selection.
        
        Example:
        - original_option_mapping = "[1, 3, 4, 2]"
        - Shuffled position 3 seçildi
        - mapping[3-1] = mapping[2] = 4 (orijinal option 4)
        """
        if pd.isna(row["chosen_option"]):
            return None
        
        try:
            import ast
            mapping = ast.literal_eval(row["original_option_mapping"])
            
            # Index check
            chosen_shuffled = letter_to_pos(row["chosen_option"])
            if 1 <= chosen_shuffled <= len(mapping):
                original_option = mapping[chosen_shuffled - 1]
                return original_option
            else:
                return None
        except Exception:
            return None

    df["chosen_original_option"] = df.apply(convert_to_original_option, axis=1)
    return df




# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("Starting inference pipeline...")

    # ── Get CSV files ──────────────────────────────────────
    csv_files = [
        f for f in os.listdir(DATASETS_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print("No CSV files found in ./data")
        exit()

    print(f"Found datasets: {csv_files}")

    all_results = []

    
    for model_key, model_name in MODELS.items():  
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key} ({model_name})")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model(model_name)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                dtype=torch.bfloat16,
                device_map="auto"
            )

            # ── Dataset loop ─────────────────────────────────
            for csv_file in csv_files:

                dataset_path = os.path.join(DATASETS_DIR, csv_file)
                dataset_name = csv_file.replace(".csv", "")

                if dataset_name not in CONDITION_MAP:
                    print(f"\nSkipping {dataset_name} — not in CONDITION_MAP")
                    continue

                print(f"\nLoading dataset: {dataset_name}")

                try:
                    dataset = load_and_shuffle_dataset(dataset_path, model_key)
                except Exception as e:
                    print(f"ERROR loading dataset {dataset_name}: {e}")
                    continue
                # Generate latin square split data and csv files
                run_splits = build_run_splits(dataset, DATASETS_DIR, dataset_name)

                for run_idx, run_df in enumerate(run_splits, 1):
                    print(f"\n--- Run {run_idx}/4 ({len(run_df)} items) ---")

                    # ── Prompt loop ─────────────────────────────
                    for prompt_type, prompt_file in PROMPT_FILES.items():

                        print(f"\n{'─'*50}")
                        print(f"Running prompt: {prompt_type}")
                        print(f"{'─'*50}")

                        metrics_path = os.path.join(
                            OUTPUTS_DIR,
                            "metrics",
                            f"{model_key}_{dataset_name}_{prompt_type}_metrics.txt"
                        )

                        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

                        # ── Build prompts ──────────────────────────
                        run_df_copy = run_df.copy()
                        run_df_copy["prompt"] = run_df_copy.apply(
                            lambda row: build_prompt(
                                row, 
                                CONDITION_MAP[dataset_name], 
                                prompt_file
                            ), 
                            axis=1
                        )
                        records_list = run_df_copy.to_dict(orient="records") 

                        # ── Output path ────────────────────────────
                        output_path = os.path.join(
                            OUTPUTS_DIR,
                            prompt_type,
                            model_key,
                            f"{model_key}_{dataset_name}_run{run_idx}.csv"
                        )

                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        try:
                            result_df = generate_predictions(
                                pipe=pipe,
                                tokenizer=tokenizer,
                                dataset=records_list,
                                model_name=model_key,
                                prompt_type=prompt_type,
                                dataset_name=dataset_name,
                                output_path=output_path,
                                metrics_path=metrics_path 
                            )

                            if result_df is not None and not result_df.empty:
                                result_df.to_csv(output_path, index=False)
                                print(f"✓ Results saved to: {output_path}")
                                all_results.append(result_df)
                            else:
                                print(f"⚠ WARNING: result_df is empty or None")

                        except Exception as e:
                            print(f"ERROR during inference for {prompt_type}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                

                # ── after ALL datasets/runs/prompts for this model ──
                model_results = [r for r in all_results if r["model"].iloc[0] == model_key]
                model_prompt_results = [r for r in model_results if r["prompt"].iloc[0] == prompt_type]
                if model_prompt_results:
                    combined_model_prompt_results = save_combined(
                        model_prompt_results,
                        output_path=os.path.join(OUTPUTS_DIR, f"{model_key}_results.csv")
                    )       
                    save_metrics(combined_model_prompt_results, metrics_path, model_name, dataset_name, prompt_type)
                    
            print(f"\n{'='*60}")
            print(f"Cleaning up model: {model_key}")
            print(f"{'='*60}")
            del model
            del tokenizer
            del pipe

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR loading model {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue