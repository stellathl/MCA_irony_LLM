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
from util.shuffle_options import build_run_splits, combine_results, format_options, get_correct_option_text, parse_options, save_combined
from util.tokenizer import build_prompt
from util.constants import (MODELS, PROMPT_FILES, SEEDS)
from util.metrics import (
    compute_classification_metrics,
    context_metrics,
    irony_metrics,
    interaction_metrics
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
    "Condition1B_context_richness_stimuli": "condition_1",
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
        original_map = [idx + 1 for idx in shuffled_indices]  # 1-indexed: [1, 2, 3, 4]

        new_pos = next(
            (i + 1 for i, opt in enumerate(options) if opt == correct_text),
            None
        )
        distractor_pos = [
            i + 1 for i, opt in enumerate(options) if opt != correct_text
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
            chosen_shuffled = int(row["chosen_option"])
            
            # Index check
            if 1 <= chosen_shuffled <= len(mapping):
                original_option = mapping[chosen_shuffled - 1]
                return original_option
            else:
                return None
        except:
            return None
    
    df["chosen_original_option"] = df.apply(convert_to_original_option, axis=1)

    # =========================================================
    # METRICS (OLD METRICS.PY - NO PARAMETERS)
    # =========================================================

    
    try:
        print("all_results after all run per model", all_results)
        combined_results = combine_results(all_results)

        overall = compute_classification_metrics(combined_results)
        context_df = context_metrics(combined_results)
        irony_df = irony_metrics(combined_results)
        interaction_df = interaction_metrics(combined_results)


        # =========================================================
        # SAVE METRICS FILE
        # =========================================================
        
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)   
        
        with open(metrics_path, "w") as f:

            f.write(f"Model      : {model_name}\n")
            f.write(f"Dataset    : {dataset_name}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"{'='*50}\n\n")

            f.write("--- Irony ---\n")
            f.write(irony_df.to_string(index=False))
            f.write("\n\n")

            f.write("--- Context × Irony ---\n")
            f.write(interaction_df.to_string(index=False))
            f.write("\n")

            # ⭐ NEW : ORIGINAL OPTION STATISTICS
            f.write("\n" + "="*50 + "\n")
            f.write("--- ORIGINAL OPTION DISTRIBUTION ---\n")
           
            f.write("="*50 + "\n\n")

            # Correct/Incorrect analysis
            option_stats = []
            for orig_opt in [1, 2, 3, 4]:
                mask = df["chosen_original_option"] == orig_opt
                count = mask.sum()
                
                if count > 0:
                    correct = (df[mask]["chosen_original_option"] == df[mask]["correct_option_pos"]).sum()
                    incorrect = count - correct
                    accuracy = correct / count
                    
                    option_stats.append({
                        "Original Option": orig_opt,
                        "Selection Count": count,
                        "Correct": correct,
                        "Incorrect": incorrect,
                        "Accuracy %": f"{accuracy*100:.1f}%",
                        "Selection %": f"{(count/len(df))*100:.1f}%"
                    })
            
            if option_stats:
                stats_df = pd.DataFrame(option_stats)
                f.write(stats_df.to_string(index=False))
                f.write("\n\n")
            
            # Seçim oranları (pie chart verisi)
            f.write("Selection Distribution:\n")
            selection_counts = df["chosen_original_option"].value_counts().sort_index()
            for opt, count in selection_counts.items():
                pct = (count / len(df)) * 100
                f.write(f"  Option {opt}: {count:3d} selections ({pct:5.1f}%)\n")

        print(f"\n✓ Metrics saved to: {metrics_path}")

    except Exception as e:
        print(f"ERROR computing metrics: {e}")
        import traceback
        traceback.print_exc()


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
                        dataset_copy = dataset.copy()
                        dataset_copy["prompt"] = dataset_copy.apply(
                            lambda row: build_prompt(
                                row, 
                                CONDITION_MAP[dataset_name], 
                                prompt_file
                            ), 
                            axis=1
                        )
                        records_list = dataset_copy.to_dict(orient="records") 

                        # ── Output path ────────────────────────────
                        output_path = os.path.join(
                            OUTPUTS_DIR,
                            prompt_type,
                            f"{model_key}_{dataset_name}.csv"
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
                if model_results:
                    save_combined(
                        model_results,
                        output_path=os.path.join(OUTPUTS_DIR, f"{model_key}_results.csv")
                    )       

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

    # =========================================================
    # CROSS-MODEL SUMMARY
    # =========================================================
    
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")

    if all_results:
        summary_rows = []
        combined_results = combine_results(all_results)
        
        for result_df in combined_results:
            if result_df is None or result_df.empty:
                continue
                
            model_name = result_df["model"].iloc[0] if "model" in result_df.columns else "Unknown"
            
            for (context, irony), grp in result_df.groupby(
                ["context_level", "irony_label"], 
                dropna=False
            ):
                # Calculate accuracy
                if "chosen_option" in grp.columns and "correct_option_pos" in grp.columns:
                    acc = (grp["chosen_option"] == grp["correct_option_pos"]).mean() * 100
                else:
                    acc = 0
                    
                summary_rows.append({
                    "model"        : model_name,
                    "context_level": context,
                    "irony_label"  : irony,
                    "accuracy"     : round(acc, 1),
                    "n"            : len(grp),
                })

        if summary_rows:
            summary = pd.DataFrame(summary_rows)
            
            # Print pivot table
            pivot = summary.pivot_table(
                index=["context_level", "irony_label"],
                columns="model",
                values="accuracy",
                aggfunc="first"
            )
            print("\nAccuracy by Context and Irony:")
            print(pivot.to_string())
            
            # Save summary
            summary_csv = os.path.join(OUTPUTS_DIR, "accuracy_summary.csv")
            summary.to_csv(summary_csv, index=False)
            print(f"\n✓ Saved → {summary_csv}")
        else:
            print("No summary rows to display")
    else:
        print("⚠ No results to summarize")

    print(f"\n{'='*60}")
    print("✓ Inference pipeline completed successfully!")
    print(f"{'='*60}")
