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
from util.parse import parse_response, score_model
from util.shuffle_options import format_options, get_correct_option_text, parse_options
from util.tokenizer import build_prompt
from util.constants import (MODELS, PROMPT_FILES, SEEDS)

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
    shuffled_options     = []
    correct_option_pos   = []
    distractor_positions = []

    for _, row in df.iterrows():
        options      = parse_options(row["answering_options"])
        correct_text = row["correct_option_text"]

        rng.shuffle(options)

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

    df["answering_options"]   = shuffled_options
    df["correct_option_pos"]  = correct_option_pos   # ground truth for scoring
    df["distractor_positions"] = distractor_positions
    df["seed"]                = seed

    print(f"\nSample after shuffle:")
    print(df[["Item_ID", "irony_label", "correct_option_pos", "presentation_order"]].head(6))

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
    output_path
):
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
            })

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # ── Build DataFrame ───────────────────────────────────
    df = pd.DataFrame(records)

    # ── Parse responses ───────────────────────────────────
    parsed = df["output"].apply(parse_response)
    df["chosen_option"] = [p[0] for p in parsed]
    df["reasoning"]     = [p[1] for p in parsed]

    # ── Score accuracy ────────────────────────────────────
    df["correct_option_pos"] = pd.to_numeric(df["correct_option_pos"], errors="coerce")
    df["correct"] = df["chosen_option"] == df["correct_option_pos"]

    # ── Print accuracy summary ────────────────────────────
    total     = len(df)
    n_parsed  = df["chosen_option"].notna().sum()
    n_correct = df["correct"].sum()

    print(f"\n{'='*50}")
    print(f"RESULTS  —  {model_name}  |  {prompt_type}  |  {dataset_name}")
    print(f"{'='*50}")
    print(f"Total items     : {total}")
    print(f"Parsed responses: {n_parsed}  ({total - n_parsed} unparseable)")
    print(f"Correct         : {n_correct}")
    print(f"Accuracy        : {n_correct / n_parsed * 100:.1f}%" if n_parsed else "Accuracy: N/A")

    print(f"\n--- by irony_label ---")
    print(
        df.groupby("irony_label")["correct"]
        .agg(["sum", "count"])
        .assign(accuracy=lambda x: (x["sum"] / x["count"] * 100).round(1))
        .rename(columns={"sum": "correct", "count": "total"})
        .to_string()
    )

    print(f"\n--- by context_level ---")
    print(
        df.groupby("context_level")["correct"]
        .agg(["sum", "count"])
        .assign(accuracy=lambda x: (x["sum"] / x["count"] * 100).round(1))
        .rename(columns={"sum": "correct", "count": "total"})
        .to_string()
    )

    print(f"\n--- by context × irony ---")
    print(
        df.groupby(["context_level", "irony_label"])["correct"]
        .agg(["sum", "count"])
        .assign(accuracy=lambda x: (x["sum"] / x["count"] * 100).round(1))
        .rename(columns={"sum": "correct", "count": "total"})
        .to_string()
    )

    # ── Save scored output ────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

    return df

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

print(f"Found datasets: {csv_files}")

all_results = []

for model_key, model_name in MODELS.items():  
    print(f"\nLoading model: {model_key} ({model_name})")

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

        if dataset_name not in CONDITION_MAP:
            print(f"Skipping {dataset_name} — not in CONDITION_MAP")
            continue

        print(f"\nLoading dataset: {dataset_name}")

        dataset = load_and_shuffle_dataset(dataset_path, model_key)

        for prompt_type, prompt_file in PROMPT_FILES.items():

            print(f"\nRunning prompt: {prompt_type}")

            # Build prompts
            dataset["prompt"] = dataset.apply(
                lambda row: build_prompt(row, CONDITION_MAP[dataset_name], prompt_file), axis=1
            )
            records_list = dataset.to_dict(orient="records") 

            output_path = os.path.join(
                OUTPUTS_DIR,
                prompt_type,
                f"{model_key}_{dataset_name}.csv"
            )

            output_file = os.path.join(
                OUTPUTS_DIR,
                f"{model_key}_{dataset_name}.csv"
            )

            # Run inference
            result_df = generate_predictions(
                pipe=pipe,
                tokenizer=tokenizer,
                dataset=records_list,
                model_name=model_key,
                prompt_type=prompt_type,
                dataset_name=dataset_name,
                output_path=output_path
            )

            all_results.append(result_df)

    del model
    del tokenizer
    del pipe

    gc.collect()
    torch.cuda.empty_cache()

print("\nInference completed successfully.")

# ── Run for all models ────────────────────────────────────

all_results = []
for model in MODELS:
    result = score_model(
        stimuli_csv   = f"stimuli_{model}.csv",
        responses_csv = f"responses_{model}.csv",  # ← your model output file
        model_name    = model
    )
    all_results.append(result)

# ── Cross-model summary ───────────────────────────────────
print(f"\n{'='*50}")
print("CROSS-MODEL SUMMARY")
print(f"{'='*50}")

summary_rows = []
for model, df in zip(MODELS, all_results):
    for condition, grp in df.groupby(["context_level", "irony_label"]):
        acc = grp["correct"].mean() * 100
        summary_rows.append({
            "model"        : model,
            "context_level": condition[0],
            "irony_label"  : condition[1],
            "accuracy"     : round(acc, 1),
            "n"            : len(grp),
        })

summary = pd.DataFrame(summary_rows)
pivot = summary.pivot_table(
    index=["context_level", "irony_label"],
    columns="model",
    values="accuracy"
)
print(pivot.to_string())
summary.to_csv("accuracy_summary.csv", index=False)
print("\n✓ Saved → accuracy_summary.csv")