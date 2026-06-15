import pandas as pd
import re

from util.constants import MODELS

# ── Helper: parse "number — reasoning" from model response ──
def parse_response(response_text):
    """
    Extracts the chosen option number and reasoning from model output.
    Handles formats like:
        "2 — Because..."
        "2. Because..."
        "Option 2, because..."
        "2\nBecause..."
    Returns (chosen_number, reasoning_text)
    """
    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None

    text = str(response_text).strip()

    # Extract the first number that appears (1–4)
    match = re.search(r"\b([1-4])\b", text)
    if not match:
        return None, text  # couldn't parse a number

    chosen = int(match.group(1))

    # Everything after the number is the reasoning
    reasoning = text[match.end():].strip()
    # Clean leading punctuation/separator (—, -, ., :, etc.)
    reasoning = re.sub(r"^[\s\-—.,;:]+", "", reasoning).strip()

    return chosen, reasoning

# ── Score accuracy for one model ────────────────────────────
def score_model(stimuli_csv, responses_csv, model_name):
    """
    stimuli_csv  : the CSV you sent to the model (has correct_option_pos)
    responses_csv: CSV with model responses in a 'response' column,
                   matched to stimuli by Item_ID
    """
    stimuli   = pd.read_csv(stimuli_csv)
    responses = pd.read_csv(responses_csv)

    # Merge on Item_ID so order doesn't matter
    merged = stimuli.merge(responses[["Item_ID", "response"]], on="Item_ID", how="left")

    # Parse responses
    parsed = merged["response"].apply(parse_response)
    merged["chosen_option"] = [p[0] for p in parsed]
    merged["reasoning"]     = [p[1] for p in parsed]

    # Score: chosen == correct_option_pos
    merged["correct"] = merged["chosen_option"] == merged["correct_option_pos"]

    # ── Overall accuracy ──────────────────────────────────
    total    = len(merged)
    n_correct = merged["correct"].sum()
    n_missing = merged["chosen_option"].isna().sum()

    print(f"\n{'='*50}")
    print(f"Model : {model_name}")
    print(f"{'='*50}")
    print(f"Total items     : {total}")
    print(f"Parsed responses: {total - n_missing}")
    print(f"Unparseable     : {n_missing}")
    print(f"Correct         : {n_correct}")
    print(f"Accuracy        : {n_correct / (total - n_missing) * 100:.1f}%")

    # ── Accuracy by condition ─────────────────────────────
    print(f"\n--- Accuracy by irony_label ---")
    print(merged.groupby("irony_label")["correct"]
          .agg(["sum","count"])
          .assign(accuracy=lambda x: x["sum"]/x["count"]*100)
          .rename(columns={"sum":"correct","count":"total"})
          .round(1).to_string())

    print(f"\n--- Accuracy by context_level ---")
    print(merged.groupby("context_level")["correct"]
          .agg(["sum","count"])
          .assign(accuracy=lambda x: x["sum"]/x["count"]*100)
          .rename(columns={"sum":"correct","count":"total"})
          .round(1).to_string())

    print(f"\n--- Accuracy by condition (context × irony) ---")
    print(merged.groupby(["context_level","irony_label"])["correct"]
          .agg(["sum","count"])
          .assign(accuracy=lambda x: x["sum"]/x["count"]*100)
          .rename(columns={"sum":"correct","count":"total"})
          .round(1).to_string())

    # ── Save scored output ────────────────────────────────
    out_cols = [
        "Item_ID", "base_item", "context_level", "irony_label",
        "target_utterance", "answering_options",
        "correct_option_pos", "chosen_option", "correct", "reasoning",
        "presentation_order", "model", "seed"
    ]
    out = merged[[c for c in out_cols if c in merged.columns]]
    fname = f"scored_{model_name}.csv"
    out.to_csv(fname, index=False)
    print(f"\n✓ Saved → {fname}")

    return merged
