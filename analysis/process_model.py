"""
process_model.py
This turns LLM inference output CSVs into a single row per trial in the same format as human_trials.csv. This is better for statistical analyses.

Usage:
  python process_model.py --input_dir outputs/ --out model_trials.csv

It reads every CSV file in the --input_dir folder (searched recursively) that has the expected columns. Each file covers one model, one prompt type and one run.
"""

import argparse
import ast
import csv
import re
from pathlib import Path
import pandas as pd

CATEGORIES = {"a": "literal", "b": "ironic", "c": "topic_echo", "d": "keyword"}

REQUIRED = {
    "Item_ID", "base_item", "context_level", "irony_label", "model",
    "prompt_type", "run", "original_option_mapping", "chosen_original_option",
}

ITEM_ID = re.compile(r"^C\d+_\d+_(A|UA)_(I|NI)$")


def read_tolerant(path):
    """Read one inference CSV and repair rows broken by the writer.

    Some files contain model reasoning text with unescaped quotation marks, which splits one record across several physical lines and,
    duplicates a block of fields in other rows. Rows whose first field is not a valid Item_ID are fragments and are dropped. Over-long rows keep their first len(header)-1 fields plus the final field, which is always
    chosen_original_option.

    Returns (DataFrame, n_repaired, n_dropped).
    """

    try:
        return pd.read_csv(path), 0, 0
    except pd.errors.ParserError:
        pass

    with open(path, encoding="utf-8", newline="") as fh:
        raw = list(csv.reader(fh))

    header = raw[0]
    width = len(header)
    good, n_repaired, n_dropped = [], 0, 0

    for row in raw[1:]:
        if not row or not ITEM_ID.match(row[0]):
            n_dropped += 1
            continue
        if len(row) == width:
            good.append(row)
        elif len(row) > width:
            good.append(row[:width - 1] + [row[-1]])
            n_repaired += 1
        else:
            n_dropped += 1

    return pd.DataFrame(good, columns=header), n_repaired, n_dropped


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

rows = []
skipped_files = []
missing_counts = {}
inconsistent = {}
repaired_files = {}

for path in sorted(Path(args.input_dir).rglob("*.csv")):
    df, n_repaired, n_dropped = read_tolerant(path)
    if not REQUIRED.issubset(df.columns):
        skipped_files.append(path.name)
        continue
    if n_repaired or n_dropped:
        repaired_files[path.name] = (n_repaired, n_dropped)

    df["chosen_original_option"] = df["chosen_original_option"].replace("", pd.NA)

    key = (df["model"].iloc[0], df["prompt_type"].iloc[0], int(df["run"].iloc[0]))
    missing_counts[key] = int(df["chosen_original_option"].isna().sum())

    #consistency check: does chosen_original_option agree with chosen_option translated through the shuffle mapping?
    if "chosen_option" in df.columns:
        n_bad = 0
        for _, r in df.iterrows():
            if pd.isna(r["chosen_option"]) or pd.isna(r["chosen_original_option"]):
                continue
            pos = str(r["chosen_option"]).strip().lower()[:1]
            if pos not in "abcd":
                continue
            mapping = ast.literal_eval(r["original_option_mapping"])
            if mapping["abcd".index(pos)] != str(r["chosen_original_option"]).strip().lower()[:1]:
                n_bad += 1
        if n_bad:
            inconsistent[key] = n_bad

    for _, r in df.iterrows():
        category_letter = r["chosen_original_option"]
        if pd.isna(category_letter):
            continue  # no parsable answer for this trial
        category_letter = str(category_letter).strip().lower()[:1]
        if category_letter not in CATEGORIES:
            continue

        category = CATEGORIES[category_letter]
        irony = "ironic" if r["irony_label"] == "ironic" else "non_ironic"

        # position the model saw, recovered from the shuffle mapping
        mapping = ast.literal_eval(r["original_option_mapping"])
        shown_position = "abcd"[mapping.index(category_letter)]

        rows.append({
            "agent": r["model"],
            "agent_type": "model",
            "prompt": r["prompt_type"],
            "run": int(r["run"]),
            "item": r["base_item"],
            "irony": irony,
            "richness": r["context_level"],
            "shown_position": shown_position,
            "chosen_category": category,
            "correct": int(category == ("ironic" if irony == "ironic" else "literal")),
        })

n_csv = len(list(Path(args.input_dir).rglob("*.csv")))
print(f"searched {args.input_dir}: found {n_csv} csv file(s), "
      f"{n_csv - len(skipped_files)} with the expected columns")

if skipped_files:
    print(f"\nskipped {len(skipped_files)} file(s) without the expected columns:")
    for name in skipped_files[:10]:
        print(f"  {name}")
    if len(skipped_files) > 10:
        print(f"  ... and {len(skipped_files) - 10} more")

if not rows:
    print("\nNo usable trials found - nothing written.")
    print("Check that --input_dir points at the folder holding the per-model "
          "inference CSVs (the ones named like "
          "Gemma-3-1B_Condition1B_context_richness_stimuli_run1.csv), "
          "not the aggregated metrics spreadsheets.")
    raise SystemExit(1)

trials = pd.DataFrame(rows)
trials.to_csv(args.out, index=False)

print(f"\n{len(trials)} usable trials from {trials['agent'].nunique()} models "
      f"x {trials['prompt'].nunique()} prompt types")
print("\nusable trials per model and prompt type:")
print(trials.groupby(["agent", "prompt"]).size().to_string())
print("\ntrials with no parsable answer (model, prompt, run):")
for key, n in sorted(missing_counts.items()):
    if n:
        print(f"  {key[0]:<22} {key[1]:<18} run {key[2]}: {n}/72")
if inconsistent:
    print("\nWARNING - chosen_option and chosen_original_option disagree:")
    for key, n in sorted(inconsistent.items()):
        print(f"  {key[0]:<22} {key[1]:<18} run {key[2]}: {n} row(s)")
    print("  (chosen_original_option was used; check these rows with your teammate)")

if repaired_files:
    print("\nWARNING - malformed csv repaired (rows fixed / fragments dropped):")
    for name, (n_rep, n_drop) in sorted(repaired_files.items()):
        print(f"  {name}: {n_rep} repaired, {n_drop} dropped")
    print("  (caused by unescaped quotes in the model's reasoning text -")
    print("   worth fixing in the writing code, not just here)")

