# =========================================================
# OPTION SHUFFLE HELPERS
# =========================================================

# Run 1: base_item_1→A_I,  base_item_2→A_NI,  base_item_3→UA_I,  base_item_4→UA_NI ...
# Run 2: base_item_1→A_NI, base_item_2→UA_I,  base_item_3→UA_NI, base_item_4→A_I  ...
# Run 3: base_item_1→UA_I, base_item_2→UA_NI, base_item_3→A_I,   base_item_4→A_NI ...
# Run 4: base_item_1→UA_NI,base_item_2→A_I,   base_item_3→A_NI,  base_item_4→UA_I ...

# util/latin_square_split.py

import pandas as pd
import numpy as np

# Fixed Latin Square rotation for 4 conditions
# Rows = runs (0-3), Cols = condition slot index mod 4
# Each condition appears once per column across all runs

LATIN_SQUARE = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
])

def build_run_splits(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.copy()
    df["condition"] = df["Item_ID"].str.extract(r"C1_\d+_(A_I|A_NI|UA_I|UA_NI)$")

    # ── Sanity check: no nulls from regex ────────────────
    null_conditions = df["condition"].isna().sum()
    if null_conditions > 0:
        bad_ids = df[df["condition"].isna()]["Item_ID"].tolist()
        raise ValueError(f"{null_conditions} Item_IDs didn't match condition regex: {bad_ids}")

    conditions = sorted(df["condition"].unique())
    base_items = sorted(df["base_item"].unique())

    print(f"\n{'='*55}")
    print(f"LATIN SQUARE SPLIT — verification")
    print(f"{'='*55}")
    print(f"Conditions found : {conditions}")
    print(f"Base items found : {len(base_items)}")
    print(f"Total rows       : {len(df)}  (expected {len(base_items) * 4})")

    assert len(conditions) == 4, f"Expected 4 conditions, got {conditions}"
    assert len(base_items) == 54, f"Expected 54 base items, got {len(base_items)}"
    assert len(df) == 216, f"Expected 216 rows, got {len(df)}"

    runs = []
    for run_idx in range(4):
        rows = []
        for item_idx, item in enumerate(base_items):
            condition_slot = LATIN_SQUARE[run_idx, item_idx % 4]
            condition = conditions[condition_slot]

            match = df[(df["base_item"] == item) & (df["condition"] == condition)]
            if len(match) != 1:
                raise ValueError(
                    f"Expected 1 row for item={item}, condition={condition}, got {len(match)}"
                )
            rows.append(match.iloc[0])

        run_df = pd.DataFrame(rows).reset_index(drop=True)
        run_df["run"] = run_idx + 1
        runs.append(run_df)

    _verify_coverage(runs, df, conditions, base_items)
    return runs


def _verify_coverage(
    runs: list[pd.DataFrame],
    full_df: pd.DataFrame,
    conditions: list[str],
    base_items: list[str]
):
    """
    Prints a coverage report and raises if anything is missing or duplicated.
    """
    combined = pd.concat(runs, ignore_index=True)

    print(f"\n--- Per-run row counts ---")
    for r in runs:
        run_num = r["run"].iloc[0]
        cond_counts = r["condition"].value_counts().to_dict()
        print(f"  Run {run_num}: {len(r)} rows | conditions: {cond_counts}")

    print(f"\n--- Condition coverage across all runs ---")
    pivot = combined.groupby(["base_item", "condition"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    # Every cell should be exactly 1
    if (pivot != 1).any().any():
        bad = pivot[pivot != 1].stack()
        raise ValueError(f"Coverage error — some item×condition combos appear ≠ 1 time:\n{bad}")

    # Every original row should appear exactly once
    original_ids = set(full_df["Item_ID"].tolist())
    covered_ids  = set(combined["Item_ID"].tolist())
    missing  = original_ids - covered_ids
    extra    = covered_ids - original_ids

    if missing:
        raise ValueError(f"Missing Item_IDs from splits: {missing}")
    if extra:
        raise ValueError(f"Unexpected Item_IDs in splits: {extra}")

    print(f"\n✓ All {len(original_ids)} Item_IDs covered exactly once")
    print(f"✓ Each base item appears in each condition exactly once across 4 runs")
    print(f"✓ Latin Square coverage verified")
    print(f"{'='*55}\n")


def save_combined(all_results: list[pd.DataFrame], output_path: str):
    """
    Concatenates all per-run result DataFrames (across all models,
    prompt types, and runs) into one CSV with a clear column order.
    """
    combined = pd.concat(all_results, ignore_index=True)

    # Put identifier columns first for readability
    front_cols = [
        "model", "prompt_type", "run", "dataset",
        "Item_ID", "base_item", "condition",
        "context_level", "irony_label",
        "presentation_order", "seed",
        "correct_option_pos", "correct_option_text",
        "chosen_option", "correct", "reasoning",
        "output", "prompt",
    ]
    # Only keep cols that actually exist, then append any extras
    existing   = [c for c in front_cols if c in combined.columns]
    extra_cols = [c for c in combined.columns if c not in existing]
    combined   = combined[existing + extra_cols]

    combined.to_csv(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"COMBINED OUTPUT saved → {output_path}")
    print(f"Total rows : {len(combined)}")
    print(f"Columns    : {list(combined.columns)}")
    print(f"\n--- Rows per model × run ---")
    print(combined.groupby(["model", "run"]).size().unstack(fill_value=0).to_string())
    print(f"\n--- Accuracy per model × condition ---")
    if "correct" in combined.columns:
        print(
            combined.groupby(["model", "condition"])["correct"]
            .agg(["sum", "count"])
            .assign(accuracy=lambda x: (x["sum"] / x["count"] * 100).round(1))
            .rename(columns={"sum": "correct", "count": "total"})
            .to_string()
        )
    print(f"{'='*55}\n")

    return combined


def parse_options(text):
    """Split a numbered answering_options string into a plain list."""
    lines = [l.strip() for l in str(text).strip().split("\n") if l.strip()]
    options = []
    for line in lines:
        if line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            options.append(line[2:].strip())
        else:
            options.append(line)
    return options

def format_options(options):
    """Rejoin a list of options into a numbered string."""
    return "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

def get_correct_option_text(row):
    """
    Return the text of the correct answer before any shuffling.
    Option 1 (index 0) = non-ironic interpretation  → correct when irony_label == non-ironic
    Option 2 (index 1) = ironic interpretation       → correct when irony_label == ironic
    Options 3 & 4 are always distractors.
    """
    options = parse_options(row["answering_options"])
    if len(options) < 2:
        return None
    return options[1] if row["irony_label"] == "ironic" else options[0]
