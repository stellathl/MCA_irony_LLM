import glob
import os
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from shuffle_options import combine_results, letter_to_pos

def compute_classification_metrics(
    df,
    pred_col="chosen_option",
    gold_col="correct_option_pos",
    average="macro"
):
    """
    Compute overall classification metrics.
    """
    df = df.copy()
    df = df.dropna(subset=[pred_col, gold_col])

    if len(df) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    y_true = df[gold_col].apply(letter_to_pos)
    y_pred = df[pred_col].apply(letter_to_pos)

    accuracy = (y_true == y_pred).mean()

    unique_classes = y_true.unique()
    if len(unique_classes) == 1:
        precision = recall = f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

    overall = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return overall

def compute_group_metrics(sub_df, pred_col="chosen_original_option", gold_col="correct_option_pos"):
    """
    Compute metrics for a subset of the data.
    """
    if len(sub_df) == 0:
        return 0.0, 0.0, 0.0, 0.0

    y_true = sub_df[gold_col].apply(letter_to_pos)
    y_pred = sub_df[pred_col].apply(letter_to_pos)

    accuracy = (y_true == y_pred).mean()

    if len(y_true.unique()) == 1:
        return accuracy, 0.0, 0.0, 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def context_metrics(df, pred_col="chosen_original_option", gold_col="correct_option_pos"):
    """
    Compute metrics for ambiguous vs unambiguous contexts.
    """
    rows = []
    for context, grp in df.groupby("context_level"):
        grp_clean = grp.dropna(subset=[pred_col, gold_col])
        acc, p, r, f1 = compute_group_metrics(grp_clean, pred_col, gold_col)
        rows.append({
            "context_level": context,
            "n": len(grp_clean),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })
    return pd.DataFrame(rows)


def irony_metrics(df, pred_col="chosen_original_option", gold_col="correct_option_pos"):
    """
    Compute metrics for ironic vs non-ironic examples.
    """
    rows = []
    for irony, grp in df.groupby("irony_label"):
        grp_clean = grp.dropna(subset=[pred_col, gold_col])
        acc, p, r, f1 = compute_group_metrics(grp_clean, pred_col, gold_col)
        rows.append({
            "irony_label": irony,
            "n": len(grp_clean),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })
    return pd.DataFrame(rows)


def interaction_metrics(df, pred_col="chosen_original_option", gold_col="correct_option_pos"):
    """
    Compute metrics for context_level × irony_label interaction.
    """
    rows = []
    for (context, irony), grp in df.groupby(["context_level", "irony_label"]):
        grp_clean = grp.dropna(subset=[pred_col, gold_col])
        acc, p, r, f1 = compute_group_metrics(grp_clean, pred_col, gold_col)
        rows.append({
            "context_level": context,
            "irony_label": irony,
            "n": len(grp_clean),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })
    return pd.DataFrame(rows)

    # =========================================================
    # METRICS (OLD METRICS.PY - NO PARAMETERS)
    # =========================================================
def save_metrics(all_results, metrics_path: str, model_name: str, dataset_name:str, prompt_type: str):
    if all_results is None:
        return
    if isinstance(all_results, pd.DataFrame):
        if all_results.empty:
            return
        combined_results = all_results  # already combined, skip combine_results()
    else:
        if len(all_results) == 0:
            return
        combined_results = combine_results(all_results)
        
    try:
        print("all_results after all run per model", all_results)

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
            for orig_opt in ["a", "b", "c", "d"]:
                mask = combined_results["chosen_original_option"] == orig_opt
                count = mask.sum()
                
                if count > 0:
                    correct = (combined_results[mask]["chosen_original_option"] == combined_results[mask]["correct_option_pos"]).sum()
                    incorrect = count - correct
                    accuracy = correct / count
                    
                    option_stats.append({
                        "Original Option": orig_opt,
                        "Selection Count": count,
                        "Correct": correct,
                        "Incorrect": incorrect,
                        "Accuracy %": f"{accuracy*100:.1f}%",
                        "Selection %": f"{(count/len(combined_results))*100:.1f}%"
                    })
            
            if option_stats:
                stats_df = pd.DataFrame(option_stats)
                f.write(stats_df.to_string(index=False))
                f.write("\n\n")
            
            # Seçim oranları (pie chart verisi)
            f.write("Selection Distribution:\n")
            selection_counts = combined_results["chosen_original_option"].value_counts().sort_index()
            for opt, count in selection_counts.items():
                pct = (count / len(combined_results)) * 100
                f.write(f"  Option {opt}: {count:3d} selections ({pct:5.1f}%)\n")

        print(f"\n✓ Metrics saved to: {metrics_path}")

    except Exception as e:
        print(f"ERROR computing metrics: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
# OPTION A: rebuild from per-run CSVs
#   outputs/{prompt_type}/{model_key}/{model_key}_{dataset_name}_run{N}.csv
# =========================================================

def load_from_run_csvs(outputs_dir, model_key,dataset_name, prompt_type):
    pattern = os.path.join(
        outputs_dir, prompt_type, model_key,
        f"{model_key}_{dataset_name}_run*.csv"
    )
    run_files = sorted(glob.glob(pattern))

    if not run_files:
        print(f"No run CSVs found matching: {pattern}")
        return None

    print(f"Found {len(run_files)} run CSV(s):")
    for f in run_files:
        print(f"  - {f}")

    dfs = [pd.read_csv(f) for f in run_files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined


# =========================================================
# OPTION B: load from the already-combined per-model CSV
#   outputs/{model_key}_results.csv
#   (filter down to just this dataset + prompt_type, since
#    that file currently mixes all prompt types together)
# =========================================================

def load_from_combined_csv(outputs_dir, model_key,dataset_name, prompt_type):
    path = os.path.join(outputs_dir, f"{model_key}_results.csv")

    if not os.path.exists(path):
        print(f"Combined results file not found: {path}")
        return None

    df = pd.read_csv(path)
    print(f"Loaded combined file: {path} ({df.shape[0]} rows)")

    filtered = df[
        (df["dataset"] == dataset_name) &
        (df["prompt_type"] == prompt_type)
    ].copy()

    print(f"Filtered to dataset={dataset_name}, prompt_type={prompt_type}: {filtered.shape[0]} rows")
    return filtered
def save_rsa_metrics(df, metrics_path, model_key, dataset_name, prompt_type="rsa"):
    """
    Compute and save per-stage RSA metrics (L0, S1, L1, Final), each broken
    down by irony, context × irony, and option distribution — same format
    as save_metrics(), but repeated once per RSA stage.
    """
    if df is None or df.empty:
        print("No data to compute RSA metrics on.")
        return

    stage_cols = {
        "L0 (Literal Listener)":     "rsa_l0",
        "S1 (Informative Speaker)":  "rsa_s1",
        "L1 (Pragmatic Listener)":   "rsa_l1",
        "Final Answer":              "rsa_final",
    }

    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as f:
            f.write(f"Model      : {model_key}\n")
            f.write(f"Dataset    : {dataset_name}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"{'='*50}\n\n")

            for stage_label, pred_col in stage_cols.items():
                if pred_col not in df.columns:
                    f.write(f"### {stage_label} — column '{pred_col}' not found, skipping\n\n")
                    continue

                stage_df = df.dropna(subset=[pred_col, "correct_option_pos"])

                f.write(f"{'#'*60}\n")
                f.write(f"### STAGE: {stage_label}\n")
                f.write(f"{'#'*60}\n\n")

                if stage_df.empty:
                    f.write("No valid predictions for this stage.\n\n")
                    continue

                irony_df = irony_metrics(stage_df, pred_col=pred_col)
                interaction_df = interaction_metrics(stage_df, pred_col=pred_col)

                f.write("--- Irony ---\n")
                f.write(irony_df.to_string(index=False))
                f.write("\n\n")

                f.write("--- Context × Irony ---\n")
                f.write(interaction_df.to_string(index=False))
                f.write("\n\n")

                # Option distribution for this stage
                f.write("--- Option Distribution ---\n")
                option_stats = []
                for opt in ["a", "b", "c", "d"]:
                    mask = stage_df[pred_col] == opt
                    count = mask.sum()
                    if count > 0:
                        correct = (stage_df[mask][pred_col] == stage_df[mask]["correct_option_pos"]).sum()
                        incorrect = count - correct
                        accuracy = correct / count
                        option_stats.append({
                            "Option": opt,
                            "Selection Count": count,
                            "Correct": correct,
                            "Incorrect": incorrect,
                            "Accuracy %": f"{accuracy*100:.1f}%",
                            "Selection %": f"{(count/len(stage_df))*100:.1f}%"
                        })

                if option_stats:
                    stats_df = pd.DataFrame(option_stats)
                    f.write(stats_df.to_string(index=False))
                    f.write("\n\n")

                f.write("Selection Distribution:\n")
                selection_counts = stage_df[pred_col].value_counts().sort_index()
                for opt, count in selection_counts.items():
                    pct = (count / len(stage_df)) * 100
                    f.write(f"  Option {opt}: {count:3d} selections ({pct:5.1f}%)\n")
                f.write("\n\n")

        print(f"\n✓ RSA stage metrics saved to: {metrics_path}")

    except Exception as e:
        print(f"ERROR computing RSA metrics: {e}")
        import traceback
        traceback.print_exc()

# =========================================================
# STANDALONE ENTRY POINT
#   Recomputes metrics from already-saved run CSVs, without
#   touching the model, tokenizer, or generate_predictions.
#   Run with:  python3 -m util.metrics   (from project root)
# =========================================================
 
if __name__ == "__main__":
 
    OUTPUTS_DIR = "./outputs"
 
    # ── Discover (model_key, dataset_name, prompt_type) combos ──
    # by scanning outputs/{prompt_type}/{model_key}/{model_key}_{dataset_name}_run*.csv
    pattern = os.path.join(OUTPUTS_DIR, "*", "*", "*_run*.csv")
    run_files = glob.glob(pattern)
 
    if not run_files:
        print(f"No run CSVs found under {OUTPUTS_DIR}. Nothing to compute.")
        exit()
 
    combos = set()
    for f in run_files:
        parts = f.split(os.sep)
        print(parts)
        prompt_type = parts[-3]
        model_key   = parts[-2]
        fname       = os.path.basename(f)  # {model_key}_{dataset_name}_run{N}.csv
 
        if not fname.startswith(model_key + "_"):
            print(f"Skipping unrecognized filename pattern: {f}")
            continue
 
        stem = fname[len(model_key) + 1:]          # {dataset_name}_run{N}.csv
        dataset_name = stem.rsplit("_run", 1)[0]    # {dataset_name}
        combos.add((model_key, dataset_name, prompt_type))
 
    print(f"Found {len(combos)} (model, dataset, prompt_type) combination(s):")
    for c in sorted(combos):
        print(f"  - {c}")
 
    for model_key, dataset_name, prompt_type in sorted(combos):
        print(f"\n{'='*60}")
        print(f"Computing metrics: model={model_key} dataset={dataset_name} prompt_type={prompt_type}")
        print(f"{'='*60}")
 
        combined_results = load_from_run_csvs(OUTPUTS_DIR, model_key, dataset_name, prompt_type)
 
        if combined_results is None or combined_results.empty:
            print("Falling back to combined per-model CSV...")
            combined_results = load_from_combined_csv(OUTPUTS_DIR, model_key, dataset_name, prompt_type)
 
        if combined_results is None or combined_results.empty:
            print(f"No data found for {model_key}/{dataset_name}/{prompt_type} — skipping.")
            continue
 
        metrics_path = os.path.join(
            OUTPUTS_DIR, "metrics",
            f"{model_key}_{dataset_name}_{prompt_type}_metrics.txt"
        )
        save_metrics(combined_results, metrics_path, model_key, dataset_name, prompt_type)
 
        if prompt_type.lower() == "rsa":
            combined_rsa_metrics_path = os.path.join(
                OUTPUTS_DIR, "metrics",
                f"{model_key}_{dataset_name}_{prompt_type}_combined_rsa_stage_metrics.txt"
            )
            save_rsa_metrics(combined_results, combined_rsa_metrics_path, model_key, dataset_name, prompt_type)
 
    print(f"\n✓ Done. All metrics written under: {os.path.join(OUTPUTS_DIR, 'metrics')}")


