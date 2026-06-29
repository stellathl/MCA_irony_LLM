import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from util.shuffle_options import combine_results, letter_to_pos


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

    # Single class check - macro average doesn't work with one class
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


def compute_group_metrics(sub_df):
    """
    Compute metrics for a subset of the data.
    """
    # Empty DataFrame check
    if len(sub_df) == 0:
        return 0.0, 0.0, 0.0, 0.0

    y_true = sub_df["correct_option_pos"].apply(letter_to_pos)
    y_pred = sub_df["chosen_original_option"].apply(letter_to_pos) 

    accuracy = (y_true == y_pred).mean()

    # Single class check
    if len(y_true.unique()) == 1:
        return accuracy, 0.0, 0.0, 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def context_metrics(df):
    """
    Compute metrics for ambiguous vs unambiguous contexts.
    """
    rows = []

    for context, grp in df.groupby("context_level"):
        # Clean missing data
        grp_clean = grp.dropna(subset=["chosen_option", "correct_option_pos"])
        
        acc, p, r, f1 = compute_group_metrics(grp_clean)

        rows.append({
            "context_level": context,
            "n": len(grp_clean),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })

    return pd.DataFrame(rows)


def irony_metrics(df):
    """
    Compute metrics for ironic vs non-ironic examples.
    """
    rows = []

    for irony, grp in df.groupby("irony_label"):
        # Clean missing data
        grp_clean = grp.dropna(subset=["chosen_option", "correct_option_pos"])
        
        acc, p, r, f1 = compute_group_metrics(grp_clean)

        rows.append({
            "irony_label": irony,
            "n": len(grp_clean),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })

    return pd.DataFrame(rows)


def interaction_metrics(df):
    """
    Compute metrics for context_level × irony_label interaction.
    """
    rows = []

    for (context, irony), grp in df.groupby(
        ["context_level", "irony_label"]
    ):
        # Clean missing data
        grp_clean = grp.dropna(subset=["chosen_option", "correct_option_pos"])
        
        acc, p, r, f1 = compute_group_metrics(grp_clean)

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
        combined_results = all_results

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

