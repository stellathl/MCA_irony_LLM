import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


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

    y_true = df[gold_col].astype(int)
    y_pred = df[pred_col].astype(int)

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

    y_true = sub_df["correct_option_pos"].astype(int)
    y_pred = sub_df["chosen_option"].astype(int)

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