import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support
)


def compute_classification_metrics(
    df,
    pred_col="chosen_option",
    gold_col="correct_option_pos",
    average="macro"
):

    df = df.copy()
    df = df.dropna(subset=[pred_col, gold_col])

    y_true = df[gold_col].astype(int)
    y_pred = df[pred_col].astype(int)

    accuracy = (y_true == y_pred).mean()

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
    Metrics for a subset of the data.
    """

    y_true = sub_df["correct_option_pos"].astype(int)
    y_pred = sub_df["chosen_option"].astype(int)

    accuracy = (y_true == y_pred).mean()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def context_metrics(df):
    """
    ambiguous vs unambiguous
    """

    rows = []

    for context, grp in df.groupby("context_level"):

        acc, p, r, f1 = compute_group_metrics(grp)

        rows.append({
            "context_level": context,
            "n": len(grp),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })

    return pd.DataFrame(rows)


def irony_metrics(df):
    """
    ironic vs non-ironic
    """

    rows = []

    for irony, grp in df.groupby("irony_label"):

        acc, p, r, f1 = compute_group_metrics(grp)

        rows.append({
            "irony_label": irony,
            "n": len(grp),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })

    return pd.DataFrame(rows)


def interaction_metrics(df):
    """
    context_level × irony_label
    """

    rows = []

    for (context, irony), grp in df.groupby(
        ["context_level", "irony_label"]
    ):

        acc, p, r, f1 = compute_group_metrics(grp)

        rows.append({
            "context_level": context,
            "irony_label": irony,
            "n": len(grp),
            "accuracy": round(acc, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3)
        })

    return pd.DataFrame(rows)