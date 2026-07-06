"""
visualize_from_txt.py

Parses the plain-text metrics reports produced by `metrics.py`
(the `*_metrics.txt` files — Irony table, Context x Irony table,
Original Option Distribution table, Selection Distribution lines)
directly, WITHOUT needing the CSV files, and generates PNG charts.

Works with a single .txt file or a whole directory tree containing
many *_metrics.txt files (one per model / dataset / prompt_type) --
in the latter case all runs are combined into comparison charts.

Usage:
    python visualize_from_txt.py path/to/one_metrics.txt
    python visualize_from_txt.py outputs/metrics                 # scans recursively for *_metrics.txt
    python visualize_from_txt.py outputs/metrics --plots-dir outputs/metrics/plots

Requires: pandas, matplotlib, seaborn
"""

import os
import re
import glob
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


# =========================================================
# PARSING
# =========================================================

META_RE = {
    "model": re.compile(r"^Model\s*:\s*(.+)$"),
    "dataset": re.compile(r"^Dataset\s*:\s*(.+)$"),
    "prompt_type": re.compile(r"^Prompt Type\s*:\s*(.+)$"),
}

# irony_label   n  accuracy  precision  recall    f1
IRONY_ROW_RE = re.compile(
    r"^\s*(?P<irony_label>\S+)\s+(?P<n>\d+)\s+(?P<accuracy>[\d.]+)\s+"
    r"(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+(?P<f1>[\d.]+)\s*$"
)

# context_level irony_label  n  accuracy  precision  recall    f1
INTERACTION_ROW_RE = re.compile(
    r"^\s*(?P<context_level>\S+)\s+(?P<irony_label>\S+)\s+(?P<n>\d+)\s+"
    r"(?P<accuracy>[\d.]+)\s+(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+(?P<f1>[\d.]+)\s*$"
)

# a  167  41  126  24.6%  58.0%
OPTION_ROW_RE = re.compile(
    r"^\s*(?P<option>[a-dA-D])\s+(?P<count>\d+)\s+(?P<correct>\d+)\s+"
    r"(?P<incorrect>\d+)\s+(?P<accuracy_pct>[\d.]+)%\s+(?P<selection_pct>[\d.]+)%\s*$"
)


def parse_metrics_txt(path: str) -> dict:
    """
    Parse one *_metrics.txt file into a dict with keys:
    meta, irony_df, interaction_df, option_df
    Any section not found is left as None.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    meta = {"model": None, "dataset": None, "prompt_type": None, "source_file": path}
    irony_rows, interaction_rows, option_rows = [], [], []

    section = None  # None | "irony" | "interaction" | "option"

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # --- metadata ---
        for key, pattern in META_RE.items():
            m = pattern.match(stripped)
            if m:
                meta[key] = m.group(1).strip()

        # --- section switches ---
        if stripped.startswith("--- Irony"):
            section = "irony"
            continue
        if stripped.startswith("--- Context"):
            section = "interaction"
            continue
        if "ORIGINAL OPTION DISTRIBUTION" in stripped:
            section = "option_pending_header"  # header row comes next non-blank line
            continue
        if stripped.startswith("Selection Distribution"):
            section = None
            continue

        if not stripped:
            continue

        # --- irony rows ---
        if section == "irony":
            m = IRONY_ROW_RE.match(line)
            if m:
                irony_rows.append(m.groupdict())
            continue

        # --- interaction rows ---
        if section == "interaction":
            m = INTERACTION_ROW_RE.match(line)
            if m:
                interaction_rows.append(m.groupdict())
            continue

        # --- option distribution ---
        if section == "option_pending_header":
            # this line is the header ("Original Option  Selection Count ..."); skip it
            if stripped.startswith("Original Option"):
                section = "option"
            continue

        if section == "option":
            m = OPTION_ROW_RE.match(line)
            if m:
                option_rows.append(m.groupdict())
            else:
                # first non-matching line means the table ended
                section = None
            continue

    irony_df = pd.DataFrame(irony_rows) if irony_rows else None
    interaction_df = pd.DataFrame(interaction_rows) if interaction_rows else None
    option_df = pd.DataFrame(option_rows) if option_rows else None

    for df, numeric_cols in [
        (irony_df, ["n", "accuracy", "precision", "recall", "f1"]),
        (interaction_df, ["n", "accuracy", "precision", "recall", "f1"]),
        (option_df, ["count", "correct", "incorrect", "accuracy_pct", "selection_pct"]),
    ]:
        if df is not None:
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

    return {
        "meta": meta,
        "irony_df": irony_df,
        "interaction_df": interaction_df,
        "option_df": option_df,
    }


def find_txt_files(path: str):
    if os.path.isfile(path):
        return [path]
    pattern = os.path.join(path, "**", "*_metrics.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # fall back to any .txt in case the naming convention differs
        pattern = os.path.join(path, "**", "*.txt")
        files = sorted(glob.glob(pattern, recursive=True))
    return files


def tag_and_collect(parsed_list):
    """Attach model/dataset/prompt_type + a run_label to every sub-dataframe and concat."""
    irony_frames, interaction_frames, option_frames = [], [], []

    for parsed in parsed_list:
        meta = parsed["meta"]
        run_label = f"{meta['model']} | {meta['prompt_type']}"
        tag = {
            "model": meta["model"],
            "dataset": meta["dataset"],
            "prompt_type": meta["prompt_type"],
            "run_label": run_label,
        }

        if parsed["irony_df"] is not None:
            irony_frames.append(parsed["irony_df"].assign(**tag))
        if parsed["interaction_df"] is not None:
            interaction_frames.append(parsed["interaction_df"].assign(**tag))
        if parsed["option_df"] is not None:
            option_frames.append(parsed["option_df"].assign(**tag))

    irony_all = pd.concat(irony_frames, ignore_index=True) if irony_frames else None
    interaction_all = pd.concat(interaction_frames, ignore_index=True) if interaction_frames else None
    option_all = pd.concat(option_frames, ignore_index=True) if option_frames else None

    return irony_all, interaction_all, option_all


# =========================================================
# PLOTS
# =========================================================

def plot_irony_accuracy(irony_df: pd.DataFrame, plots_dir: str):
    plt.figure(figsize=(max(9, 1.6 * irony_df["run_label"].nunique()), 6.5))
    ax = sns.barplot(
        data=irony_df, x="run_label", y="accuracy", hue="irony_label",
        palette="rocket"
    )
    ax.set_title("Accuracy by Irony Label")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Irony Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "irony_accuracy.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def plot_irony_full_metrics(irony_df: pd.DataFrame, plots_dir: str):
    """Extra chart: precision/recall/f1 too, not just accuracy, faceted by irony_label."""
    melted = irony_df.melt(
        id_vars=["run_label", "irony_label"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric", value_name="score"
    )
    g = sns.catplot(
        data=melted, x="run_label", y="score", hue="metric",
        col="irony_label", kind="bar", palette="viridis",
        height=5.5, aspect=1.3
    )
    g.set_xticklabels(rotation=20, ha="right")
    g.set(ylim=(0, 1))
    g.fig.suptitle("Full Metrics by Irony Label", y=1.05)

    out_path = os.path.join(plots_dir, "irony_full_metrics.png")
    g.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


def plot_interaction_heatmaps(interaction_df: pd.DataFrame, plots_dir: str):
    runs = sorted(interaction_df["run_label"].unique())
    n_runs = len(runs)
    n_cols = min(3, n_runs)
    n_rows = -(-n_runs // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for idx, run_label in enumerate(runs):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = interaction_df[interaction_df["run_label"] == run_label]
        pivot = sub.pivot(index="context_level", columns="irony_label", values="accuracy")
        sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="YlGnBu", ax=ax, cbar=idx == 0)
        ax.set_title(run_label, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for idx in range(n_runs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle("Accuracy: Context Level x Irony Label", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "context_irony_heatmaps.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_option_distribution(option_df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(
        data=option_df, x="run_label", y="selection_pct", hue="option",
        palette="Set2", ax=axes[0]
    )
    axes[0].set_title("Selection Rate by Original Option")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Selection %")
    axes[0].tick_params(axis="x", rotation=20)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")

    sns.barplot(
        data=option_df, x="run_label", y="accuracy_pct", hue="option",
        palette="Set2", ax=axes[1]
    )
    axes[1].set_title("Accuracy by Original Option")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Accuracy %")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=20)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    plt.tight_layout()

    out_path = os.path.join(plots_dir, "option_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize metrics directly from *_metrics.txt reports (no CSVs needed)."
    )
    parser.add_argument(
        "path",
        help="A single *_metrics.txt file, or a directory to scan recursively for *_metrics.txt files."
    )
    parser.add_argument(
        "--plots-dir", default=None,
        help="Where to save the PNG charts (default: same directory as 'path', or './plots' for a single file)"
    )
    args = parser.parse_args()

    files = find_txt_files(args.path)
    if not files:
        print(f"No *_metrics.txt files found at/under: {args.path}")
        return

    print(f"Found {len(files)} report(s):")
    for f in files:
        print(f"  - {f}")

    plots_dir = args.plots_dir or (
        os.path.join(args.path, "plots") if os.path.isdir(args.path)
        else os.path.join(os.path.dirname(os.path.abspath(args.path)), "plots")
    )
    os.makedirs(plots_dir, exist_ok=True)

    parsed_list = [parse_metrics_txt(f) for f in files]
    irony_all, interaction_all, option_all = tag_and_collect(parsed_list)

    if irony_all is not None:
        plot_irony_accuracy(irony_all, plots_dir)
        plot_irony_full_metrics(irony_all, plots_dir)
    else:
        print("[skip] No Irony table found in any file.")

    if interaction_all is not None:
        plot_interaction_heatmaps(interaction_all, plots_dir)
    else:
        print("[skip] No Context x Irony table found in any file.")

    if option_all is not None:
        plot_option_distribution(option_all, plots_dir)
    else:
        print("[skip] No Original Option Distribution table found in any file.")

    print(f"\nAll available charts saved to: {plots_dir}")


if __name__ == "__main__":
    main()