"""
visualize_from_txt.py

Parses the plain-text metrics reports produced by `metrics.py`
(the `*_metrics.txt` files — Irony table, Context x Irony table,
Option Distribution table, Selection Distribution lines)
directly, WITHOUT needing the CSV files, and generates PNG charts.

Handles two report shapes:
  - save_metrics()     -> a single Irony / Context x Irony / Option table
  - save_rsa_metrics() -> the same three tables repeated once per RSA
                           stage (L0, S1, L1, Final Answer), each behind
                           a "### STAGE: <label>" marker

Only the essential comparison charts are produced (no duplicate
heatmap/bar-chart pairs for the same metric):

  1. overview_rsa_by_model.png                 all models' RSA — every stage kept
                                                 as its own bar, never collapsed.
  2. overview_accuracy_by_model.png            general + general_reasoning + ALL rsa
                                                 stages together in ONE combined chart,
                                                 accuracy only, models grouped side by
                                                 side within each cluster.
  3. overview_all_metrics_by_model.png         same combined grouping, but
                                                 accuracy/precision/recall/f1 side by
                                                 side (secondary chart).
  4. <model>_option_by_prompt_type.png          selection rate: general vs
                                                 general_reasoning, per model.
  5. <model>_rsa_option_by_stage.png            selection rate across RSA
                                                 stages, per model.
  6. overview_option_selection_rate.png         selection rate, every model
                                                 and every prompt strategy
                                                 side by side.

Files whose Irony metrics are entirely zero (a broken/empty run) are
automatically excluded and reported.

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

# =========================================================
# STYLE
# =========================================================

sns.set_theme(style="white", context="notebook")

# Used for per-letter option bars (a/b/c/d) — kept separate from the
# prompt-strategy palette below so the two never look confusable.
PALETTE = ["#3B6E8F", "#4FA88A", "#D99A4E", "#C1584B", "#7C6A9C", "#5A8B8B"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#C9CDD3",
    "axes.linewidth": 0.9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#4B5157",
    "xtick.color": "#5B6169",
    "ytick.color": "#5B6169",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "savefig.facecolor": "white",
})

TITLE_COLOR = "#20242A"
SUBTITLE_COLOR = "#8A9099"


def _style_axis(ax):
    """Strip chartjunk: no box, faint horizontal grid only, no tick marks."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C9CDD3")
    ax.yaxis.grid(True, color="#EAEBED", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False)


def _add_bar_labels(ax, fmt="%.2f", color="#3A3F45", size=8.5):
    for container in ax.containers:
        labels = ax.bar_label(container, fmt=fmt, padding=2, fontsize=size, color=color)
        for lab in labels:
            lab.set_fontweight("normal")


def _style_legend(ax, title, **kwargs):
    leg = ax.legend(
        title=title, bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, handlelength=1.1, handleheight=1.1, **kwargs,
    )
    leg.get_title().set_fontsize(9.5)
    leg.get_title().set_color("#4B5157")
    return leg


def _panel_title(ax, title):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="medium", color=TITLE_COLOR, pad=10)


def _page_title(fig, title, subtitle=None, x=0.01, y=1.02):
    fig.text(x, y + 0.02, title, fontsize=17, fontweight="medium", color=TITLE_COLOR, ha="left", va="bottom")
    if subtitle:
        fig.text(x, y - 0.005, subtitle, fontsize=11, color=SUBTITLE_COLOR, ha="left", va="bottom")


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()


# =========================================================
# PARSING
# =========================================================

META_RE = {
    "model": re.compile(r"^Model\s*:\s*(.+)$"),
    "dataset": re.compile(r"^Dataset\s*:\s*(.+)$"),
    "prompt_type": re.compile(r"^Prompt Type\s*:\s*(.+)$"),
}

# "### STAGE: L0 (Literal Listener)" — marks an RSA stage block (save_rsa_metrics output).
# Files without any of these lines are treated as single-stage (classic save_metrics output).
STAGE_RE = re.compile(r"^###\s*STAGE:\s*(.+?)\s*$")

# Canonical stage order for RSA reports; anything else found gets appended alphabetically.
STAGE_ORDER = [
    "L0 (Literal Listener)",
    "S1 (Informative Speaker)",
    "L1 (Pragmatic Listener)",
    "Final Answer",
]

# Short axis labels for the canonical stages (used on chart x-axes).
STAGE_SHORT = {
    "L0 (Literal Listener)": "L0",
    "S1 (Informative Speaker)": "S1",
    "L1 (Pragmatic Listener)": "L1",
    "Final Answer": "Final",
}

# Category order/colors used by the "general / general_reasoning / rsa"
# comparison charts (one bar-group per model). Deliberately a DIFFERENT
# palette from PALETTE (which colors option letters a/b/c/d) so the two
# never read as the same color meaning two different things.
BASE_CATEGORY_ORDER = ["general", "general_reasoning"]
RSA_CATEGORY_ORDER = [f"rsa-{STAGE_SHORT[s]}" for s in STAGE_ORDER]
CATEGORY_PALETTE = ["#1B4965", "#5FA777", "#EE9B00", "#BB3E03", "#6A4C93", "#3A5A78"]
CATEGORY_COLORS = dict(zip(
    BASE_CATEGORY_ORDER + RSA_CATEGORY_ORDER,
    CATEGORY_PALETTE,
))

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

    Every row is tagged with a "stage" column: the stage label for RSA
    reports (save_rsa_metrics), or None for classic single-stage reports
    (save_metrics). Any section not found is left as None.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    meta = {"model": None, "dataset": None, "prompt_type": None, "source_file": path}
    irony_rows, interaction_rows, option_rows = [], [], []

    section = None  # None | "irony" | "interaction" | "option_pending_header" | "option"
    current_stage = None  # None, or e.g. "L0 (Literal Listener)"

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # --- metadata ---
        for key, pattern in META_RE.items():
            m = pattern.match(stripped)
            if m:
                meta[key] = m.group(1).strip()

        # --- RSA stage marker ---
        m_stage = STAGE_RE.match(stripped)
        if m_stage:
            current_stage = m_stage.group(1).strip()
            continue

        # --- section switches ---
        if stripped.startswith("--- Irony"):
            section = "irony"
            continue
        if stripped.startswith("--- Context"):
            section = "interaction"
            continue
        if "OPTION DISTRIBUTION" in stripped.upper():
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
                irony_rows.append({**m.groupdict(), "stage": current_stage})
            continue

        # --- interaction rows ---
        if section == "interaction":
            m = INTERACTION_ROW_RE.match(line)
            if m:
                interaction_rows.append({**m.groupdict(), "stage": current_stage})
            continue

        # --- option distribution ---
        if section == "option_pending_header":
            # Skip separator/header lines (their wording and count vary
            # between report types: "====" dividers, "Original Option ..."
            # vs "Option ..." headers). Only switch to the data-reading
            # state once a real data row is found.
            m = OPTION_ROW_RE.match(line)
            if m:
                option_rows.append({**m.groupdict(), "stage": current_stage})
                section = "option"
            continue

        if section == "option":
            m = OPTION_ROW_RE.match(line)
            if m:
                option_rows.append({**m.groupdict(), "stage": current_stage})
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


def file_is_all_zero(parsed: dict) -> bool:
    """
    True if every numeric metric in the file is zero — a broken/empty run
    (e.g. the model produced no valid outputs). Checks Irony metrics first
    (most reports have this table); falls back to Option selection counts
    if Irony is absent.
    """
    irony_df = parsed.get("irony_df")
    if irony_df is not None and not irony_df.empty:
        cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in irony_df.columns]
        if cols:
            return bool((irony_df[cols] == 0).all().all())

    option_df = parsed.get("option_df")
    if option_df is not None and not option_df.empty and "count" in option_df.columns:
        return bool((option_df["count"] == 0).all())

    # No tables at all parsed from the file — treat as unusable/zero.
    return irony_df is None and option_df is None


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


def stage_order_present(df: pd.DataFrame):
    """Ordered list (L0 -> S1 -> L1 -> Final) of the stages actually present in df."""
    present = df["stage"].dropna().unique().tolist()
    ordered = [s for s in STAGE_ORDER if s in present]
    extra = sorted(s for s in present if s not in STAGE_ORDER)
    return ordered + extra


def stage_short(stage: str) -> str:
    return STAGE_SHORT.get(stage, stage)


def assign_category(row) -> str:
    """
    Map a row to a comparison "category":
      - classic (no RSA stage)  -> its prompt_type as-is (e.g. "general", "general_reasoning")
      - RSA row (has a stage)   -> "rsa-<short stage>" (e.g. "rsa-L0", "rsa-Final")
    RSA always keeps ALL its stages as separate categories — never collapsed
    down to a single representative value.
    """
    if pd.isna(row.get("stage")):
        return row.get("prompt_type") or "unknown"
    return f"rsa-{stage_short(row['stage'])}"


def category_order_present(categories):
    """Known categories first in a fixed order, anything unrecognised appended after."""
    known = BASE_CATEGORY_ORDER + RSA_CATEGORY_ORDER
    present = set(categories)
    ordered = [c for c in known if c in present]
    extra = sorted(c for c in present if c not in known)
    return ordered + extra


def category_palette(categories_ordered):
    fallback_cycle = CATEGORY_PALETTE
    colors = []
    fi = 0
    for c in categories_ordered:
        if c in CATEGORY_COLORS:
            colors.append(CATEGORY_COLORS[c])
        else:
            colors.append(fallback_cycle[fi % len(fallback_cycle)])
            fi += 1
    return colors


def model_palette(models_ordered):
    """
    Distinct color per model, stable across charts (same model always gets
    the same color). Used when models are the hue (grouped bars within each
    general / general_reasoning / rsa-stage cluster), so e.g. gemma-1b and
    gemma-3b always sit right next to each other in the same color scheme.
    """
    base = sns.color_palette("tab10", n_colors=max(10, len(models_ordered)))
    return [base[i % len(base)] for i in range(len(models_ordered))]


# =========================================================
# PLOTS — per-model option selection rate
# (general vs general_reasoning, and RSA stages)
# =========================================================

def plot_option_by_prompt_type(option_df_model: pd.DataFrame, model: str, plots_dir: str):
    """
    One figure per model: selection rate, x-axis is the classic prompt type
    (general -> general_reasoning), bars grouped by option.
    """
    prompt_types = option_df_model["prompt_type"].dropna().unique().tolist()
    order = [p for p in BASE_CATEGORY_ORDER if p in prompt_types] + \
            sorted(p for p in prompt_types if p not in BASE_CATEGORY_ORDER)
    n_options = option_df_model["option"].nunique()

    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(order)), 5.5))

    sns.barplot(
        errorbar=None,
        data=option_df_model, x="prompt_type", y="selection_pct", hue="option",
        order=order, palette=PALETTE[:n_options], ax=ax, width=0.6,
        edgecolor="white", linewidth=1, zorder=3,
    )
    _panel_title(ax, "Selection Rate")
    ax.set_xlabel("")
    ax.set_ylabel("Selection %")
    ax.set_ylim(0, max(100, option_df_model["selection_pct"].max() * 1.15))
    _style_axis(ax)
    _add_bar_labels(ax, fmt="%.1f")
    _style_legend(ax, "Option")

    _page_title(fig, "Option Distribution — General vs. General+Reasoning", subtitle=model, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    out_path = os.path.join(plots_dir, f"{_slug(model)}_option_by_prompt_type.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_option_by_stage(option_df_run: pd.DataFrame, run_label: str, plots_dir: str):
    """
    One figure per run: selection rate, x-axis is the RSA stage
    (L0 -> S1 -> L1 -> Final), bars grouped by option.
    """
    stages = stage_order_present(option_df_run)
    stage_labels = [stage_short(s) for s in stages]
    n_options = option_df_run["option"].nunique()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    sns.barplot(
        errorbar=None,
        data=option_df_run, x="stage", y="selection_pct", hue="option",
        order=stages, palette=PALETTE[:n_options], ax=ax, width=0.6,
        edgecolor="white", linewidth=1, zorder=3,
    )
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels)
    _panel_title(ax, "Selection Rate")
    ax.set_xlabel("")
    ax.set_ylabel("Selection %")
    ax.set_ylim(0, max(100, option_df_run["selection_pct"].max() * 1.15))
    _style_axis(ax)
    _add_bar_labels(ax, fmt="%.1f")
    _style_legend(ax, "Option")

    _page_title(fig, "Option Distribution by RSA Stage", subtitle=run_label, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    out_path = os.path.join(plots_dir, f"{_slug(run_label)}_option_by_stage.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


# =========================================================
# PLOTS — COMBINED OVERVIEW: every model side by side, with
# general / general_reasoning / ALL rsa stages together in ONE panel
# (never split into separate general/general_reasoning/rsa charts)
# =========================================================

def build_overview_irony_df(irony_all: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the irony table down to one row per (model, category, irony_label),
    where category is "general" / "general_reasoning" / "rsa-L0" / "rsa-S1" /
    "rsa-L1" / "rsa-Final" (etc). Multiple source files landing on the same
    (model, category, irony_label) are combined with an n-weighted average.
    """
    df = irony_all.copy()
    df["category"] = df.apply(assign_category, axis=1)

    rows = []
    for (model, category, irony_label), g in df.groupby(["model", "category", "irony_label"]):
        n_sum = g["n"].sum()
        rows.append({
            "model": model,
            "category": category,
            "irony_label": irony_label,
            "n": n_sum,
            "accuracy": (g["accuracy"] * g["n"]).sum() / n_sum,
            "precision": (g["precision"] * g["n"]).sum() / n_sum,
            "recall": (g["recall"] * g["n"]).sum() / n_sum,
            "f1": (g["f1"] * g["n"]).sum() / n_sum,
        })
    return pd.DataFrame(rows)


def collapse_overview_overall(overview: pd.DataFrame) -> pd.DataFrame:
    """
    Further collapse an (model, category, irony_label) overview down to one
    row per (model, category) — a single n-weighted overall value, so every
    model's RSA sits together with its general / general_reasoning results
    in ONE panel, instead of being split by irony_label.
    """
    rows = []
    for (model, category), g in overview.groupby(["model", "category"]):
        n_sum = g["n"].sum()
        rows.append({
            "model": model,
            "category": category,
            "irony_label": "overall",
            "n": n_sum,
            "accuracy": (g["accuracy"] * g["n"]).sum() / n_sum,
            "precision": (g["precision"] * g["n"]).sum() / n_sum,
            "recall": (g["recall"] * g["n"]).sum() / n_sum,
            "f1": (g["f1"] * g["n"]).sum() / n_sum,
        })
    return pd.DataFrame(rows)


def _render_overview_chart(
    overview: pd.DataFrame, plots_dir: str, categories: list, out_filename: str,
    page_title: str, page_subtitle: str, metrics=None, facet_by_irony_label: bool = True,
    group_by: str = "category",
):
    """
    Shared renderer: N-column figure (one column per metric in `metrics`).

    `group_by` controls which variable becomes the x-axis cluster and which
    becomes the hue (the grouped bars sitting right next to each other
    inside each cluster):
      - "model"    -> x=model, hue=category. Every model is its own cluster,
                      general/general_reasoning/rsa-stages sit next to each
                      other within it.
      - "category" -> x=category, hue=model. general / general_reasoning /
                      rsa-L0 / rsa-S1 / ... are each their own cluster, and
                      every model's bar sits right next to the others inside
                      that cluster (e.g. gemma-1b right next to gemma-3b).

    `categories` is always the full, ordered list of general/general_reasoning/
    rsa-stage categories present — passed together so nothing gets split into
    separate charts.
    """
    if overview.empty:
        return

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    models = sorted(overview["model"].dropna().unique())
    irony_labels = sorted(overview["irony_label"].dropna().unique())

    if group_by == "model":
        x_col, x_order = "model", models
        hue_col, hue_order = "category", categories
        palette = category_palette(categories)
        legend_title = "Prompt Strategy / RSA Stage"
    else:
        x_col, x_order = "category", categories
        hue_col, hue_order = "model", models
        palette = model_palette(models)
        legend_title = "Model"

    n_label_rows = len(irony_labels) if facet_by_irony_label and len(irony_labels) > 1 else 1
    n_cols = len(metrics)

    # Scale width by how many bars actually need to fit side by side
    # (x_order x hue_order), so a combined general+rsa panel doesn't
    # look cramped.
    bars_per_col = max(1, len(x_order) * len(hue_order))
    col_width = max(5.2, 0.55 * bars_per_col)

    fig, axes = plt.subplots(
        n_label_rows, n_cols,
        figsize=(col_width * n_cols, 4.8 * n_label_rows),
        squeeze=False,
    )

    row_slices = irony_labels if n_label_rows > 1 else [None]

    for r, label in enumerate(row_slices):
        sub_row = overview if label is None else overview[overview["irony_label"] == label]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sns.barplot(
                errorbar=None,
                data=sub_row, x=x_col, y=metric, hue=hue_col,
                order=x_order, hue_order=hue_order, palette=palette,
                ax=ax, width=0.78, edgecolor="white", linewidth=1, zorder=3,
            )
            title = metric.capitalize() if label is None else f"{metric.capitalize()} — {label}"
            _panel_title(ax, title)
            ax.set_xlabel("")
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            _style_axis(ax)
            _add_bar_labels(ax, size=7.5)
            if r == 0 and c == n_cols - 1:
                _style_legend(ax, legend_title)
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    _page_title(fig, page_title, subtitle=page_subtitle, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94 if n_label_rows == 1 else 0.96])

    out_path = os.path.join(plots_dir, out_filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_overview_split_by_category(irony_all: pd.DataFrame, plots_dir: str, facet_by_irony_label: bool = False):
    """
    RSA-only comparison chart, every model side by side, as ONE combined
    panel per metric (not split by irony_label — so all models' RSA
    accuracy sits together):
      - overview_rsa_by_model.png   (all models' RSA — every stage kept
                                      as its own bar, never collapsed)
    """
    overview_by_label = build_overview_irony_df(irony_all)
    if overview_by_label.empty:
        return

    overview = overview_by_label if facet_by_irony_label else collapse_overview_overall(overview_by_label)
    all_categories = category_order_present(overview["category"].unique())

    rsa_cats = [c for c in all_categories if c.startswith("rsa-")]
    if not rsa_cats:
        return
    sub = overview[overview["category"].isin(rsa_cats)]
    _render_overview_chart(
        sub, plots_dir, rsa_cats,
        out_filename="overview_rsa_by_model.png",
        page_title="RSA — Overview",
        page_subtitle="every model side by side, all RSA stages  ·  accuracy, precision, recall, f1",
        facet_by_irony_label=facet_by_irony_label,
        group_by="model",
    )


def plot_overview_combined_by_model(
    irony_all: pd.DataFrame, plots_dir: str, facet_by_irony_label: bool = False,
    group_by: str = "category",
):
    """
    ONE combined comparison: general, general_reasoning, AND every rsa stage
    sitting together in the SAME panel (never split into three separate
    files). Two charts are produced:

      - overview_accuracy_by_model.png       accuracy only (primary chart)
      - overview_all_metrics_by_model.png     accuracy/precision/recall/f1
                                               side by side (secondary chart)

    `group_by="category"` (default) clusters by general/general_reasoning/
    rsa-stage, with every MODEL's bar sitting right next to the others inside
    each cluster — e.g. gemma-1b right next to gemma-3b within the "rsa-L0"
    group, within "general", within "general_reasoning", etc.
    Pass `group_by="model"` to flip it back: cluster by model instead, with
    the categories sitting next to each other inside each model's group.
    """
    overview_by_label = build_overview_irony_df(irony_all)
    if overview_by_label.empty:
        return

    overview = overview_by_label if facet_by_irony_label else collapse_overview_overall(overview_by_label)
    categories = category_order_present(overview["category"].unique())

    # Primary: accuracy-only, everything (general + general_reasoning + rsa) together.
    _render_overview_chart(
        overview, plots_dir, categories,
        out_filename="overview_accuracy_by_model.png",
        page_title="Accuracy — Overview",
        page_subtitle="general + general_reasoning + all rsa stages, models grouped side by side",
        metrics=["accuracy"],
        facet_by_irony_label=facet_by_irony_label,
        group_by=group_by,
    )

    # Secondary: full metric breakdown, same combined grouping.
    _render_overview_chart(
        overview, plots_dir, categories,
        out_filename="overview_all_metrics_by_model.png",
        page_title="Accuracy / Precision / Recall / F1 — Overview",
        page_subtitle="general + general_reasoning + all rsa stages, models grouped side by side",
        metrics=["accuracy", "precision", "recall", "f1"],
        facet_by_irony_label=facet_by_irony_label,
        group_by=group_by,
    )


def build_overview_option_df(option_all: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the option-distribution table down to one row per
    (model, category, option), where category is "general" /
    "general_reasoning" / "rsa-L0" / "rsa-S1" / "rsa-L1" / "rsa-Final" (etc).
    Multiple source files landing on the same (model, category, option) are
    combined with a count-weighted average.
    """
    df = option_all.copy()
    df["category"] = df.apply(assign_category, axis=1)

    rows = []
    for (model, category, option), g in df.groupby(["model", "category", "option"]):
        count_sum = g["count"].sum()
        rows.append({
            "model": model,
            "category": category,
            "option": option,
            "count": count_sum,
            "correct": g["correct"].sum(),
            "incorrect": g["incorrect"].sum(),
            "accuracy_pct": (g["accuracy_pct"] * g["count"]).sum() / count_sum,
            "selection_pct": (g["selection_pct"] * g["count"]).sum() / count_sum,
        })
    return pd.DataFrame(rows)


def plot_overview_option_by_model(option_all: pd.DataFrame, plots_dir: str, facet_by_option: bool = True):
    """
    Combined comparison chart for option distribution: for every model,
    general / general_reasoning / rsa-L0 / rsa-S1 / rsa-L1 / rsa-Final bars
    sit side by side for Selection Rate.
    """
    overview = build_overview_option_df(option_all)
    if overview.empty:
        return

    models = sorted(overview["model"].dropna().unique())
    categories = category_order_present(overview["category"].unique())
    palette = category_palette(categories)
    options = sorted(overview["option"].dropna().unique())

    n_rows = len(options) if facet_by_option and len(options) > 1 else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(9, 1.6 * len(models) * len(categories) / 2), 4.6 * n_rows), squeeze=False)
    row_slices = options if n_rows > 1 else [None]

    for r, option in enumerate(row_slices):
        sub_row = overview if option is None else overview[overview["option"] == option]
        ax = axes[r][0]
        sns.barplot(
            errorbar=None,
            data=sub_row, x="model", y="selection_pct", hue="category",
            order=models, hue_order=categories, palette=palette,
            ax=ax, width=0.72, edgecolor="white", linewidth=1, zorder=3,
        )
        panel_title = "Selection Rate" if option is None else f"Selection Rate — Option {str(option).upper()}"
        _panel_title(ax, panel_title)
        ax.set_xlabel("")
        ax.set_ylabel("Selection %")
        ax.set_ylim(0, max(100, sub_row["selection_pct"].max() * 1.15))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        _style_axis(ax)
        _add_bar_labels(ax, fmt="%.1f", size=7.5)
        if r == 0:
            _style_legend(ax, "Prompt Strategy")
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    _page_title(
        fig, "Option Selection Rate — Overview",
        subtitle="every model, general vs general_reasoning vs rsa, selection rate by option",
        y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94 if n_rows == 1 else 0.96])

    out_path = os.path.join(plots_dir, "overview_option_selection_rate.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
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

    parsed_all = [parse_metrics_txt(f) for f in files]

    # Drop files whose metrics are entirely zero (broken/empty runs).
    parsed_list, zero_files = [], []
    for parsed in parsed_all:
        if file_is_all_zero(parsed):
            zero_files.append(parsed["meta"]["source_file"])
        else:
            parsed_list.append(parsed)
    if zero_files:
        print("\n[excluded] All-zero metrics, not used in any chart:")
        for f in zero_files:
            print(f"  - {f}")

    if not parsed_list:
        print("\nNo usable (non-zero) reports remain — nothing to plot.")
        return

    # Warn about files that collide on (model, prompt_type): their rows will
    # be combined/averaged together in every chart. This is almost always
    # unintentional (e.g. an old/stale report left in the folder alongside
    # a newer one).
    run_label_files = {}
    for parsed in parsed_list:
        meta = parsed["meta"]
        run_label = f"{meta['model']} | {meta['prompt_type']}"
        run_label_files.setdefault(run_label, []).append(meta["source_file"])
    dupes = {k: v for k, v in run_label_files.items() if len(v) > 1}
    if dupes:
        print("\n[warning] Multiple files share the same Model + Prompt Type — their rows will be combined/averaged together:")
        for run_label, run_files in dupes.items():
            print(f"  - {run_label}:")
            for f in run_files:
                print(f"      {f}")
        print("  If any of these are old/stale reports, remove or rename them so only one file per (model, prompt type) remains.")

    irony_all, interaction_all, option_all = tag_and_collect(parsed_list)

    print()

    # ---- Irony metrics: old 3-way split (general / general_reasoning / rsa separately) ----
    # ---- plus the new combined overview (everything together, models grouped side by side) ----
    if irony_all is not None:
        plot_overview_split_by_category(irony_all, plots_dir)
        plot_overview_combined_by_model(irony_all, plots_dir)

    # ---- Option selection rate: per-model + combined overview ----
    if option_all is not None:
        classic = option_all[option_all["stage"].isna()]
        rsa = option_all[option_all["stage"].notna()]

        if not classic.empty:
            for model in sorted(classic["model"].dropna().unique()):
                plot_option_by_prompt_type(classic[classic["model"] == model], model, plots_dir)

        if not rsa.empty:
            for run_label in sorted(rsa["run_label"].unique()):
                plot_option_by_stage(rsa[rsa["run_label"] == run_label], run_label, plots_dir)

        plot_overview_option_by_model(option_all, plots_dir)

    print(f"\nAll charts saved to: {plots_dir}")


if __name__ == "__main__":
    main()