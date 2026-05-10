"""
Token Counter with Real Tokenizers
====================================
Run this locally where HuggingFace Hub is reachable.

Install dependencies first:
    pip install transformers sentencepiece pandas

Tokenizers are downloaded automatically on first run (~5–50 MB each).
No model weights are downloaded — tokenizer files only.
"""

import pandas as pd
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────
# MODELS  — HuggingFace tokenizer IDs
# ─────────────────────────────────────────────────────────

MODELS = {
    "Gemma-3-4B":          "google/gemma-3-4b-it",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "OLMo-2-7B":           "allenai/OLMo-2-1124-7B-Instruct",
    "Qwen3-8B":            "Qwen/Qwen3-8B",
    "ModernBERT-8B":       "answerdotai/ModernBERT-large",
    "Llama-3-8B":          "meta-llama/Llama-3.1-8B"
}

# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# Adjust the system prompt and template to your actual task
# ─────────────────────────────────────────────────────────

# YAML


def read_yaml(yaml_file_path: str) -> dict:
    with open(yaml_file_path, "r") as f:
        yaml_contents = yaml.safe_load(f)
    if yaml_contents is None:
        return {}
    return yaml_contents


def write_yaml(yaml_contents: dict, yaml_file_path: str) -> None:
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_contents, f)


# SYSTEM_GENERAL_PROMPT = (
#     "You are a linguistic analysis assistant. "
#     "Determine whether the target utterance is ironic or non-ironic "
#     "given the context. Answer with 'ironic' or 'non-ironic', "
#     "then briefly explain."
# )

# GENERAL_PROMPT_TEMPLATE = (
#     "Task: You will read short stories that describe everyday situations\
# \ and which finish with a character saying something. Your task is to decide,\
# \ given the situation, what meaning the character is most likely conveying.\
# \ Each story will be followed by 4 possible meaning interpretations listed from\
# \ 1 to 4. Read each story and choose the number corresponding to the most likely\
# \ meaning. You can only answer with 1, 2, 3, or 4. \n\n{scenario} {question}\n\
# \n{options}\n\nAnswer:"
# "question:"" What meaning is X likely conveying?"
# )
def build_prompt_c1b(row: pd.Series) -> str:
    return (
        # f"{GENERAL_PROMPT_TEMPLATE}\n\n"
        f"Context: {row['context']}\n\n"
        f"Target utterance: \"{row['target_utterance']}\"\n\n"
        "Is the target utterance ironic or non-ironic?"
    )

def build_prompt_c2(row: pd.Series) -> str:
    cg = str(row.get("cg_level", "")).strip().lower()
    cg_framing  = str(row.get("cg_framing",         "")).strip()
    explicit    = str(row.get("Explicit_sentence",   "")).strip()

    middle = ""
    if cg == "high" and cg_framing and cg_framing.lower() != "nan":
        middle = f"{cg_framing}"
        if explicit and explicit.lower() != "nan":
            middle += f" {explicit}"
        middle = middle.strip() + "\n\n"

    return (
        # f"{GENERAL_PROMPT_TEMPLATE}\n\n"
        f"Context: {row['context']}\n\n"
        f"{middle}"
        f"Target utterance: \"{row['target_utterance']}\"\n\n"
        "Is the target utterance ironic or non-ironic?"
    )

# ─────────────────────────────────────────────────────────
# TOKENIZER LOADING
# ─────────────────────────────────────────────────────────

def load_tokenizers() -> dict:
    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"  Loading {name} ({model_id}) …", flush=True)
        try:
            # tokenizer_only=True avoids downloading model weights
            tok = AutoTokenizer.from_pretrained(model_id)
            tokenizers[name] = tok
            vocab_size = tok.vocab_size
            print(f"    ✓  vocab_size={vocab_size:,}")
        except Exception as e:
            print(f"    ✗  Failed: {e}")
    return tokenizers

# ─────────────────────────────────────────────────────────
# TOKEN COUNTING
# ─────────────────────────────────────────────────────────

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens for a plain text string."""
    return len(tokenizer.encode(text, add_special_tokens=True))

def count_tokens_df(df: pd.DataFrame, prompt_col: str,
                    tokenizers: dict) -> pd.DataFrame:
    """Add one token-count column per model to df."""
    for name, tok in tokenizers.items():
        col = f"tok_{name}"
        df[col] = df[prompt_col].apply(lambda p: count_tokens(p, tok))
        print(f"  {name:<25}  done  (median={df[col].median():.0f})")
    return df

# ─────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────

MAX_CONTEXT = {
    "Gemma-3-4B":          131_072,
    "Mistral-7B-Instruct":  32_768,
    "OLMo-2-7B":           131_072,
    "Qwen3-8B":            131_072,
    "ModernBERT-8B":         8_192,
}

def print_report(label: str, df: pd.DataFrame):
    tok_cols = [c for c in df.columns if c.startswith("tok_")]
    print(f"\n{'═'*65}")
    print(f"  {label}  ({len(df)} prompts)")
    print(f"{'═'*65}")
    print(f"  {'Model':<25} {'Min':>5} {'Med':>5} {'Max':>5} {'Total':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
    for col in tok_cols:
        model = col.replace("tok_", "")
        mn  = int(df[col].min())
        med = int(df[col].median())
        mx  = int(df[col].max())
        tot = int(df[col].sum())
        ctx = MAX_CONTEXT.get(model, "?")
        safe = "✓" if isinstance(ctx, int) and mx < ctx else "⚠"
        print(f"  {model:<25} {mn:>5} {med:>5} {mx:>5} {tot:>8,}  {safe}")

    # Sub-group breakdown
    sub = next((c for c in ("context_level", "cg_level") if c in df.columns), None)
    if sub:
        rep_col = tok_cols[0]  # use first model as representative
        print(f"\n  Breakdown by {sub}  [{rep_col.replace('tok_','')}]:")
        for grp, gdf in df.groupby(sub):
            print(f"    {grp:>12}:  "
                  f"median={gdf[rep_col].median():.0f}  "
                  f"total={int(gdf[rep_col].sum()):,}")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    c1b_path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/Condition1B_context_richness_stimuli.csv"
    c2_path  = sys.argv[2] if len(sys.argv) > 2 else \
        "data/Condition2_common_ground_stimuli.csv"

    # 1. Load tokenizers
    print("Loading tokenizers …")
    tokenizers = load_tokenizers()
    if not tokenizers:
        sys.exit("No tokenizers loaded — check your internet connection and HF access.")

    results = {}

    # 2. Condition 1B
    print("\nCounting tokens — Condition 1B …")
    df1 = pd.read_csv(c1b_path)
    df1["_prompt"] = df1.apply(build_prompt_c1b, axis=1)
    df1 = count_tokens_df(df1, "_prompt", tokenizers)
    print_report("Condition 1B – Context Richness", df1)
    results["Condition1B"] = df1

    # 3. Condition 2
    print("\nCounting tokens — Condition 2 …")
    df2 = pd.read_csv(c2_path)
    df2["_prompt"] = df2.apply(build_prompt_c2, axis=1)
    df2 = count_tokens_df(df2, "_prompt", tokenizers)
    print_report("Condition 2 – Common Ground", df2)
    results["Condition2"] = df2

    # 4. Grand total
    combined = pd.concat(results.values(), ignore_index=True)
    tok_cols  = [c for c in combined.columns if c.startswith("tok_")]
    print(f"\n{'═'*65}")
    print("  GRAND TOTAL (both conditions)")
    print(f"{'═'*65}")
    for col in tok_cols:
        model = col.replace("tok_", "")
        print(f"  {model:<25}  total={int(combined[col].sum()):>8,} tokens")

    # 5. Save per-row CSV
    out = "token_counts_real.csv"
    drop = ["_prompt"]
    combined.drop(columns=[c for c in drop if c in combined.columns]).to_csv(out, index=False)
    print(f"\n  Saved → {out}")