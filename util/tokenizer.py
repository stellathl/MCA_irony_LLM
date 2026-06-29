"""
Token Counter with Tokenizers
====================================
Using HuggingFace Hub Tokenizer, count the tokens.

Install dependencies first:
    pip install transformers sentencepiece pandas

Tokenizers are downloaded automatically on first run (~5–50 MB each).
No model weights are downloaded — tokenizer files only.
"""

import pandas as pd
import yaml
from transformers import AutoTokenizer
from util.constants import (MODELS, PROMPT_FILES)

# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# Adjust the system prompt and template to your actual task
# ─────────────────────────────────────────────────────────

def read_yaml(yaml_file_path: str) -> dict:
    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format: {yaml_file_path}")
    return data

def write_yaml(yaml_contents: dict, yaml_file_path: str) -> None:
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_contents, f)

def build_prompt(row, condition, template_file, prompt_type="general"):
    """
    Build prompt based on prompt type.
    - general: needs question, cg_framing
    - rsa: only needs context, speaker, target_utterance, options
    """
    template = read_yaml(f"prompts/{template_file}")[condition]
    task = template["task"]
    
    # ── General prompt parameters ──
    if prompt_type.lower() == "general":
        question_line = template.get("question", "").format(pronoun=row.get("pronoun", ""))
        cg_framing = str(row.get("cg_framing", "")).strip()
        
        return task.format(
            context=row.get("context", ""),
            target_utterance=row.get("target_utterance", ""),
            question=question_line,
            cg_framing=cg_framing,
            speaker=row.get("speaker", ""),
            options=row.get("answering_options", "")
        )
    
    # ── RSA prompt parameters ──
    elif prompt_type.lower() == "rsa":
        return task.format(
            context=row.get("context", ""),
            speaker=row.get("speaker", ""),
            target_utterance=row.get("target_utterance", ""),
            options=row.get("answering_options", "")
        )
    
    # ── Default ──
    else:
        return task.format(
            context=row.get("context", ""),
            speaker=row.get("speaker", ""),
            target_utterance=row.get("target_utterance", ""),
            options=row.get("answering_options", "")
        )
# ─────────────────────────────────────────────────────────
# TOKENIZER LOADING
# ─────────────────────────────────────────────────────────

def load_tokenizers() -> dict:
    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"  Loading {name} ({model_id}) …", flush=True)
        try:
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
    "Llama-3-8B":          131_072,  # was missing — caused ⚠
}

# ─────────────────────────────────────────────────────────
# EXPERIMENT SCALE CONSTANTS
# ─────────────────────────────────────────────────────────
N_STIMULI        = 250
N_PROMPT_TYPES   = 3
N_CONDITIONS     = 3
N_MODELS         = len(MODELS)
FEWSHOT_TOKENS   = 0   # set to estimated tokens per few-shot block if used

TOTAL_RUNS = N_STIMULI * N_PROMPT_TYPES * N_CONDITIONS * N_MODELS


def print_report(label: str, df: pd.DataFrame):
    tok_cols = [c for c in df.columns if c.startswith("tok_")]
    print(f"\n{'═'*75}")
    print(f"  {label}  ({len(df)} prompts)")
    print(f"{'═'*75}")
    print(f"  {'Model':<25} {'Min':>5} {'Med':>5} {'Max':>5} {'Total':>8}  {'% ctx':>6}  {'Status':>6}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*8}  {'-'*6}  {'-'*6}")

    for col in tok_cols:
        model = col.replace("tok_", "")
        mn    = int(df[col].min())
        med   = int(df[col].median())
        mx    = int(df[col].max())
        tot   = int(df[col].sum())
        ctx   = MAX_CONTEXT.get(model)

        if ctx:
            pct  = mx / ctx * 100
            # warn if few-shot would push over 80% of context
            fewshot_pct = (mx + FEWSHOT_TOKENS) / ctx * 100
            safe = "✓" if mx < ctx else "✗ EXCEEDS"
            fewshot_warn = " ⚠ few-shot risky" if fewshot_pct > 80 else ""
        else:
            pct, safe, fewshot_warn = 0, "?", ""

        print(f"  {model:<25} {mn:>5} {med:>5} {mx:>5} {tot:>8,}  "
              f"{pct:>5.2f}%  {safe}{fewshot_warn}")

    # Sub-group breakdown
    sub = next((c for c in ("context_level", "cg_level") if c in df.columns), None)
    if sub:
        rep_col = tok_cols[0]
        print(f"\n  Breakdown by {sub}  [{rep_col.replace('tok_','')}]:")
        for grp, gdf in df.groupby(sub):
            print(f"    {grp:>12}:  "
                  f"median={gdf[rep_col].median():.0f}  "
                  f"total={int(gdf[rep_col].sum()):,}")


def print_hpc_summary(results: dict):
    """Print HPC planning summary across all conditions and prompt types."""
    combined  = pd.concat(results.values(), ignore_index=True)
    tok_cols  = [c for c in combined.columns if c.startswith("tok_")]

    print(f"\n{'═'*75}")
    print(f"  HPC PLANNING SUMMARY")
    print(f"  Scale: {N_STIMULI} stimuli × {N_PROMPT_TYPES} prompt types × "
          f"{N_CONDITIONS} conditions × {N_MODELS} models")
    print(f"  Total inference runs: {TOTAL_RUNS:,}")
    if FEWSHOT_TOKENS:
        print(f"  Few-shot overhead:    +{FEWSHOT_TOKENS} tokens/prompt")
    print(f"{'═'*75}")
    print(f"  {'Model':<25} {'Max tok':>8}  {'Ctx win':>8}  {'% used':>7}  "
          f"{'Total tok (all runs)':>22}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*7}  {'-'*22}")

    for col in tok_cols:
        model  = col.replace("tok_", "")
        mx     = int(combined[col].max())
        ctx    = MAX_CONTEXT.get(model, "?")
        pct    = f"{mx/ctx*100:.2f}%" if isinstance(ctx, int) else "?"
        # project total tokens across full experiment
        median = combined[col].median()
        projected = int(median * TOTAL_RUNS / N_MODELS)  # per model
        print(f"  {model:<25} {mx:>8,}  {ctx:>8,}  {pct:>7}  "
              f"~{projected:>20,}")

    print(f"\n  ⚠  Set FEWSHOT_TOKENS > 0 to check if few-shot fits safely.")
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

    # 2. Condition 1B with general template
    print("\nCounting tokens — Condition 1B …")
    df1 = pd.read_csv(c1b_path)
    df1["_prompt"] = df1.apply(lambda row: build_prompt(row, "condition", PROMPT_FILES.get("general")), axis=1)
    print("First prompt\n", df1["_prompt"].iloc[0])
    df1 = count_tokens_df(df1, "_prompt", tokenizers)
    print_report("Condition 1B – Context Richness", df1)
    results["Condition1B"] = df1

    # 3. Condition 2 with general template
    print("\nCounting tokens — Condition 2 …")
    df2 = pd.read_csv(c2_path)
    df2["_prompt"] = df2.apply(lambda row: build_prompt(row, "condition", PROMPT_FILES.get("general")), axis=1)
    print("First prompt\n", df2["_prompt"].iloc[0])
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

    # 4. HPC summary across all conditions
    print_hpc_summary(results)

    # 5. Save per-row CSV
    out = "token_counts_real.csv"
    drop = ["_prompt"]
    combined.drop(columns=[c for c in drop if c in combined.columns]).to_csv(out, index=False)
    print(f"\n  Saved → {out}")