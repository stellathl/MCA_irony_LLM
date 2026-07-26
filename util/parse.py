import pandas as pd
import re


# ──────────────────────────────────────────────────────────────
# 1) GENERAL PARSER
# ──────────────────────────────────────────────────────────────
def parse_response(response_text, prompt_type):
    """
    Extract the chosen letter (a-d) and reasoning from model output.
    Behaviour varies by prompt_type:
      "general"           – find the first bare a-d letter; no reasoning expected.
      "general_reasoning" – expect "Answer: x" then "Reasoning: ...".
      anything else       – try "Answer: x" first, fall back to bare leading letter.
    Returns (chosen_letter, reasoning_text).
    """
    if pd.isna(response_text) or not str(response_text).strip():
        return None, None

    text = str(response_text).strip()

    if prompt_type == "general":
        m = re.search(r"\b([a-dA-D])\b", text)
        if m:
            return m.group(1).lower(), None
        return None, text

    if prompt_type == "general_reasoning":
        matches = list(re.finditer(r"answer\s*:\s*", text, re.IGNORECASE))
        if matches:
            after = text[matches[-1].end():]
            m = re.search(r"\b([a-dA-D])\b", after)
            if m:
                chosen = m.group(1).lower()
                reasoning = _extract_reasoning_label(text)
                return chosen, reasoning
        # fallback: bare leading letter
        m = re.match(r"\s*([a-dA-D])\b", text)
        if m:
            return m.group(1).lower(), _extract_reasoning_label(text)
        return None, text

    # ── generic fallback ──────────────────────────────────────────
    matches = list(re.finditer(r"answer\s*:\s*", text, re.IGNORECASE))
    if matches:
        after = text[matches[-1].end():]
        m = re.search(r"\b([a-dA-D])\b", after)
        if m:
            chosen = m.group(1).lower()
            trailing = _clean(after[m.end():])
            return chosen, (trailing or None)

    # bare leading letter fallback
    m = re.match(r"\s*([a-dA-D])\b", text)
    if m:
        chosen = m.group(1).lower()
        reasoning = _clean(text[m.end():]) or None
        return chosen, reasoning

    return None, text


# ──────────────────────────────────────────────────────────────
# 2) RSA PARSER
# ──────────────────────────────────────────────────────────────
def parse_response_rsa(response_text):
    """
    RSA (Rational Speech Act) response parser.

    Extracts four separate stages from the model's response, each as its
    own value, so they can later be scored/analyzed independently:

      - l0    : Literal Listener (L0)
      - s1    : Informative Speaker (S1)
      - l1    : Pragmatic Listener (L1)
      - final : Final Answer

    Returns a dict:
        {"l0": x, "s1": x, "l1": x, "final": x, "raw_text": text}
    where each of l0/s1/l1/final is a lowercase letter 'a'-'d', or None
    if that stage wasn't found in the text.
    """
    if pd.isna(response_text) or str(response_text).strip() == "":
        return {"l0": None, "s1": None, "l1": None, "final": None, "raw_text": None}

    text = str(response_text).strip()

    def extract(label_pattern):
        m = re.search(label_pattern + r"\s*:?\s*([a-dA-D])", text, re.IGNORECASE | re.DOTALL)
        return m.group(1).lower() if m else None

    l0 = extract(r"Literal\s+Listener\s*\(L0\)")
    s1 = extract(r"Informative\s+Speaker\s*\(S1\)")
    l1 = extract(r"Pragmatic\s+Listener\s*\(L1\)")
    final = extract(r"Final\s*Answer")

    if l0 is None:
        m = re.search(r"\bL0\b\s*[:=-]?\s*([a-dA-D])", text, re.IGNORECASE)
        if m:
            l0 = m.group(1).lower()
    if s1 is None:
        m = re.search(r"\bS1\b\s*[:=-]?\s*([a-dA-D])", text, re.IGNORECASE)
        if m:
            s1 = m.group(1).lower()
    if l1 is None:
        m = re.search(r"\bL1\b\s*[:=-]?\s*([a-dA-D])", text, re.IGNORECASE)
        if m:
            l1 = m.group(1).lower()

    return {"l0": l0, "s1": s1, "l1": l1, "final": final, "raw_text": text}


# ──────────────────────────────────────────────────────────────
# 3) HELPER: add RSA columns to an existing DataFrame
#    (call this from run_inference.py right before saving a CSV)
# ──────────────────────────────────────────────────────────────
def add_rsa_columns(df, output_col="output"):
    """
    Take a DataFrame with a model-output column and add the four
    RSA-stage columns (rsa_l0, rsa_s1, rsa_l1, rsa_final, rsa_raw_text)
    to it. Returns the same DataFrame with new columns appended.
    Does NOT read or write any files — pure in-memory operation.
    """
    parsed = df[output_col].apply(parse_response_rsa).apply(pd.Series)
    parsed = parsed.rename(columns={
        "l0": "rsa_l0",
        "s1": "rsa_s1",
        "l1": "rsa_l1",
        "final": "rsa_final",
        "raw_text": "rsa_raw_text",
    })
    return pd.concat([df, parsed], axis=1)


# ──────────────────────────────────────────────────────────────
# 4) SMALL HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def _extract_reasoning_label(text):
    """Return text after the last 'Reasoning:' label, or None."""
    m = re.search(r"reasoning\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip() or None
    return None


def _clean(s):
    return re.sub(r"^[\s\-—.,;:]+", "", s.strip()).strip()