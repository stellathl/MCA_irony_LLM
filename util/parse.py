import pandas as pd
import re


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

    return chosen, reasoning

def parse_response_rsa(response_text):
    """
    RSA parser.
    Priority:
    1. Final Answer
    2. Pragmatic Listener (L1)
    3. L1
    """

    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None

    text = str(response_text).strip()

    # 1. Final Answer
    m = re.search(
        r"Final\s*Answer\s*:?\s*([a-d])",
        text,
        re.IGNORECASE,
    )

    if m:
        chosen = m.group(1).lower()

    else:
        # 2. Pragmatic Listener (L1)
        m = re.search(
            r"Pragmatic\s+Listener.*?\(L1\)\s*:?\s*([a-d])",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if m:
            chosen = m.group(1).lower()

        else:
            # 3. L1
            m = re.search(
                r"\bL1\b\s*[:=-]?\s*([a-d])",
                text,
                re.IGNORECASE,
            )

            if m:
                chosen = m.group(1).lower()
            else:
                chosen = None

    return chosen, text


def _extract_reasoning_label(text):
    """Return text after the last 'Reasoning:' label, or None."""
    m = re.search(r"reasoning\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip() or None
    return None


def _clean(s):
    return re.sub(r"^[\s\-—.,;:]+", "", s.strip()).strip()
