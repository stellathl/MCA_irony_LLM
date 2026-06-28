import pandas as pd
import re
from util.constants import MODELS

# ── Helper: parse "Answer: N" + reasoning block from model response ──
def parse_response(response_text):
    """
    Extracts the chosen option letter and reasoning from model output.
    Expects output containing a "Reasoning:" section and an "Answer:" line, e.g.:
        Reasoning:
        * bullet one
        * bullet two
        Answer: c
    Also tolerates inline variants like "Answer: b — because...".
    Returns (chosen_letter, reasoning_text)
    """
    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None

    text = str(response_text).strip()
    # Find the LAST "Answer:" occurrence (in case the prompt itself contains one,
    # e.g. the echoed INPUT block also has "Answer:" with nothing after it)
    answer_matches = list(re.finditer(r"answer\s*:\s*", text, flags=re.IGNORECASE))
    if not answer_matches:
        return None, text  # no "Answer:" marker found at all
    last_answer = answer_matches[-1]
    # Grab the first letter (a-d) that follows "Answer:"
    after_answer = text[last_answer.end():]
    letter_match = re.search(r"\b([a-dA-D])\b", after_answer)
    if not letter_match:
        return None, text  # found "Answer:" but no valid letter after it
    chosen = letter_match.group(1).lower()
    # Anything after the letter, on the Answer line, in case there's trailing reasoning
    trailing = after_answer[letter_match.end():].strip()
    trailing = re.sub(r"^[\s\-—.,;:]+", "", trailing).strip()
    # Try to grab the "Reasoning:" section that precedes this Answer line
    reasoning_match = re.search(
        r"reasoning\s*:\s*(.*?)(?=answer\s*:)",
        text[:last_answer.start()] + text[last_answer.start():],  # search whole text up to Answer
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = trailing if trailing else None
    return chosen, reasoning