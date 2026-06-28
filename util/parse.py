import pandas as pd
import re

from util.constants import MODELS

# ── Helper: parse "number — reasoning" from model response ──
def parse_response(response_text):
    """
    Extracts the chosen option number and reasoning from model output.
    Handles formats like:
        "2 — Because..."
        "2. Because..."
        "Option 2, because..."
        "2\nBecause..."
    Returns (chosen, reasoning_text)
    """
    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None

    text = str(response_text).strip()

    # Extract the first letter that appears (a–d)
    match = re.search(r"\b([a-dA-D])\b", text)
    if not match:
        return None, text  # couldn't parse a number

    chosen = match.group(1).lower() if match else None

    # Everything after the number is the reasoning
    reasoning = text[match.end():].strip()
    # Clean leading punctuation/separator (—, -, ., :, etc.)
    reasoning = re.sub(r"^[\s\-—.,;:]+", "", reasoning).strip()

    return chosen, reasoning
