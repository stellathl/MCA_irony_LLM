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
    Returns (chosen_number, reasoning_text)
    """
    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None

    text = str(response_text).strip()

    # Extract the first number that appears (1–4)
    match = re.search(r"\b([1-4])\b", text)
    if not match:
        return None, text  # couldn't parse a number

    chosen = int(match.group(1))

    # Everything after the number is the reasoning
    reasoning = text[match.end():].strip()
    # Clean leading punctuation/separator (—, -, ., :, etc.)
    reasoning = re.sub(r"^[\s\-—.,;:]+", "", reasoning).strip()

    return chosen, reasoning

def parse_response_rsa(response_text):
    """RSA only - extracts Pragmatic Listener (L1)"""
    if pd.isna(response_text) or str(response_text).strip() == "":
        return None, None
    text = str(response_text).strip()
    l1_match = re.search(r'Pragmatic Listener.*?\(L1\):\s*(\d)', text, re.IGNORECASE)
    
    if not l1_match:
        all_numbers = re.findall(r'\b([1-4])\b', text)
        chosen = int(all_numbers[-1]) if all_numbers else None
    else:
        chosen = int(l1_match.group(1))
    
    l1_index = text.find('Pragmatic Listener')
    reasoning = ""
    if l1_index != -1:
        remaining = text[l1_index:]
        lines = remaining.split('\n', 1)
        reasoning = lines[1].strip() if len(lines) > 1 else ""
    
    reasoning = re.sub(r"^[\s\-—.,;:*]+", "", reasoning).strip()
    return chosen, reasoning