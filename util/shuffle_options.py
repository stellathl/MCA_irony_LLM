# =========================================================
# OPTION SHUFFLE HELPERS
# =========================================================

def parse_options(text):
    """Split a numbered answering_options string into a plain list."""
    lines = [l.strip() for l in str(text).strip().split("\n") if l.strip()]
    options = []
    for line in lines:
        if line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            options.append(line[2:].strip())
        else:
            options.append(line)
    return options

def format_options(options):
    """Rejoin a list of options into a numbered string."""
    return "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

def get_correct_option_text(row):
    """
    Return the text of the correct answer before any shuffling.
    Option 1 (index 0) = non-ironic interpretation  → correct when irony_label == non-ironic
    Option 2 (index 1) = ironic interpretation       → correct when irony_label == ironic
    Options 3 & 4 are always distractors.
    """
    options = parse_options(row["answering_options"])
    if len(options) < 2:
        return None
    return options[1] if row["irony_label"] == "ironic" else options[0]
