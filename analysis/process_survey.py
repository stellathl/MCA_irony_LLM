"""
process_survey.py
Turns one LimeSurvey export + its stimulus CSV into one row per trial.

Usage:
  python process_survey.py --export results_list1.csv --stimuli stimuli_list1.csv --list 1 --out trials_list1.csv
"""
import argparse
import ast
import re

import pandas as pd

CATEGORIES = {"a": "literal", "b": "ironic", "c": "topic_echo", "d": "keyword"}

parser = argparse.ArgumentParser()
parser.add_argument("--export", required=True)
parser.add_argument("--stimuli", required=True)
parser.add_argument("--list", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ex = pd.read_csv(args.export)
st = pd.read_csv(args.stimuli).set_index("Item_ID")

#Answer columns (question text) and item codes (from timing columns), same order
q_cols = [c for c in ex.columns if c.endswith("What did the speaker mean?")]
conf_cols = [c for c in ex.columns if c.startswith("How confident are you")]
codes = [re.search(r"Q\d+(C\w+)", c).group(1)
         for c in ex.columns if re.match(r"Question time: Q\d+", c)]
assert len(q_cols) == len(conf_cols) == len(codes) == 72, "column count mismatch"


def code_to_item_id(code):
    # 'C101ANI' -> 'C1_01_A_NI'
    m = re.match(r"C(\d)(\d\d)(A|UA)(I|NI)$", code)
    return f"C{m[1]}_{m[2]}_{m[3]}_{m[4]}"


rows = []
for _, p in ex.iterrows():
    if p.iloc[7] != "I agree.":  # consent column
        continue
    for q, conf, code in zip(q_cols, conf_cols, codes):
        answer = p[q]
        if pd.isna(answer):
            continue
        item_id = code_to_item_id(code)
        s = st.loc[item_id]
        shown_pos = answer.strip()[0]  # letter the participant saw
        mapping = ast.literal_eval(s["original_option_mapping"])
        category = CATEGORIES[mapping["abcd".index(shown_pos)]]
        irony = "ironic" if s["irony_label"] == "ironic" else "non_ironic"
        rows.append({
            "participant": f"L{args.list}_{p['Response ID']}",
            "list": args.list,
            "completed": p["Last page"] == 74,
            "item": s["base_item"],
            "irony": irony,
            "richness": s["context_level"],
            "shown_position": shown_pos,
            "chosen_category": category,
            "correct": int(category == ("ironic" if irony == "ironic" else "literal")),
            "confidence": p[conf],
        })

trials = pd.DataFrame(rows)
trials.to_csv(args.out, index=False)
print(f"{len(trials)} trials from {trials['participant'].nunique()} participants "
      f"({trials.groupby('participant')['completed'].first().sum()} completed)")
