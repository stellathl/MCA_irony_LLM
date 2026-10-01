import pandas as pd

trials = pd.concat([pd.read_csv(f"trials_list{i}.csv") for i in [1, 2, 4]])

print(trials.groupby("list")["participant"].nunique())
print(trials.groupby(["irony", "richness"])["correct"].mean().round(2))

trials.to_csv("human_trials.csv", index = False)
