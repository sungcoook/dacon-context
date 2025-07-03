import pandas as pd
from collections import Counter


files = [
    "../submission/exp19.csv",
    "../submission/exp21.csv",
    "../submission/exp22.csv",
    "../submission/exp36.csv"
]

dfs = [pd.read_csv(fp).set_index("ID") for fp in files]

results = []
for idx in dfs[0].index:
    row_votes = [tuple(df.loc[idx]) for df in dfs]
    vote_counter = Counter(row_votes)
    top_tuple, freq = vote_counter.most_common(1)[0]

    chosen = top_tuple if freq >= 2 else row_votes[0]
    results.append((idx, *chosen))

out_cols = ["ID", "answer_0", "answer_1", "answer_2", "answer_3"]
pd.DataFrame(results, columns=out_cols).to_csv("../submission/ensemble.csv", index=False)

