import pandas as pd
import matplotlib.pyplot as plt

from config import paths

files = {
    "Response Groundedness baseline": paths.E2_RESULTS / "e2_responsegroundedness_baseline.csv",
    "Response Groundedness final": paths.E2_RESULTS / "e2_responsegroundedness_final.csv",
}

series = {}

for label, path in files.items():
    df = pd.read_csv(path)

    s = df.set_index("query_id")["judge_avg_output_tokens_per_call"]

    print(label)
    print("rows:", len(s))
    print("unique:", s.index.is_unique)
    print("duplicates:", s.index[s.index.duplicated()].tolist())

    series[label] = s

    series[label] = df.set_index("query_id")["score"]

scores = pd.DataFrame(series).sort_index()

plt.figure(figsize=(16, 6))

colors = {}

for label in scores.columns:
    line, = plt.plot(
        range(len(scores)),
        scores[label],
        label=label,
        marker=".",
        alpha=0.8
    )
    colors[label] = line.get_color()

# Mark evaluation errors below the score axis
for i, (label, path) in enumerate(files.items()):
    df = pd.read_csv(path)
    
    error_ids = df.loc[df["status"] == "ERROR", "query_id"]
    nan_ids = df.loc[df["score"].isna(), "query_id"]

    positions = [
    scores.index.get_loc(qid)
    for qid in nan_ids
    if qid in scores.index
]

    if positions:
        plt.scatter(
            positions,
            [-0.08 - i * 0.05] * len(positions),
            color=colors[label],
            marker="x",
            s=65,
            label=f"{label} — NaN",
            zorder=5,
        )

plt.ylim(-0.15, 1.05)
plt.axhline(0, color="gray", linewidth=0.5)

plt.xticks(
    range(len(scores)),
    scores.index,
    rotation=90
)

plt.legend()
plt.tight_layout()
plt.show()