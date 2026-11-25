import pandas as pd
import matplotlib.pyplot as plt

from config import paths


files = {
    "Dataset V2 + Hybrid": paths.E3_RESULTS / "deterministic_retrieval_processed_jobs_dataset_v2_hybrid.csv",
    "Dataset V2": paths.E3_RESULTS / "deterministic_retrieval_processed_jobs_dataset_v2.csv",
}

series = {}

for label, path in files.items():
    df = pd.read_csv(path)

    # Exclude negative queries
    df = df[df["relevant_job_ids"].fillna("[]") != "[]"]

    s = df.set_index(["query_id", "k"])["precision"]

    print(label)
    print("rows:", len(s))
    print("unique:", s.index.is_unique)
    print("duplicates:", s.index[s.index.duplicated()].tolist())

    series[label] = s

scores = pd.DataFrame(series).sort_index()


plt.figure(figsize=(16, 6))

colors = {}

for label in scores.columns:
    line, = plt.plot(
        range(len(scores)),
        scores[label],
        label=label,
        marker=".",
        alpha=0.8,
    )
    colors[label] = line.get_color()


plt.ylim(0, 1)
plt.axhline(0, linewidth=0.5)

plt.xticks(
    range(len(scores)),
    [f"{query_id}\nK={k}" for query_id, k in scores.index],
    rotation=90,
)

plt.ylabel("Precision@K")
plt.title("Precision@K — Baseline vs Dataset V2")

plt.legend()
plt.tight_layout()
plt.show()