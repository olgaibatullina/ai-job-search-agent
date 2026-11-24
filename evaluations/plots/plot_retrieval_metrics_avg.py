import pandas as pd
import matplotlib.pyplot as plt

from config import paths


# Path to the evaluation CSV
csv_path = paths.E3_RESULTS / "deterministic_retrieval_2026-11-19.csv"

# Load results
df = pd.read_csv(csv_path)

# Exclude negative queries
df = df[df["relevant_job_ids"].notna()]
df = df[df["relevant_job_ids"].str.strip() != "[]"]

# Calculate mean metric value for each K
metrics_by_k = (
    df.groupby("k")[["precision", "recall", "f1"]]
      .mean()
      .reset_index()
      .sort_values("k")
)

# Print aggregated values
print(metrics_by_k.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    metrics_by_k["k"],
    metrics_by_k["precision"],
    marker="o",
    label="Precision@K",
)

plt.plot(
    metrics_by_k["k"],
    metrics_by_k["recall"],
    marker="o",
    label="Recall@K",
)

plt.plot(
    metrics_by_k["k"],
    metrics_by_k["f1"],
    marker="o",
    label="F1@K",
)

plt.xlabel("K")
plt.ylabel("Score")
plt.title("Retrieval Metrics by K")

plt.xticks(metrics_by_k["k"])
plt.ylim(0, 1)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()