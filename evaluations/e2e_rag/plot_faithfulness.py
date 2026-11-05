import pandas as pd
import matplotlib.pyplot as plt

from config.paths import E1_RESULTS

files = {
    # "V1 + Thinking ON": E1_RESULTS / "faithfulness_baseline_2026-10-30.csv",
    # "V1 + Thinking OFF": E1_RESULTS / "faithfulness_prompt_v1_nothinking_2026-11-03.csv",
    # "V2 + Thinking ON": E1_RESULTS / "faithfulness_prompt_v2_2026-11-03.csv",
    # "V2 + Thinking OFF": E1_RESULTS / "faithfulness_prompt_v2_nothinking_2026-11-03.csv",
    "V1 + Thinking ON + temp0.6 + top_p0.95": E1_RESULTS / "faithfulness_prompt_v1_thinking_on_temp06_top_p095_2026-11-04.csv",
    "V1 + Thinking ON + temp0.6 + top_p0.95 + repeat_penalty1.1": E1_RESULTS / "faithfulness_prompt_v1_thinking_on_temp06_top_p095_repeat_penalty_1-1_2026-11-04.csv",
    # "Faithfulness baseline": "local_tools/data/e1_faithfulness_baseline.csv",
    # "Faithfulness final": "local_tools/data/e1_faithfulness_final.csv",
    # "Answer Relevancy baseline": "local_tools/data/e1_relevancy_baseline.csv",
    # "Answer Relevancy baseline": "local_tools/data/e1_relevancy_final.csv",
}

series = {}

for label, path in files.items():
    df = pd.read_csv(path)
    series[label] = df.set_index("query_id")["judge_avg_output_tokens_per_call"]

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

    positions = [
        scores.index.get_loc(qid)
        for qid in error_ids
        if qid in scores.index
    ]

    plt.scatter(
        positions,
        [-0.08 - i * 0.05] * len(positions),
        color=colors[label],
        marker="x",
        s=65,
        label=f"{label} — ERROR",
        zorder=5
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