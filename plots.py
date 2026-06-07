import matplotlib.pyplot as plt
import numpy as np

pipelines = [
    "BM25+BGE",
    "SBERT+BGE",
    "Hybrid+BGE",
    "Redis+SBERT+BGE"
]

faithfulness = [0.552, 0.752, 0.811, 0.764]
latency = [1.469, 1.192, 1.435, 0.764]

# -------------------------------
# 1. Grouped Bar Chart
# -------------------------------
x = np.arange(len(pipelines))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, faithfulness, width, label="Faithfulness")
plt.bar(x + width/2, latency, width, label="Latency (s)")

plt.xlabel("Pipeline")
plt.ylabel("Score / Seconds")
plt.title("Faithfulness and Latency Across RAG Pipelines")
plt.xticks(x, pipelines, rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("faithfulness_latency_grouped_bar.png", dpi=300)
# plt.show()

# -------------------------------
# 2. Scatter Plot
# -------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(latency, faithfulness)

for i, label in enumerate(pipelines):
    plt.annotate(label, (latency[i], faithfulness[i]), textcoords="offset points", xytext=(6, 6))

plt.xlabel("Average Latency (seconds)")
plt.ylabel("Faithfulness Score")
plt.title("Faithfulness vs Latency Trade-off")
plt.tight_layout()
plt.savefig("faithfulness_latency_scatter.png", dpi=300)
# plt.show()

# -------------------------------
# 3. Grouped Box Chart
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load faithfulness scores
bm25 = pd.read_csv("faithfulness_scores_RAG_BM25.csv")
sbert = pd.read_csv("faithfulness_scores_RAG.csv")
hybrid = pd.read_csv("faithfulness_scores_RAG_HYBRID_SEARCH.csv")
redis = pd.read_csv("faithfulness_scores_RAG_Redis.csv")

# Data for boxplot
data = [
    bm25["faithfulness"],
    sbert["faithfulness"],
    hybrid["faithfulness"],
    redis["faithfulness"]
]

labels = [
    "BM25+BGE",
    "SBERT+BGE",
    "Hybrid+BGE",
    "Redis+SBERT+BGE"
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    showmeans=True
)

plt.ylabel("Faithfulness Score")
plt.xlabel("Pipeline")
plt.title("Distribution of Faithfulness Scores Across RAG Pipelines")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "faithfulness_boxplot_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()

# ====

import matplotlib.pyplot as plt
import numpy as np

pipelines = [
    "BM25+BGE",
    "SBERT+BGE",
    "Hybrid+BGE",
    "Redis+SBERT+BGE"
]

faithfulness = [0.552, 0.752, 0.811, 0.764]
answer_relevancy = [0.7394, 0.7263, 0.7516, 0.7408]
latency = [1.469, 1.192, 1.435, 0.764]

x = np.arange(len(pipelines))
width = 0.25

plt.figure(figsize=(9, 5))

plt.bar(
    x - width,
    answer_relevancy,
    width,
    label="Answer Relevancy"
)

plt.bar(
    x,
    faithfulness,
    width,
    label="Faithfulness"
)

plt.bar(
    x + width,
    latency,
    width,
    label="Latency (s)"
)

plt.xlabel("Pipeline")
plt.ylabel("Score / Seconds")
plt.title("Answer Relevancy, Faithfulness and Latency Across RAG Pipelines")

plt.xticks(x, pipelines, rotation=20, ha="right")
plt.legend()

plt.tight_layout()
plt.savefig("all_metrics_grouped_bar.png", dpi=300)
# plt.show()

# heatmap 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Data
# -------------------------------
data = {
    "Answer Relevancy": [0.7394, 0.7263, 0.7516, 0.7408],
    "Faithfulness": [0.552, 0.752, 0.811, 0.764],
    "Latency": [1.469, 1.192, 1.435, 0.764]
}

pipelines = [
    "BM25+BGE",
    "SBERT+BGE",
    "Hybrid+BGE",
    "Redis+SBERT+BGE"
]

df = pd.DataFrame(data, index=pipelines)

# -------------------------------
# Normalize metrics
# -------------------------------
scaler = MinMaxScaler()

df_norm = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# For latency, lower is better
df_norm["Latency"] = 1 - df_norm["Latency"]

# -------------------------------
# Heatmap
# -------------------------------
plt.figure(figsize=(8, 4))

sns.heatmap(
    df_norm,
    annot=True,
    cmap="YlGnBu",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Normalized Performance Comparison Across RAG Pipelines")
plt.ylabel("Pipeline")
plt.xlabel("Metric")

plt.tight_layout()
plt.savefig("rag_heatmap.png", dpi=300)
# plt.show()


# ======bubble 



pipelines = [
    "BM25+BGE",
    "SBERT+BGE",
    "Hybrid+BGE",
    "Redis+SBERT+BGE"
]

latency = [1.469, 1.192, 1.435, 0.764]
faithfulness = [0.552, 0.752, 0.811, 0.764]
answer_relevancy = [0.7394, 0.7263, 0.7516, 0.7408]

# Bubble size from answer relevancy
sizes = [x * 3500 for x in answer_relevancy]

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    latency,
    faithfulness,
    s=sizes,
    c=latency,           # color based on latency
    cmap="Blues",        # shades of blue
    alpha=0.7,
    edgecolors="black"
)

for i, label in enumerate(pipelines):
    plt.annotate(
        label,
        (latency[i], faithfulness[i]),
        xytext=(8, 8),
        textcoords="offset points"
    )

cbar = plt.colorbar(scatter)
cbar.set_label("Latency (seconds)")

plt.xlabel("Average Latency (seconds)")
plt.ylabel("Faithfulness Score")
plt.title("RAG Pipeline Performance Comparison")

plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("rag_bubble_plot_blue.png", dpi=300)
plt.show()