"""
evaluate.py
Compare 3 e-commerce RAG retrieval pipelines x LLM. Redis pipeline evaluated using other metrics

Pipelines:
  vector   — RAG.py(SBERT and BGEReranker)
  bm25     — RAG_BM25.py
  hybrid   — RAG_HYBRID_SEARCH.py (Hybrid Retriever and BGE Reranker)

Models:
  llama

What this script does:
  1. Loads your Chroma product catalog from ./product_db
  2. Loads evaluation queries from test_queries.json. Each query contains a product search request and a target category.
  3. Runs each pipeline on each query
  4. Computes retrieval metrics: Precision@K, NDCG@K
  5. Saves result files under ./eval_metric.csv

Important:
  change rereank top_n to 5 in RAG.py, and RAG_HYBRID_SEARCH.py.
"""

import json
import math
import asyncio
import argparse
import importlib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PIPELINES = {
    "vector": "RAG",
    "bm25": "RAG_BM25",
    "hybrid": "RAG_HYBRID_SEARCH"
}


def precision_at_k(retrieved_categories, target_category, k):
    top_k = retrieved_categories[:k]
    hits = sum(1 for c in top_k if c == target_category)
    return hits / k


def ndcg_at_k(retrieved_categories, target_category, k):
    dcg = 0

    for i, c in enumerate(retrieved_categories[:k], start=1):
        rel = 1 if c == target_category else 0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(
        sum(1 for c in retrieved_categories if c == target_category),
        k
    )

    if ideal_hits == 0:
        return 0

    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    return dcg / idcg


async def run_one_pipeline(module_name, queries):
    module = importlib.import_module(module_name)

    rows = []

    for q in queries:
        question = q["question"]
        target_category = q["category"]

        answer, products, contexts = await module.search_products_async(
            question,
            target_category
        )

        retrieved_categories = []

        for c in contexts:
            if target_category.lower() in c.lower():
                retrieved_categories.append(target_category)
            else:
                retrieved_categories.append("Other")


        row = {
            "question": question,
            "target_category": target_category,
            "retrieved_categories": retrieved_categories,
            "precision@5": precision_at_k(
                retrieved_categories,
                target_category,
                5
            ),
            "ndcg@5": ndcg_at_k(
                retrieved_categories,
                target_category,
                5
            ),
            "answer": answer,
        }

        rows.append(row)

    return rows


async def main(args):
    with open(args.queries, "r") as f:
        queries = json.load(f)

    all_rows = []

    for pipeline_name, module_name in PIPELINES.items():
        print(f"Running {pipeline_name}...")

        rows = await run_one_pipeline(
            module_name,
            queries
        )

        for row in rows:
            row["pipeline"] = pipeline_name

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df.to_csv(args.output, index=False)

    summary = df.groupby("pipeline")[
        [
            "precision@5",
            "ndcg@5"
        ]
    ].mean()

    print(summary)

    summary.to_csv("eval_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="test_queries.json")
    parser.add_argument("--output", default="eval_metric.csv")
    args = parser.parse_args()

    asyncio.run(main(args))