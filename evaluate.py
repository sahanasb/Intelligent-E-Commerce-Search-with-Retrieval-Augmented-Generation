"""
evaluate.py — Compare 4 RAG retrieval pipelines × 2 LLMs using RAGAS metrics.

Pipelines:
  vector   — SBERT embeddings + BGE reranker          (RAG.py)
  bm25     — BM25 keyword search + BGE reranker        (RAG_BM25.py)
  hybrid   — BM25 + vector via RRF + BGE reranker      (RAG_HYBRID_SEARCH.py)
  redis    — hybrid + Redis semantic cache             (RAG_Redis.py)

Models (set via GROQ_API_KEY in .env):
  llama    — meta-llama/llama-3.3-70b-versatile
  qwen     — qwen/qwen-2.5-72b-instruct  (or qwen-2.5-coder-32b-instruct)

Usage:
  python evaluate.py                          # all pipelines, all models
  python evaluate.py --pipelines vector bm25  # subset of pipelines
  python evaluate.py --models llama           # single model
  python evaluate.py --queries test_queries.json
  python evaluate.py --output results.csv
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)
from datasets import Dataset

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen-2.5-72b-instruct",
}

# ---------------------------------------------------------------------------
# Pipeline registry
# Each entry: (module_name, human_label)
# Every module exposes:  search_products_async(question, category_filter) ->
#                        (answer: str, products: list[dict], contexts: list[str])
# ---------------------------------------------------------------------------

PIPELINES = {
    "vector": ("RAG",               "Vector (SBERT+BGE)"),
    "bm25":   ("RAG_BM25",          "BM25+BGE"),
    "hybrid": ("RAG_HYBRID_SEARCH", "Hybrid (BM25+Vector+BGE)"),
    "redis":  ("RAG_Redis",         "Hybrid+Redis Cache"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_llm(module, model_id: str):
    """
    Hot-swap the LLM inside an already-imported RAG module.
    Each module caches its chain in a module-level `_rag_chain`; reset it so
    `_build_chain()` rebuilds with the new model on the next call.
    """
    module._rag_chain = None  # force chain rebuild

    # Monkey-patch ChatGroq so the next _build_chain() call picks up our model
    import langchain_groq as _groq_mod

    original_init = _groq_mod.ChatGroq.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["model"] = model_id
        kwargs.setdefault("groq_api_key", os.getenv("GROQ_API_KEY"))
        original_init(self, *args, **kwargs)

    _groq_mod.ChatGroq.__init__ = patched_init


def _restore_llm(original_init):
    import langchain_groq as _groq_mod
    _groq_mod.ChatGroq.__init__ = original_init


async def _run_pipeline(module, question: str, category: Optional[str]):
    """Call search_products_async and return (answer, contexts)."""
    try:
        answer, _, contexts = await module.search_products_async(question, category)
        return answer, contexts
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        return "", []


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(
    pipeline_keys: list[str],
    model_keys: list[str],
    queries: list[dict],
    output_path: str,
):
    import langchain_groq as _groq_mod
    original_init = _groq_mod.ChatGroq.__init__

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        sys.exit("GROQ_API_KEY not set in .env — aborting.")

    # Judge LLM used by RAGAS (reuse llama for scoring to keep costs predictable)
    judge_llm = ChatGroq(
        model=MODELS["llama"],
        groq_api_key=groq_api_key,
        temperature=0,
        max_tokens=1024,
    )

    all_rows = []

    for pipeline_key in pipeline_keys:
        module_name, pipeline_label = PIPELINES[pipeline_key]

        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_label}  (module: {module_name})")
        print(f"{'='*60}")

        # Import the pipeline module (only once per process)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"  [SKIP] Could not import {module_name}: {exc}")
            continue

        for model_key in model_keys:
            model_id = MODELS[model_key]
            print(f"\n  Model: {model_key} ({model_id})")

            # Swap the LLM inside the module
            _patch_llm(module, model_id)

            ragas_questions = []
            ragas_answers   = []
            ragas_contexts  = []

            for i, q in enumerate(queries, 1):
                question = q["question"]
                category = q.get("category")
                print(f"    [{i}/{len(queries)}] {question[:60]}")

                t0 = time.time()
                answer, contexts = await _run_pipeline(module, question, category)
                elapsed = time.time() - t0
                print(f"           → {len(contexts)} context chunks, {elapsed:.1f}s")

                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts if contexts else [""])

            # Build RAGAS dataset
            dataset = Dataset.from_dict({
                "question":  ragas_questions,
                "answer":    ragas_answers,
                "contexts":  ragas_contexts,
            })

            print(f"\n  Running RAGAS scoring for {pipeline_label} / {model_key} ...")
            try:
                result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision],
                    llm=judge_llm,
                    raise_exceptions=False,
                )
                scores = result.to_pandas()
                mean_faith   = scores["faithfulness"].mean()
                mean_rel     = scores["answer_relevancy"].mean()
                mean_ctx_prec = scores["context_precision"].mean()
            except Exception as exc:
                print(f"  [RAGAS ERROR] {exc}")
                mean_faith = mean_rel = mean_ctx_prec = float("nan")

            row = {
                "pipeline":          pipeline_label,
                "model":             model_key,
                "faithfulness":      round(mean_faith, 4),
                "answer_relevancy":  round(mean_rel, 4),
                "context_precision": round(mean_ctx_prec, 4),
            }
            all_rows.append(row)
            print(f"\n  Results → faithfulness={mean_faith:.3f}  "
                  f"answer_relevancy={mean_rel:.3f}  "
                  f"context_precision={mean_ctx_prec:.3f}")

    # Restore original ChatGroq.__init__
    _restore_llm(original_init)

    # Save and display summary
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Best pipeline per model
    for m in df["model"].unique():
        sub = df[df["model"] == m].copy()
        sub["avg_score"] = sub[["faithfulness", "answer_relevancy", "context_precision"]].mean(axis=1)
        best = sub.sort_values("avg_score", ascending=False).iloc[0]
        print(f"\nBest pipeline for [{m}]: {best['pipeline']}  "
              f"(avg={best['avg_score']:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipelines × LLMs")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=list(PIPELINES.keys()),
        default=list(PIPELINES.keys()),
        help="Pipelines to evaluate (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="LLMs to evaluate (default: llama qwen)",
    )
    parser.add_argument(
        "--queries",
        default="test_queries.json",
        help="Path to JSON file with test queries (default: test_queries.json)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="Output CSV path (default: eval_results.csv)",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        sys.exit(f"Queries file not found: {queries_path}")

    with open(queries_path) as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Pipelines : {args.pipelines}")
    print(f"Models    : {args.models}")
    print(f"Output    : {args.output}")

    asyncio.run(
        run_evaluation(
            pipeline_keys=args.pipelines,
            model_keys=args.models,
            queries=queries,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
