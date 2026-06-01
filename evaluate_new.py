"""
evaluate.py — Compare 4 RAG retrieval pipelines x 2 LLMs using RAGAS metrics.

Pipelines:
  vector   — SBERT embeddings + BGE reranker          (RAG.py)
  bm25     — BM25 keyword search + BGE reranker        (RAG_BM25.py)
  hybrid   — BM25 + vector via RRF + BGE reranker      (RAG_HYBRID_SEARCH.py)
  redis    — hybrid + Redis semantic cache             (RAG_Redis.py)

Models (set via GROQ_API_KEY in .env):
  llama    — llama-3.1-8b-instant
  qwen     — qwen-2.5-72b-instruct

Usage:
  python evaluate.py                           # all pipelines, all models
  python evaluate.py --pipelines vector bm25   # subset
  python evaluate.py --models llama            # single model
  python evaluate.py --queries test_queries.json --output results.csv
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import traceback
import types
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Fix numpy 2.0 breaking change
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

# ---------------------------------------------------------------------------
# Compatibility stubs — must run before any ragas/langchain import
# ---------------------------------------------------------------------------

# Step 1: import real langchain_community packages first so stubs don't overwrite them
try:
    import langchain_community
    import langchain_community.llms
    import langchain_community.chat_models
    import langchain_community.vectorstores
except Exception:
    pass

# Step 2: inject missing attributes into real packages
_dummy = lambda name: type(name, (), {})

try:
    if not hasattr(langchain_community.llms, "VertexAI"):
        langchain_community.llms.VertexAI = _dummy("VertexAI")
    if not hasattr(langchain_community.chat_models, "ChatVertexAI"):
        langchain_community.chat_models.ChatVertexAI = _dummy("ChatVertexAI")
    if not hasattr(langchain_community.chat_models, "ChatGooglePalm"):
        langchain_community.chat_models.ChatGooglePalm = _dummy("ChatGooglePalm")
except Exception:
    pass

# Step 3: stub missing submodules ragas tries to import from
for _mod, _attr in [
    ("langchain_community.llms.vertexai",           "VertexAI"),
    ("langchain_community.chat_models.vertexai",    "ChatVertexAI"),
    ("langchain_community.chat_models.google_palm", "ChatGooglePalm"),
]:
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        setattr(_m, _attr, _dummy(_attr))
        sys.modules[_mod] = _m

# Step 4: stub langchain_core.pydantic_v1 (removed in langchain-core>=0.3)
try:
    import langchain_core.pydantic_v1
except (ImportError, ModuleNotFoundError):
    try:
        import pydantic.v1 as _pv1
    except ImportError:
        import pydantic as _pv1
    sys.modules["langchain_core.pydantic_v1"] = _pv1

# Step 5: stub ModelProfile (missing in older langchain-core)
try:
    from langchain_core.language_models import ModelProfile
except ImportError:
    _MP = _dummy("ModelProfile")
    import langchain_core.language_models as _lm
    _lm.__dict__["ModelProfile"] = _MP
    try:
        import langchain_core.language_models.base as _lmb
        _lmb.ModelProfile = _MP
    except Exception: pass
    try:
        import langchain_core.language_models.chat_models as _lmc
        _lmc.ModelProfile = _MP
    except Exception: pass

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

def _score_answer_relevancy(questions, answers):
    valid = [(q, a) for q, a in zip(questions, answers) if q.strip() and a.strip()]
    if not valid:
        return float("nan")
    qs, ans = zip(*valid)
    q_embs = _embeddings_model.embed_documents(list(qs))
    a_embs = _embeddings_model.embed_documents(list(ans))
    scores = [cosine_similarity([qe], [ae])[0][0] for qe, ae in zip(q_embs, a_embs)]
    return float(sum(scores) / len(scores))

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

MODELS = {
    "llama": "llama-3.1-8b-instant",
    "qwen":  "qwen-2.5-72b-instruct",
}

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
    module._rag_chain = None
    import langchain_groq as _g
    original = _g.ChatGroq.__init__
    def patched(self, *args, **kwargs):
        kwargs["model"] = model_id
        kwargs.setdefault("groq_api_key", os.getenv("GROQ_API_KEY"))
        original(self, *args, **kwargs)
    _g.ChatGroq.__init__ = patched

def _restore_llm(original):
    import langchain_groq as _g
    _g.ChatGroq.__init__ = original

async def _run_pipeline(module, question: str, category: Optional[str]):
    try:
        answer, _, contexts = await module.search_products_async(question, category)
        if not answer:   print("    [WARN] Empty answer returned")
        if not contexts: print("    [WARN] No contexts returned")
        return answer, contexts
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        print(traceback.format_exc())
        return "", []

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(pipeline_keys, model_keys, queries, output_path):
    import langchain_groq as _g
    original = _g.ChatGroq.__init__

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        sys.exit("GROQ_API_KEY not set in .env — aborting.")



    all_rows = []

    for pipeline_key in pipeline_keys:
        module_name, pipeline_label = PIPELINES[pipeline_key]
        print(f"\n{'='*60}\nPipeline: {pipeline_label}  (module: {module_name})\n{'='*60}")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"  [SKIP] Could not import {module_name}: {exc}"); continue

        for model_key in model_keys:
            model_id = MODELS[model_key]
            print(f"\n  Model: {model_key} ({model_id})")
            _patch_llm(module, model_id)

            questions, answers, contexts_list = [], [], []
            for i, q in enumerate(queries, 1):
                question = q["question"]
                category = q.get("category")
                print(f"    [{i}/{len(queries)}] {question[:60]}")
                t0 = time.time()
                answer, contexts = await _run_pipeline(module, question, category)
                await asyncio.sleep(3)  # avoid rate limit
                print(f"           -> {len(contexts)} chunks, {time.time()-t0:.1f}s")
                questions.append(question)
                answers.append(answer)
                contexts_list.append(contexts if contexts else [""])

            print(f"\n  Computing answer_relevancy...")
            r = _score_answer_relevancy(questions, answers)

            all_rows.append({
                "pipeline":         pipeline_label,
                "model":            model_key,
                "answer_relevancy": round(r, 4),
            })
            print(f"\n  Results -> answer_relevancy={r:.3f}")

    _restore_llm(original)

    if not all_rows:
        print("[ERROR] No results collected — all pipelines failed"); return

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    for m in df["model"].unique():
        sub = df[df["model"] == m].copy()
        best = sub.sort_values("answer_relevancy", ascending=False).iloc[0]
        print(f"\nBest pipeline for [{m}]: {best['pipeline']}  "
              f"(answer_relevancy={best['answer_relevancy']:.3f})")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", choices=list(PIPELINES.keys()), default=list(PIPELINES.keys()))
    parser.add_argument("--models",    nargs="+", choices=list(MODELS.keys()),    default=list(MODELS.keys()))
    parser.add_argument("--queries", default="/content/drive/MyDrive/intelligent_ecom/test_queries.json")
    parser.add_argument("--output",  default="/content/drive/MyDrive/intelligent_ecom/eval_results.csv")
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

    asyncio.run(run_evaluation(args.pipelines, args.models, queries, args.output))

if __name__ == "__main__":
    main()
