# run_all_pipelines.py
import asyncio
import json
import sys
import types

from langchain_core.globals import set_llm_cache
set_llm_cache(None)
print("LLM cache disabled ✅")

# ── Compatibility stubs ───────────────────────────────────────────────────────
_dummy = lambda name: type(name, (), {})
for _mod, _attr in [
    ("langchain_community.chat_models.vertexai",    "ChatVertexAI"),
    ("langchain_community.chat_models.google_palm", "ChatGooglePalm"),
    ("langchain_community.llms.vertexai",           "VertexAI"),
]:
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        setattr(_m, _attr, _dummy(_attr))
        sys.modules[_mod] = _m

# ── Pipeline registry — add or remove as needed ───────────────────────────────
PIPELINES = [
             # BM25 + BGE
    # "RAG_HYBRID_SEARCH",  # hybrid + BGE  ← already done, will be skipped
    "RAG_Redis",          # hybrid + Redis
]

SKIP_IF_EXISTS = False   # ← set False to re-run pipelines you already have


async def run_pipeline(module_name: str, queries: list[dict]):
    import importlib
    out_file  = f"rag_outputs_{module_name}.json"
    skip_file = f"rag_skipped_{module_name}.json"

    # Skip if output already exists
    import os
    if SKIP_IF_EXISTS and os.path.exists(out_file):
        print(f"\n[SKIP] {out_file} already exists — skipping {module_name}")
        return

    print(f"\n{'='*60}")
    print(f"Running pipeline: {module_name}")
    print(f"{'='*60}")

    try:
        pipeline = importlib.import_module(module_name)
    except Exception as e:
        print(f"  [ERROR] Could not import {module_name}: {e}")
        return

    outputs = []
    skipped = []

    for i, q in enumerate(queries, 1):
        question = q["question"]
        category = q.get("category")
        print(f"  [{i:02d}/{len(queries)}] {question[:65]}")

        try:
            answer, _, contexts = await pipeline.search_products_async(
                question, category
            )

            if not answer.strip():
                print(f"    [WARN] Empty answer — skipping")
                skipped.append(q)
                continue

            outputs.append({
                "question": question,
                "category": category,
                "answer":   answer,
                "context":  "\n".join(contexts),
            })
            print(f"    ✓ {len(contexts)} chunks retrieved")

        except Exception as e:
            print(f"    [ERROR] {e}")
            skipped.append(q)

        await asyncio.sleep(15)  # avoid Groq rate limit

    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=2)
    # with open(skip_file, "w") as f:
    #     json.dump(skipped, f, indent=2)

    print(f"\n  ✅ Done: {len(outputs)} saved → {out_file}")
    print(f"      Skipped: {len(skipped)} → {skip_file}")


async def main():
    with open("test_queries50.json") as f:
        queries = json.load(f)

    # queries = queries[:30]

    print(f"Loaded {len(queries)} queries")
    print(f"Pipelines to run: {PIPELINES}")

    for module_name in PIPELINES:
        await run_pipeline(module_name, queries)

    print(f"\n{'='*60}")
    print("ALL PIPELINES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())