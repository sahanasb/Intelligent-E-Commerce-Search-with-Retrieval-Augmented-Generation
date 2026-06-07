# latency_test.py

import asyncio
import json
import time
import importlib

PIPELINES = [
    "RAG_BM25",
    "RAG",
    "RAG_HYBRID_SEARCH",
    # "RAG_Redis",
]

async def run_pipeline(module_name, queries):
    pipeline = importlib.import_module(module_name)

    latencies = []

    print(f"\nRunning: {module_name}")

    for i, q in enumerate(queries, 1):
        question = q["question"]
        category = q.get("category")

        start = time.perf_counter()

        try:
            answer, _, contexts = await pipeline.search_products_async(
                question,
                category
            )

            latency = time.perf_counter() - start
            latencies.append(latency)

            print(
                f"[{i:02d}/{len(queries)}] "
                f"{latency:.2f}s"
            )

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(3)

    avg_latency = sum(latencies) / len(latencies)

    print(
        f"\nAverage Latency ({module_name}): "
        f"{avg_latency:.3f} seconds"
    )

    return avg_latency


async def main():
    with open("test_queries10.json") as f:
        queries = json.load(f)

    results = {}

    for pipeline_name in PIPELINES:
        avg_latency = await run_pipeline(
            pipeline_name,
            queries
        )

        results[pipeline_name] = round(avg_latency, 3)

    print("\nLatency Summary")
    print("-" * 40)

    for k, v in results.items():
        print(f"{k:<25} {v:.3f}s")

    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to latency_results.json")


if __name__ == "__main__":
    asyncio.run(main())
