# # score_faithfulness.py
# import json
# import re
# import sys
# from langchain_huggingface import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity

# pipeline_name = sys.argv[1] if len(sys.argv) > 1 else "RAG_HYBRID_SEARCH"
# input_file    = f"rag_outputs_{pipeline_name}.json"
# output_file   = f"faithfulness_{pipeline_name}.json"

# print(f"Scoring: {input_file}")

# with open(input_file) as f:
#     records = json.load(f)

# emb = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
# )

# THRESHOLD = 0.40

# def extract_claims(answer: str) -> list[str]:
#     answer = answer.split("=== PRODUCTS ===")[0].strip()
#     sentences = re.split(r'(?<=[.!?])\s+', answer)
#     return [s.strip() for s in sentences if len(s.split()) >= 8]

# def score_faithfulness_record(answer: str, context: str) -> dict:
#     claims = extract_claims(answer)
#     if not claims:
#         return {"score": 0.0, "supported": 0, "total": 0, "unsupported_claims": []}

#     chunks     = [c.strip() for c in context.split("\n") if c.strip()]
#     claim_vecs = emb.embed_documents(claims)
#     chunk_vecs = emb.embed_documents(chunks)

#     supported   = []
#     unsupported = []

#     for claim, claim_vec in zip(claims, claim_vecs):
#         sims     = [float(cosine_similarity([claim_vec], [cv])[0][0]) for cv in chunk_vecs]
#         best_sim = max(sims)
#         if best_sim >= THRESHOLD:
#             supported.append(claim)
#         else:
#             unsupported.append((claim, round(best_sim, 3)))

#     score = len(supported) / len(claims)
#     return {
#         "score":              round(score, 4),
#         "supported_count":    len(supported),
#         "total_claims":       len(claims),
#         "unsupported_claims": unsupported,
#     }

# results = []
# total   = 0.0

# for i, rec in enumerate(records):
#     r = score_faithfulness_record(rec["answer"], rec["context"])
#     total += r["score"]
#     results.append({
#         "question":           rec["question"],
#         "faithfulness":       r["score"],
#         "supported_claims":   r["supported_count"],
#         "total_claims":       r["total_claims"],
#         "unsupported_claims": [c for c, _ in r["unsupported_claims"]],
#     })
#     print(f"[{i+1:02d}] {rec['question'][:55]:<55} "
#           f"faith={r['score']:.3f}  "
#           f"({r['supported_count']}/{r['total_claims']} claims supported)")

# avg = total / len(records)
# print(f"\nAverage Faithfulness ({pipeline_name}): {avg:.4f}")

# with open(output_file, "w") as f:
#     json.dump({
#         "pipeline":            pipeline_name,
#         "average_faithfulness": round(avg, 4),
#         "per_query":           results
#     }, f, indent=2)

# print(f"Saved to {output_file}")


# ====================================================================================

# score_faithfulness.py

import json
import re
import sys
import csv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

pipeline_name = sys.argv[1] if len(sys.argv) > 1 else "RAG_HYBRID_SEARCH"

input_file = f"rag_outputs_{pipeline_name}.json"
output_file = f"faithfulness_{pipeline_name}.json"
csv_file = f"faithfulness_scores_{pipeline_name}.csv"

print(f"Scoring: {input_file}")

with open(input_file) as f:
    records = json.load(f)

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

THRESHOLD = 0.40


def extract_claims(answer: str) -> list[str]:
    answer = answer.split("=== PRODUCTS ===")[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    return [s.strip() for s in sentences if len(s.split()) >= 8]


def score_faithfulness_record(answer: str, context: str) -> dict:
    claims = extract_claims(answer)

    if not claims:
        return {
            "score": 0.0,
            "supported_count": 0,
            "total_claims": 0,
            "unsupported_claims": [],
        }

    chunks = [c.strip() for c in context.split("\n") if c.strip()]

    if not chunks:
        return {
            "score": 0.0,
            "supported_count": 0,
            "total_claims": len(claims),
            "unsupported_claims": [(c, 0.0) for c in claims],
        }

    claim_vecs = emb.embed_documents(claims)
    chunk_vecs = emb.embed_documents(chunks)

    supported = []
    unsupported = []

    for claim, claim_vec in zip(claims, claim_vecs):
        sims = [
            float(cosine_similarity([claim_vec], [cv])[0][0])
            for cv in chunk_vecs
        ]

        best_sim = max(sims)

        if best_sim >= THRESHOLD:
            supported.append(claim)
        else:
            unsupported.append((claim, round(best_sim, 3)))

    score = len(supported) / len(claims)

    return {
        "score": round(score, 4),
        "supported_count": len(supported),
        "total_claims": len(claims),
        "unsupported_claims": unsupported,
    }


results = []
total = 0.0

for i, rec in enumerate(records):
    r = score_faithfulness_record(rec["answer"], rec["context"])
    total += r["score"]

    row = {
        "query_id": i + 1,
        "question": rec["question"],
        "faithfulness": r["score"],
        "supported_claims": r["supported_count"],
        "total_claims": r["total_claims"],
        "unsupported_claims": [c for c, _ in r["unsupported_claims"]],
    }

    results.append(row)

    print(
        f"[{i+1:02d}] {rec['question'][:55]:<55} "
        f"faith={r['score']:.3f} "
        f"({r['supported_count']}/{r['total_claims']} claims supported)"
    )

avg = total / len(records)

print(f"\nAverage Faithfulness ({pipeline_name}): {avg:.4f}")

with open(output_file, "w") as f:
    json.dump(
        {
            "pipeline": pipeline_name,
            "average_faithfulness": round(avg, 4),
            "per_query": results,
        },
        f,
        indent=2,
    )

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query_id",
            "question",
            "faithfulness",
            "supported_claims",
            "total_claims",
        ],
    )

    writer.writeheader()

    for row in results:
        writer.writerow(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "faithfulness": row["faithfulness"],
                "supported_claims": row["supported_claims"],
                "total_claims": row["total_claims"],
            }
        )

print(f"Saved JSON to {output_file}")
print(f"Saved CSV to {csv_file}")