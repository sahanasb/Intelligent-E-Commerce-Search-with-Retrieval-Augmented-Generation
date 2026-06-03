## RAG With SBERT + BGE + Redis
import os
import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
import json
import hashlib
from redis import Redis
from langchain_redis import RedisVectorStore as LangchainRedisVectorStore
# NOTE: SemanticCache import kept in case you want it for other use, but
# set_llm_cache is intentionally NOT called — see explanation below.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# set_llm_cache import intentionally removed — using it with RedisSemanticCache
# was causing every query to receive the same LLM answer because the prompt
# template is structurally identical across queries (same system message + same
# formatting), so the global cache matched on prompt similarity and bypassed the
# LLM entirely, ignoring the different context/documents that were retrieved.
from langchain_chroma import Chroma

try:
    from langchain_redis import RedisSemanticCache
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from Reranker import BGEReranker

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are a professional product search assistant.
Help customers find the best products based on their needs.
Use the provided product catalog to recommend relevant items.
If you find matching products, provide detailed recommendations with key features.
If no products match the query, politely suggest browsing our catalog or refining the search.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    (
        "user",
        "Customer Query:\n{input}\n\n"
        "Product Catalog:\n{context}\n\n"
        "Please provide professional product recommendations with key features and benefits.",
    ),
])

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# ---------------------------------------------------------------------------
# Redis connectivity check
# ---------------------------------------------------------------------------

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL:
    try:
        r = Redis.from_url(REDIS_URL)
        r.ping()
        print(f"[Redis] Connected ✅ — {REDIS_URL}")
    except Exception as e:
        print(f"[Redis] Connection FAILED ❌ — {e}")
else:
    print("[Redis] REDIS_URL not set ❌")

# ---------------------------------------------------------------------------
# WHY set_llm_cache IS REMOVED
# ---------------------------------------------------------------------------
# set_llm_cache(RedisSemanticCache(...)) caches at the *LLM call* level.
# LangChain computes the cache key from the fully-rendered prompt string.
# Because our PROMPT template has the same system message and the same
# structural boilerplate for every query, two semantically-similar *questions*
# (e.g. "show me a watch" vs "I need something to tell time") produce prompt
# strings that are very close in embedding space — well above the 0.98
# threshold — even when the retrieved *context* is completely different.
# The result: the cache returns the first-ever LLM answer for almost every
# subsequent query, making it look like the LLM always recommends the same
# products.
#
# The CachedRetriever below already handles caching at the *retrieval* layer
# (documents + reranking), which is the expensive part. Letting the LLM run
# fresh on each distinct (input, context) pair is both correct and cheap.
#
# If you still want LLM-response caching in the future, use an exact-match
# cache (e.g. Redis key = hash of the full rendered prompt) rather than a
# semantic/vector cache — or lower the threshold drastically (e.g. 0.50) and
# accept more cache misses.

# ---------------------------------------------------------------------------
# Redis lazy-init helpers
# ---------------------------------------------------------------------------

_redis_client = None
_redis_context_store = None


def _get_redis_client():
    global _redis_client
    if _redis_client is None and REDIS_URL:
        try:
            _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping()
            print("[Redis] Client connected ✅")
        except Exception as e:
            print(f"[Redis] Client connection failed: {e}")
    return _redis_client


def _get_redis_context_store():
    global _redis_context_store
    if _redis_context_store is None and REDIS_URL:
        try:
            _redis_context_store = LangchainRedisVectorStore(
                redis_url=REDIS_URL,
                embeddings=embeddings,
                index_name="product_context_cache",
            )
            print("[Redis] Context store initialized ✅")
        except Exception as e:
            print(f"[Redis] Context store init failed: {e}")
    return _redis_context_store


# ---------------------------------------------------------------------------
# Cache thresholds
# ---------------------------------------------------------------------------

# Cosine-similarity threshold for the *retrieval* cache.
# Only reuse cached documents when the new query is very close to a past one.
# 0.92 is a reasonable starting point — lower it (e.g. 0.85) if you are
# seeing too many cache misses, raise it if unrelated queries share results.
CONTEXT_CACHE_THRESHOLD = 0.92
CONTEXT_CACHE_TTL = 60 * 60 * 24  # 24 hours


# ---------------------------------------------------------------------------
# CachedRetriever
# ---------------------------------------------------------------------------

class CachedRetriever(BaseRetriever):
    """
    Retrieval flow:
      1. Semantic search in Redis for a similar past query.
      2. HIT  (similarity >= threshold) → return cached Document list.
      3. MISS → query Chroma → BGE-rerank → store in Redis → return.

    Only the *retrieval + reranking* result is cached, not the LLM answer.
    This avoids the stale-answer bug that arises from caching at the LLM layer.
    """

    base_retriever: Any
    reranker: Any

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        redis_store = _get_redis_context_store()
        redis_client = _get_redis_client()

        print(f"[Cache] redis_store={'✅' if redis_store else '❌ None'}")
        print(f"[Cache] redis_client={'✅' if redis_client else '❌ None'}")

        if redis_store and redis_client:
            try:
                # Guard: skip search when the index has no documents yet.
                index_info = redis_client.execute_command(
                    "FT.INFO", "product_context_cache"
                )
                info_dict = dict(zip(index_info[::2], index_info[1::2]))
                doc_count = int(info_dict.get("num_docs", 0))
                print(f"[Cache] Index has {doc_count} docs")

                if doc_count > 0:
                    cached_results = redis_store.similarity_search_with_score(
                        query, k=1
                    )
                    if cached_results:
                        top_doc, score = cached_results[0]
                        # langchain_redis returns cosine *distance* (0 = identical).
                        # Convert to similarity so the threshold feels intuitive.
                        similarity = 1.0 - score
                        print(
                            f"[Cache] Top similarity: {similarity:.3f} "
                            f"(threshold: {CONTEXT_CACHE_THRESHOLD})"
                        )

                        if similarity >= CONTEXT_CACHE_THRESHOLD:
                            cache_key = top_doc.metadata.get("context_cache_key")
                            print(f"[Cache HIT] key={cache_key}")
                            if cache_key:
                                raw = redis_client.get(cache_key)
                                if raw:
                                    return [
                                        Document(
                                            page_content=d["page_content"],
                                            metadata=d["metadata"],
                                        )
                                        for d in json.loads(raw)
                                    ]
                                else:
                                    print("[Cache] Key expired or missing — fetching fresh")
                        else:
                            print(
                                f"[Cache MISS] Similarity {similarity:.3f} "
                                f"below threshold {CONTEXT_CACHE_THRESHOLD}"
                            )
                else:
                    print("[Cache MISS] Index empty — fetching fresh")

            except Exception as e:
                print(f"[Cache] Lookup error (falling through to Chroma): {e}")

        # ── Cache miss: retrieve from Chroma and rerank ──────────────────────
        print(f"[Chroma] Retrieving for: '{query}'")
        docs = self.base_retriever.invoke(query)
        reranked_docs = self.reranker.rank(query, docs, top_n=3)

        # Store the reranked docs so the next similar query gets a cache hit.
        if redis_store and redis_client:
            try:
                cache_key = "ctx:" + hashlib.md5(query.encode()).hexdigest()
                redis_client.setex(
                    cache_key,
                    CONTEXT_CACHE_TTL,
                    json.dumps([
                        {"page_content": d.page_content, "metadata": d.metadata}
                        for d in reranked_docs
                    ]),
                )
                redis_store.add_texts(
                    texts=[query],
                    metadatas=[{"context_cache_key": cache_key}],
                )
                print(f"[Cache] Stored ✅ key={cache_key}")
            except Exception as e:
                print(f"[Cache] Store failed: {e}")

        return reranked_docs


# ---------------------------------------------------------------------------
# Vector store helper
# ---------------------------------------------------------------------------

def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory="./product_db",
        embedding_function=embeddings,
    )


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

_rag_chain = None


def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page_content into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain():
    """Build (and module-level cache) the RAG chain."""
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    store = get_vector_store()
    base_retriever = store.as_retriever(search_kwargs={"k": 5})
    reranker = BGEReranker()
    retriever = CachedRetriever(base_retriever=base_retriever, reranker=reranker)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512,
    )

    # LCEL pipeline
    # Input:  {"input": "<user question>"}
    # Output: {"input": ..., "context": [Document, ...], "answer": "..."}
    _rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retriever.invoke(x["input"]))
        )
        | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(
                    lambda x: PROMPT.invoke(
                        {
                            "input": x["input"],
                            "context": _format_docs(x["context"]),
                        }
                    )
                )
                | llm
                | StrOutputParser()
            )
        )
    )

    return _rag_chain


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

async def search_products_async(
    question: str,
    category_filter: Optional[str] = None,
) -> tuple[str, list[dict], list[str]]:
    """
    Run the RAG pipeline.

    Parameters
    ----------
    question : str
        The customer's natural-language query.
    category_filter : str, optional
        If provided, only return products whose ``category`` metadata field
        matches this value.

    Returns
    -------
    answer : str
        LLM-generated product recommendation.
    products : list[dict]
        Matching product metadata (optionally filtered by category).
    contexts : list[str]
        Raw page-content chunks passed to the LLM as context.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    chain = _build_chain()
    result: dict = await chain.ainvoke({"input": question})

    answer: str = result["answer"]
    docs: list[Document] = result["context"]

    product_ids: list[str] = [
        doc.metadata["product_id"]
        for doc in docs
        if doc.metadata.get("product_id")
    ]

    products: list[dict] = []

    if product_ids:
        store = get_vector_store()
        chroma_result = store._collection.get(ids=product_ids)

        for product_id, metadata in zip(
            chroma_result.get("ids", []),
            chroma_result.get("metadatas", []),
        ):
            if category_filter and metadata.get("category") != category_filter:
                continue

            products.append(
                {
                    "id": product_id,
                    "name": metadata.get("name"),
                    "description": metadata.get("description"),
                    "price": metadata.get("price"),
                    "category": metadata.get("category"),
                    "image_url": metadata.get("image_url"),
                }
            )

    contexts: list[str] = [d.page_content for d in docs]
    return answer, products, contexts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Product search RAG pipeline")
    parser.add_argument(
        "--question",
        default="I always want to check time, what should I buy?",
        help="Customer query",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter (e.g. 'watches', 'electronics')",
    )
    args = parser.parse_args()

    answer, products, contexts = asyncio.run(
        search_products_async(args.question, args.category)
    )

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== PRODUCTS ===")
    if products:
        for p in products:
            print(p)
    else:
        print("No products found (check product_db and category filter).")

    print("\n=== CONTEXTS ===")
    for c in contexts:
        print(c)