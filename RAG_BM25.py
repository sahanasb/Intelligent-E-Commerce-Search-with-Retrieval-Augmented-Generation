### RAG With just BM25  Retriever 
import os
import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.globals import set_llm_cache
from langchain_community.vectorstores import Chroma

try:
    from langchain_redis import RedisSemanticCache
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from Reranker import BGEReranker 

# Prompt

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

# Embeddings 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Redis semantic cache (optional)

REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL and _REDIS_AVAILABLE:
    try:
        set_llm_cache(
            RedisSemanticCache(
                redis_url=REDIS_URL,
                embeddings=embeddings,
                distance_threshold=0.98,
            )
        )
        print("Redis semantic cache enabled.")
    except Exception as exc:
        print(f"Redis cache setup failed (continuing without cache): {exc}")

# Vector store helper

def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory="./product_db",
        embedding_function=embeddings,
    )

# Chain builder 

_rag_chain = None

def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page_content into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

all_docs: list[Document] = get_vector_store().similarity_search("", k=1000)  # pull all docs

bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 5 

def _build_chain():
    """Build (and cache) the RAG chain using LCEL."""
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    store = get_vector_store()
    base_retriever = store.as_retriever(search_kwargs={"k": 5})
    reranker = BGEReranker()
    retriever = bm25_retriever

    load_dotenv()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=512,
    )

    # LCEL pipeline — replaces the removed create_retrieval_chain /
    # create_stuff_documents_chain helpers from langchain.chains.
    #
    # Input:  {"input": "<user question>"}
    # Output: {"input": ..., "context": [Document, ...], "answer": "..."}
    _rag_chain = (
        RunnablePassthrough.assign(
            # 1. Retrieve + rerank; keep the raw Document list in "context"
            context=RunnableLambda(lambda x: retriever.invoke(x["input"]))
        )
        | RunnablePassthrough.assign(
            # 2. Format docs into a plain-text string for the prompt,
            #    then call the LLM and parse its output as a string.
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

# Public search function

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

    # Collect product IDs present in the retrieved docs
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



# CLI entry-point

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