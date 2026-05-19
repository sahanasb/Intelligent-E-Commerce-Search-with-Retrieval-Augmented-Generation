from ast import Store
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEndpoint
from langchain_cohere import CohereRerank
from Reranker import BGEReranker

# ✅ ADD THIS (Chroma import)
from langchain_community.vectorstores import Chroma


SYSTEM = """You are a professional product search assistant.
Help customers find the best products based on their needs.
Use the provided product catalog to recommend relevant items.
If you find matching products, provide detailed recommendations with key features.
If no products match the query, politely suggest browsing our catalog or refining the search.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Customer Query:\n{input}\n\n"
     "Product Catalog:\n{context}\n\n"
     "Please provide professional product recommendations with key features and benefits.")
])

REDIS_URL = os.getenv("REDIS_URL")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if REDIS_URL:
    try:
        set_llm_cache(
            RedisSemanticCache(
                redis_url=REDIS_URL,
                embeddings=embeddings,
                distance_threshold=0.98
            )
        )
        print("Redis cache enabled")
    except Exception as e:
        print(f"Redis cache setup failed: {e}")


# ✅ MINIMAL FIX: load your existing Chroma DB
def get_vector_store():
    return Chroma(
        persist_directory="./product_db",
        embedding_function=embeddings
    )

async def _build_chain():
    store = get_vector_store()

    # ✅ retriever from your persisted vector DB
    base_retriever = store.as_retriever(
        search_kwargs={"k": 5}
    )
    reranker = BGEReranker()

    class CustomRetriever:
     def __init__(self, base_retriever, reranker):
        self.base_retriever = base_retriever
        self.reranker = reranker

     def get_relevant_documents(self, query):
        docs = self.base_retriever.get_relevant_documents(query)
        return self.reranker.rank(query, docs, top_n=3)
    # compressor = CohereRerank(
    #     top_n=3,
    #     model="rerank-multilingual-v3.0",
    # ) 
    retriever = CustomRetriever(base_retriever, reranker)
    # retriever = ContextualCompressionRetriever(
    #     base_retriever=base_retriever,
    #     base_compressor=compressor,
    # )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
        max_new_tokens=512,
    )

    doc_chain = create_stuff_documents_chain(
        llm,
        PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever,
        doc_chain
    )

    return rag_chain


async def search_products_async(question: str, category_filter: str = None):
    chain = await _build_chain()

    # ✅ hardcoded query as requested
    result = await chain.ainvoke({
        "input": "i always want to check time, what should i buy?"
    })

    answer: str = result["answer"]
    docs = result["context"]

    product_ids = [
        doc.metadata.get("product_id")
        for doc in docs
        if doc.metadata.get("product_id")
    ]

    products = []

    if product_ids:
        chroma_result = Store.collection.get(ids=product_ids)

        ids = chroma_result.get("ids", [])
        metadatas = chroma_result.get("metadatas", [])

        for product_id, metadata in zip(ids, metadatas):

            if category_filter:
                if metadata.get("category") != category_filter:
                    continue

            products.append({
                "id": product_id,
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "price": metadata.get("price"),
                "category": metadata.get("category"),
                "image_url": metadata.get("image_url"),
            })

    contexts = [d.page_content for d in docs]

    return answer, products, contexts

import asyncio

if __name__ == "__main__":
    answer, products, contexts = asyncio.run(
        search_products_async("dummy")  # input is ignored in your current code
    )

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== PRODUCTS ===")
    for p in products:
        print(p)

    print("\n=== CONTEXTS ===")
    for c in contexts:
        print(c)