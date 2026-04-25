"""
Advanced RAG pipeline.

Stages
------
1. Query Optimisation  — LLM rewrites the raw user question for better retrieval.
2. FAISS Retrieval     — Fetch `retrieval_top_k` candidate chunks.
3. Cross-Encoder Re-rank — Score & filter down to `reranker_top_k` chunks.
4. Generation          — Gemini 2.5 Flash (via OpenRouter) synthesises the answer.

All LLM calls are wrapped in tenacity exponential-backoff retry.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import get_settings
from app.services.document_processor import load_vector_store
from app.services.reranker import rerank

logger = logging.getLogger(__name__)


# ── LLM singleton ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    """
    Gemini 2.5 Flash routed through OpenRouter.
    OpenRouter exposes an OpenAI-compatible endpoint, so we use ChatOpenAI
    with a custom base_url and the openrouter API key.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=2048,
        default_headers={
            # OpenRouter strongly recommends these headers
            "HTTP-Referer": "https://my-learning-ai-assistant.local",
            "X-Title": "My Learning AI Assistant",
        },
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert query optimiser for a document retrieval system. "
                "Your task is to rewrite the user's question to be more precise and "
                "retrieval-friendly, using technical language where appropriate. "
                "Return ONLY the rewritten query — no explanation, no preamble."
            ),
        ),
        ("human", "Original question: {question}"),
    ]
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful and knowledgeable AI learning assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the context does not contain enough information, say so honestly. "
                "Cite relevant page numbers when possible. "
                "Be thorough, structured, and educational in your response."
            ),
        ),
        (
            "human",
            (
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        ),
    ]
)


# ── Retry-wrapped LLM invoke ──────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status and (status == 429 or status >= 500):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


@retry(
    retry=lambda rs: _is_retryable(rs.outcome.exception()) if rs.outcome.failed else False,
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm(llm: ChatOpenAI, prompt: ChatPromptTemplate, **kwargs) -> str:
    chain = prompt | llm
    response = chain.invoke(kwargs)
    return response.content


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _optimise_query(question: str) -> str:
    """Stage 1: Rewrite the question for better vector-space retrieval."""
    llm = _get_llm()
    try:
        rewritten = _invoke_llm(llm, _REWRITE_PROMPT, question=question)
        logger.info("Query rewrite: '%s' → '%s'", question, rewritten)
        return rewritten.strip()
    except Exception as exc:
        logger.warning("Query rewrite failed (%s), using original question.", exc)
        return question


def _retrieve(query: str, document_id: str | None) -> list[Document]:
    """Stage 2: FAISS similarity search, optionally filtered by document_id."""
    settings = get_settings()
    vector_store = load_vector_store()

    if document_id:
        # LangChain's FAISS filter= parameter is unreliable (returns far fewer
        # results than the matching chunks). Fetch the full index and filter manually.
        total = len(vector_store.docstore._dict)
        all_results = vector_store.similarity_search(query, k=total)
        docs = [
            d for d in all_results
            if d.metadata.get("document_id") == document_id
        ][:settings.retrieval_top_k]
    else:
        docs = vector_store.similarity_search(query, k=settings.retrieval_top_k)

    logger.debug("Retrieved %d raw candidates.", len(docs))
    return docs


def _build_context(docs: list[Document]) -> str:
    """Format reranked docs into a structured context block for the LLM."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score", "n/a")
        parts.append(
            f"[Chunk {i} | Page {page} | Relevance {score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ── Public entry point ────────────────────────────────────────────────────────

def run_rag(
    question: str,
    document_id: str | None = None,
) -> dict:
    """
    Execute the full RAG pipeline and return a result dict containing:
    - answer         : LLM-generated answer
    - rewritten_query: query after optimisation
    - sources        : list of SourceChunk-compatible dicts
    """
    # Stage 1 — query optimisation
    rewritten_query = _optimise_query(question)

    # Stage 2 — retrieval
    candidates = _retrieve(rewritten_query, document_id)
    if not candidates:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "rewritten_query": rewritten_query,
            "sources": [],
        }

    # Stage 3 — re-ranking
    top_docs = rerank(rewritten_query, candidates) # response from the re-ranking

    # Stage 4 — generation
    context = _build_context(top_docs)
    llm = _get_llm()
    answer = _invoke_llm(llm, _RAG_PROMPT, context=context, question=rewritten_query)

    sources = [
        {
            "content": doc.page_content[:400],  # truncate for payload size
            "page": doc.metadata.get("page"),
            "score": doc.metadata.get("rerank_score"),
        }
        for doc in top_docs
    ]

    return {
        "answer": answer.strip(),
        "rewritten_query": rewritten_query,
        "sources": sources,
    }