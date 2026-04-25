"""
POST /api/query

Runs the full RAG pipeline (query rewrite → retrieval → rerank → generation)
and returns the answer with source citations.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models import QueryRequest, QueryResponse, SourceChunk
from app.services.rag_pipeline import run_rag

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse, #pydantic  model
    summary="Ask a question against ingested documents",
)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    """
    Ask a natural-language question.

    The pipeline will:
    1. Rewrite/expand your question for optimal retrieval.
    2. Fetch the top-k semantically relevant chunks from FAISS.
    3. Re-rank them with a Cross-Encoder.
    4. Feed the top results + question to Gemini 2.5 Flash via OpenRouter.

    Optionally supply `document_id` to restrict answers to a single document.
    """
    try:
        result = run_rag(
            question=payload.question,
            document_id=payload.document_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("RAG pipeline error for question: %s", payload.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    sources = [SourceChunk(**s) for s in result["sources"]]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        rewritten_query=result["rewritten_query"],
        document_id=payload.document_id,
    )