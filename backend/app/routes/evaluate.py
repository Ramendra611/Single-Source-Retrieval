"""
POST /api/evaluate

Trigger RAGAS evaluation on a previously generated answer.
Metrics are returned in the response AND persisted to a JSONL log file.
"""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, status

from app.models import EvalMetrics, EvalRequest, EvalResponse
from app.services.evaluator import evaluate_rag

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Evaluate RAG output quality with RAGAS",
)
async def evaluate_output(payload: EvalRequest) -> EvalResponse:
    """
    Run RAGAS evaluation metrics on a question/answer/context triplet.

    - **faithfulness**      — Is the answer supported by the contexts?
    - **answer_relevancy**  — Is the answer relevant to the question?
    - **context_precision** — Are the contexts precise and on-topic?
    - **context_recall**    — (Only when `ground_truth` is provided.)

    Scores are appended to the `EVAL_LOG_PATH` JSONL file for
    offline analysis and observability.
    """
    if not payload.contexts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one context string is required for evaluation.",
        )

    try:
        # Run in a separate thread so RAGAS can create its own event loop
        # without conflicting with FastAPI's uvloop.
        fn = partial(
            evaluate_rag,
            question=payload.question,
            answer=payload.answer,
            contexts=payload.contexts,
            ground_truth=payload.ground_truth,
        )
        metrics_dict = await asyncio.to_thread(fn)
    except Exception as exc:
        logger.exception("Evaluation endpoint error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {exc}",
        )

    return EvalResponse(
        metrics=EvalMetrics(**metrics_dict),
        logged=True,
    )