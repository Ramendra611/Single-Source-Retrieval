"""
RAGAS-based evaluation of RAG outputs.

Metrics computed
----------------
- faithfulness       : Is the answer grounded in the retrieved context?
- answer_relevancy   : How relevant is the answer to the question?
- context_precision  : Are the retrieved contexts precise and on-topic?
- context_recall     : (Requires ground_truth) Did we retrieve what was needed?

Results are logged to both stdout (structlog) and a JSONL file for persistence.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


def _ensure_log_dir(log_path: str) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_rag(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    """
    Run RAGAS evaluation and return a metrics dict.

    Parameters
    ----------
    question     : The original user question.
    answer       : The LLM-generated answer.
    contexts     : List of retrieved context strings used for generation.
    ground_truth : Optional reference answer (enables context_recall).

    Returns
    -------
    Dict with metric names → float scores (None if metric could not be computed).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        metrics_to_run = [faithfulness, answer_relevancy, context_precision]

        # context_recall requires ground_truth
        if ground_truth:
            from ragas.metrics import context_recall
            metrics_to_run.append(context_recall)

        # Build a single-row HuggingFace Dataset (ragas input format)
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        settings = get_settings()

        # RAGAS uses LangChain under the hood; point it at our LLM + local embeddings
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        ragas_llm = ChatOpenAI(
            model=settings.chat_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0,
        )
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        result = evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
        

        scores = result.to_pandas().iloc[0].to_dict()

        metrics = {
            "faithfulness": _safe_float(scores.get("faithfulness")),
            "answer_relevancy": _safe_float(scores.get("answer_relevancy")),
            "context_precision": _safe_float(scores.get("context_precision")),
            "context_recall": _safe_float(scores.get("context_recall")),
        }

    except Exception as exc:
        logger.warning("RAGAS evaluation failed", error=str(exc))
        metrics = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    _log_metrics(question, answer, metrics)
    return metrics


def _safe_float(value) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _log_metrics(question: str, answer: str, metrics: dict) -> None:
    """Append evaluation record to the JSONL log file and emit to stdout."""
    settings = get_settings()
    log_path = _ensure_log_dir(settings.eval_log_path)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question[:200],
        "answer_preview": answer[:200],
        "metrics": metrics,
    }

    # Structured log to stdout
    logger.info("ragas_eval", **record)

    # Persist to JSONL file
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")