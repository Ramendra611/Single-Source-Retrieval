"""
RAGAS-based evaluation of RAG outputs.

Metrics computed
----------------
- faithfulness       : Is the answer grounded in the retrieved context?
- answer_relevancy   : How relevant is the answer to the question?
- context_precision  : Are the retrieved contexts precise and on-topic?
- context_recall     : (Requires ground_truth) Did we retrieve what was needed?

Results are logged to both stdout (structlog) and a JSONL file for persistence.

NOTE: Uses RAGAS 0.2.x API (EvaluationDataset + SingleTurnSample).
"""

import json
import logging
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
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings

        settings = get_settings()

        # Wrap LLM and embeddings for RAGAS 0.2.x
        ragas_llm = LangchainLLMWrapper(ChatOpenAI(
            model=settings.chat_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0,
        ))
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        ))

        # Build metrics list
        metrics_to_run = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
        ]

        if ground_truth:
            from ragas.metrics import ContextRecall
            metrics_to_run.append(ContextRecall(llm=ragas_llm))

        # Build RAGAS 0.2.x dataset
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(dataset=dataset, metrics=metrics_to_run)
        scores = result.to_pandas().iloc[0].to_dict()

        metrics = {
            "faithfulness": _safe_float(scores.get("faithfulness")),
            "answer_relevancy": _safe_float(scores.get("answer_relevancy")),
            "context_precision": _safe_float(scores.get("context_precision")),
            "context_recall": _safe_float(scores.get("context_recall")),
        }

    except Exception as exc:
        logger.warning("RAGAS evaluation failed", error=str(exc), exc_info=True)
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
