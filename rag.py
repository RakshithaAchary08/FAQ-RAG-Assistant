import logging
import os
import re
from contextlib import closing
from functools import lru_cache
from typing import Any

import sqlite_vec
from groq import Groq

from database import get_connection, get_embedding_model

DEFAULT_TOP_K = 3
RERANK_CANDIDATES = 10
DEFAULT_MAX_DISTANCE = 0.72
RRF_K = 60
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
logger = logging.getLogger(__name__)


def _new_groq_client(api_key: str) -> Groq:
    """Build a Groq client with bounded timeout and transient-error retries."""
    return Groq(api_key=api_key, max_retries=3, timeout=30.0)


def rewrite_follow_up_query(
    question: str,
    recent_messages: list[dict[str, str]],
    api_key: str | None,
    model: str,
) -> str | None:
    """Rewrite a contextual follow-up into a standalone FAQ search question."""
    if not api_key:
        return None

    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in recent_messages[-4:]
    )
    try:
        response = _new_groq_client(api_key).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the customer's latest message as one standalone "
                        "question for FAQ search. Use the recent conversation only "
                        "to resolve references. Return only the rewritten question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation:\n{conversation}\n\n"
                        f"Latest message:\n{question}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content
        return rewritten.strip() if rewritten and rewritten.strip() else None
    except Exception as error:
        logger.warning(
            "Could not rewrite follow-up query; using conversation-context search.",
            extra={
                "event": "groq_query_rewrite_failed",
                "error_type": type(error).__name__,
            },
            exc_info=True,
        )
        return None


def _fts_query(question: str) -> str:
    """Build a safe FTS5 OR query from the search terms."""
    terms = re.findall(r"\w+", question, flags=re.UNICODE)
    return " OR ".join(f'"{term}"' for term in terms)


def _merge_rankings(
    vector_rows: list[tuple[Any, ...]],
    keyword_rows: list[tuple[Any, ...]],
    top_k: int,
) -> list[int]:
    """Return FAQ ids ordered by Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for rows in (vector_rows, keyword_rows):
        for rank, row in enumerate(rows, start=1):
            faq_id = int(row[0])
            scores[faq_id] = scores.get(faq_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]


def _passes_relevance_cutoff(
    vector_rows: list[tuple[Any, ...]],
    max_distance: float,
) -> bool:
    """Check the nearest vector match against the configured relevance cutoff."""
    return bool(vector_rows) and vector_rows[0][3] <= max_distance


def faq_similarity(distance: float, max_distance: float | None = None) -> float:
    """Return the normalized FAQ similarity as a fraction between zero and one."""
    if max_distance is None:
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", DEFAULT_MAX_DISTANCE))
    if max_distance <= 0:
        return 1.0 if distance <= 0 else 0.0
    return max(0.0, 1.0 - distance / max_distance)


def closest_faqs(
    question: str,
    top_k: int | None = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    """Return nearest FAQ candidates without applying the answer relevance cutoff."""
    if not question.strip() or (top_k is not None and top_k < 1):
        return []

    from database import ensure_index

    ensure_index()
    query_embedding = next(get_embedding_model().query_embed(question))
    query_blob = sqlite_vec.serialize_float32(query_embedding)
    with closing(get_connection()) as connection:
        if top_k is None:
            count_row = connection.execute(
                "SELECT COUNT(*) FROM faqs"
            ).fetchone()
            top_k = int(count_row[0]) if count_row else 0
        if top_k == 0:
            return []
        rows = connection.execute(
            """
            SELECT faqs.id, faqs.question, distance
            FROM faq_vectors
            JOIN faqs ON faqs.id = faq_vectors.rowid
            WHERE faq_vectors.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (query_blob, top_k),
        ).fetchall()
    return [
        {"id": faq_id, "question": faq_question, "distance": distance}
        for faq_id, faq_question, distance in rows
    ]


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    """Load and cache the FastEmbed cross-encoder reranker."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANKER_MODEL)


def _rerank_results(
    question: str,
    results: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    """Rerank candidate FAQs by cross-encoder score, preserving order on failure."""
    try:
        documents = [
            f"Question: {result['question']}\nAnswer: {result['answer']}"
            for result in results
        ]
        scores = list(_get_reranker().rerank(question, documents))
        ranked = sorted(
            zip(scores, results, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        return [result for _, result in ranked[:top_k]]
    except Exception:
        logger.warning(
            "Could not rerank FAQ results; using retrieval ranking.",
            exc_info=True,
        )
        return results[:top_k]


def search_faq(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, object]]:
    if not question.strip():
        return []

    from database import ensure_index

    ensure_index()

    query_embedding = next(get_embedding_model().query_embed(question))
    query_blob = sqlite_vec.serialize_float32(query_embedding)
    max_distance = float(os.getenv("RAG_MAX_DISTANCE", DEFAULT_MAX_DISTANCE))
    use_hybrid = os.getenv("RAG_USE_HYBRID", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_rerank = os.getenv("RAG_USE_RERANK", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    candidate_k = RERANK_CANDIDATES if use_rerank else top_k

    with closing(get_connection()) as connection:
        vector_rows = connection.execute(
            """
            SELECT faqs.id, faqs.question, faqs.answer, distance
            FROM faq_vectors
            JOIN faqs ON faqs.id = faq_vectors.rowid
            WHERE faq_vectors.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (query_blob, candidate_k),
        ).fetchall()

        if not _passes_relevance_cutoff(vector_rows, max_distance):
            return []

        if not use_hybrid:
            vector_results = [
                {
                    "id": faq_id,
                    "question": faq_question,
                    "answer": answer,
                    "distance": distance,
                }
                for faq_id, faq_question, answer, distance in vector_rows
                if distance <= max_distance
            ]
            if use_rerank:
                return _rerank_results(question, vector_results, top_k)
            return vector_results

        fts_query = _fts_query(question)
        keyword_rows = (
            connection.execute(
                """
                SELECT rowid, bm25(faq_fts)
                FROM faq_fts
                WHERE faq_fts MATCH ?
                ORDER BY bm25(faq_fts)
                LIMIT ?
                """,
                (fts_query, candidate_k),
            ).fetchall()
            if fts_query
            else []
        )
        faq_ids = _merge_rankings(vector_rows, keyword_rows, candidate_k)
        placeholders = ", ".join("?" for _ in faq_ids)
        ranked_rows = connection.execute(
            f"""
            SELECT faqs.id, faqs.question, faqs.answer,
                   vec_distance_L2(faq_vectors.embedding, ?) AS distance
            FROM faqs
            JOIN faq_vectors ON faq_vectors.rowid = faqs.id
            WHERE faqs.id IN ({placeholders})
            """,
            (query_blob, *faq_ids),
        ).fetchall()
        results_by_id = {
            faq_id: {
                "id": faq_id,
                "question": faq_question,
                "answer": answer,
                "distance": distance,
            }
            for faq_id, faq_question, answer, distance in ranked_rows
        }

    results = [
        results_by_id[faq_id] for faq_id in faq_ids if faq_id in results_by_id
    ]
    if use_rerank:
        return _rerank_results(question, results, top_k)
    return results[:top_k]