"""Small pure helpers for chat titles, fallback suggestions, and session state."""

import os
from collections.abc import Callable, MutableMapping, Sequence
from math import sqrt
from typing import Any


def environment_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment setting using common true-value spellings."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_admin_page_enabled(admin_password: str | None) -> bool:
    """Return whether a non-empty admin password enables the Admin page."""
    return bool(admin_password)


def clean_conversation_title(question: str) -> str:
    """Create a short, readable title from the first non-small-talk message."""
    cleaned = "".join(
        character for character in question if character.isprintable()
    ).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned[:40].rstrip() or "New chat"


def build_fallback_suggestions(
    close_matches: Sequence[dict[str, object]],
    defaults: Sequence[str],
    limit: int = 3,
) -> list[str]:
    """Choose up to three closest FAQ questions, filling from default prompts."""
    suggestions: list[str] = []
    for match in close_matches:
        question = match.get("question")
        if isinstance(question, str) and question and question not in suggestions:
            suggestions.append(question)
        if len(suggestions) == limit:
            break

    for question in defaults:
        if len(suggestions) >= limit:
            break
        if question not in suggestions:
            suggestions.append(question)
    return suggestions[:limit]


def build_related_questions(
    close_matches: Sequence[dict[str, object]],
    excluded_faq_ids: Sequence[int],
    max_distance: float,
    limit: int = 3,
) -> list[str]:
    """Return the nearest eligible FAQ questions within the related cutoff."""
    if limit < 1:
        return []

    cutoff = max_distance * 1.5
    excluded = set(excluded_faq_ids)
    related: list[str] = []
    for match in close_matches:
        question = match.get("question")
        distance = match.get("distance")
        if (
            match.get("id") in excluded
            or not isinstance(question, str)
            or not question
            or not isinstance(distance, (int, float))
            or distance > cutoff
            or question in related
        ):
            continue
        related.append(question)
        if len(related) == limit:
            break
    return related


def cluster_unanswered_questions(
    rows: Sequence[dict[str, object]],
    embeddings: Sequence[Sequence[float]],
    threshold: float,
) -> list[dict[str, object]]:
    """Greedily group unanswered rows by cosine similarity to cluster centroids."""
    if not 0 <= threshold <= 1:
        raise ValueError("The unanswered-question cluster threshold must be 0-1.")
    if len(rows) != len(embeddings):
        raise ValueError("Each unanswered question must have one embedding.")

    clusters: list[tuple[list[dict[str, object]], list[float]]] = []
    for row, embedding in zip(rows, embeddings, strict=True):
        vector = [float(value) for value in embedding]
        best_cluster_index: int | None = None
        best_similarity = threshold
        for index, (_, centroid) in enumerate(clusters):
            similarity = _cosine_similarity(vector, centroid)
            if similarity >= best_similarity:
                best_cluster_index = index
                best_similarity = similarity

        if best_cluster_index is None:
            clusters.append(([row], vector))
            continue

        cluster_rows, cluster_centroid = clusters[best_cluster_index]
        if len(vector) != len(cluster_centroid):
            raise ValueError("Unanswered question embeddings must have equal sizes.")
        cluster_rows.append(row)
        count = len(cluster_rows)
        clusters[best_cluster_index] = (
            cluster_rows,
            [
                (old_value * (count - 1) + new_value) / count
                for old_value, new_value in zip(
                    cluster_centroid,
                    vector,
                    strict=True,
                )
            ],
        )

    return [
        {
            "rows": cluster_rows,
            "count": len(cluster_rows),
            "example_questions": [
                row["question"] for row in cluster_rows[:3]
            ],
        }
        for cluster_rows, _ in clusters
    ]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity, treating zero vectors as unrelated."""
    if len(left) != len(right):
        raise ValueError("Unanswered question embeddings must have equal sizes.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


def get_or_create_conversation(
    state: MutableMapping[str, Any],
    create: Callable[[str], int],
) -> int:
    """Create a conversation at most once while an active chat ID is retained."""
    conversation_id = state.get("current_conversation_id")
    if conversation_id is not None:
        return int(conversation_id)

    conversation_id = create("New chat")
    state["current_conversation_id"] = conversation_id
    state["_loaded_conversation_id"] = conversation_id
    state["_conversation_needs_title"] = True
    state["_conversation_title_is_custom"] = False
    return conversation_id


def activate_conversation(
    state: MutableMapping[str, Any],
    conversation_id: int,
) -> None:
    """Select a saved conversation without creating a new database row."""
    state["current_conversation_id"] = conversation_id
    state["_loaded_conversation_id"] = None


def rename_new_chat_from_question(
    state: MutableMapping[str, Any],
    conversation_id: int,
    question: str,
    is_small_talk: bool,
    rename: Callable[[int, str, bool], bool],
    *,
    title_is_custom: bool = False,
    has_confident_answer: bool = True,
) -> bool:
    """Rename a non-custom new chat once its first confident FAQ answer exists."""
    if (
        is_small_talk
        or not has_confident_answer
        or title_is_custom
        or not state.get("_conversation_needs_title", False)
    ):
        return False
    renamed = rename(conversation_id, clean_conversation_title(question), False)
    if renamed:
        state["_conversation_needs_title"] = False
    return renamed


def sidebar_title(title: str, max_length: int = 32) -> str:
    """Truncate a saved-chat title for the sidebar button label."""
    if len(title) <= max_length:
        return title
    return f"{title[:max_length - 1].rstrip()}…"
