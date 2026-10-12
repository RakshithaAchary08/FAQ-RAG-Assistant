import pytest

import rag


def test_groq_client_uses_bounded_timeout_and_sdk_retries(monkeypatch):
    calls = []

    class FakeGroq:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(rag, "Groq", FakeGroq)

    rag._new_groq_client("test-key")

    assert calls == [
        {"api_key": "test-key", "max_retries": 3, "timeout": 30.0}
    ]


def test_relevance_cutoff_uses_nearest_vector_result():
    rows = [(1, "question", "answer", 0.2), (2, "question 2", "answer 2", 0.8)]

    assert rag._passes_relevance_cutoff(rows, 0.72)
    assert not rag._passes_relevance_cutoff([(1, "question", "answer", 0.73)], 0.72)
    assert not rag._passes_relevance_cutoff([], 0.72)


@pytest.mark.parametrize(
    ("distance", "max_distance", "expected"),
    [
        (0.16, 0.72, 0.7777777777777778),
        (0.8, 0.72, 0.0),
        (0.0, 0.0, 1.0),
        (0.1, 0.0, 0.0),
    ],
)
def test_faq_similarity_is_normalized_and_clamped(
    distance,
    max_distance,
    expected,
):
    assert rag.faq_similarity(distance, max_distance) == pytest.approx(expected)


def test_closest_faqs_keeps_candidates_above_answer_cutoff(monkeypatch):
    import database

    class FakeEmbeddingModel:
        def query_embed(self, question):
            return iter([[0.0]])

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def close(self):
            pass

        def execute(self, query, parameters):
            assert "distance <=" not in query
            assert parameters[1] == 3
            return self

        def fetchall(self):
            return [(8, "Closest FAQ?", 0.91)]

    monkeypatch.setattr(database, "ensure_index", lambda: None)
    monkeypatch.setattr(rag, "get_embedding_model", FakeEmbeddingModel)
    monkeypatch.setattr(rag, "get_connection", FakeConnection)
    monkeypatch.setattr(rag.sqlite_vec, "serialize_float32", lambda vector: b"query")

    matches = rag.closest_faqs("out of scope", top_k=3)

    assert matches == [{"id": 8, "question": "Closest FAQ?", "distance": 0.91}]
