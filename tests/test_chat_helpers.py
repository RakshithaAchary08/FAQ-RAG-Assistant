import pytest

from chat_helpers import (
    activate_conversation,
    build_fallback_suggestions,
    build_related_questions,
    cluster_unanswered_questions,
    clean_conversation_title,
    environment_flag,
    get_or_create_conversation,
    is_admin_page_enabled,
    rename_new_chat_from_question,
    sidebar_title,
)


def test_environment_flag_defaults_off_and_reads_enabled_values(monkeypatch):
    monkeypatch.delenv("SHOW_FEEDBACK", raising=False)
    assert environment_flag("SHOW_FEEDBACK") is False
    assert environment_flag("SHOW_FEEDBACK", default=True) is True

    monkeypatch.setenv("SHOW_FEEDBACK", "true")
    assert environment_flag("SHOW_FEEDBACK") is True
    monkeypatch.setenv("SHOW_FEEDBACK", "false")
    assert environment_flag("SHOW_FEEDBACK") is False


def test_admin_page_requires_a_non_empty_password():
    assert is_admin_page_enabled(None) is False
    assert is_admin_page_enabled("") is False
    assert is_admin_page_enabled("configured-password") is True


def test_fallback_suggestions_use_closest_matches_then_defaults():
    suggestions = build_fallback_suggestions(
        [{"question": "Closest FAQ?"}],
        ["Default one?", "Default two?"],
    )

    assert suggestions == ["Closest FAQ?", "Default one?", "Default two?"]


def test_fallback_suggestions_use_defaults_when_no_matches():
    suggestions = build_fallback_suggestions(
        [],
        ["Default one?", "Default two?", "Default three?"],
    )

    assert suggestions == ["Default one?", "Default two?", "Default three?"]


def test_related_questions_exclude_used_faqs_and_respect_distance_and_limit():
    matches = [
        {"id": 4, "question": "Used FAQ?", "distance": 0.1},
        {"id": 5, "question": "Related one?", "distance": 0.72},
        {"id": 6, "question": "Related two?", "distance": 1.0},
        {"id": 7, "question": "Too far?", "distance": 1.081},
        {"id": 8, "question": "Related three?", "distance": 1.05},
    ]

    assert build_related_questions(matches, [4], 0.72) == [
        "Related one?",
        "Related two?",
        "Related three?",
    ]


def test_unanswered_questions_cluster_greedily_by_cosine_similarity():
    rows = [
        {"id": 1, "question": "First?"},
        {"id": 2, "question": "Similar?"},
        {"id": 3, "question": "Different?"},
    ]

    clusters = cluster_unanswered_questions(
        rows,
        [[1, 0], [0.99, 0.01], [0, 1]],
        threshold=0.8,
    )

    assert [cluster["count"] for cluster in clusters] == [2, 1]
    assert clusters[0]["rows"] == rows[:2]
    assert clusters[0]["example_questions"] == ["First?", "Similar?"]
    assert clusters[1]["rows"] == rows[2:]


def test_unanswered_question_clustering_rejects_invalid_threshold_and_input():
    with pytest.raises(ValueError, match="threshold must be 0-1"):
        cluster_unanswered_questions([], [], threshold=1.1)
    with pytest.raises(ValueError, match="one embedding"):
        cluster_unanswered_questions([{"id": 1, "question": "Question?"}], [], 0.8)


def test_small_talk_conversation_title_is_updated_from_first_real_question():
    created_titles = []
    renamed_titles = []
    state = {"current_conversation_id": None}

    def create_conversation(title):
        created_titles.append(title)
        return 21

    conversation_id = get_or_create_conversation(
        state,
        create_conversation,
    )
    assert conversation_id == 21
    assert created_titles == ["New chat"]
    assert state["_conversation_needs_title"] is True

    assert (
        rename_new_chat_from_question(
            state,
            conversation_id,
            "Thanks",
            True,
            lambda chat_id, title, is_custom: renamed_titles.append(
                (chat_id, title, is_custom)
            ) or True,
        )
        is False
    )
    assert renamed_titles == []

    renamed = rename_new_chat_from_question(
        state,
        conversation_id,
        "  How do I reset my password?  ",
        False,
        lambda chat_id, title, is_custom: renamed_titles.append(
            (chat_id, title, is_custom)
        ) or True,
    )
    assert renamed is True
    assert renamed_titles == [(21, "How do I reset my password?", False)]
    assert state["_conversation_needs_title"] is False
    assert rename_new_chat_from_question(
        state,
        conversation_id,
        "A second FAQ question?",
        False,
        lambda chat_id, title, is_custom: renamed_titles.append(
            (chat_id, title, is_custom)
        ) or True,
    ) is False
    assert renamed_titles == [(21, "How do I reset my password?", False)]
    assert len(clean_conversation_title("x" * 80)) == 40


def test_fallback_does_not_rename_and_custom_title_is_preserved():
    rename_calls = []
    state = {"_conversation_needs_title": True}
    rename = lambda *args: rename_calls.append(args) or True

    assert rename_new_chat_from_question(
        state,
        3,
        "Out-of-scope question?",
        is_small_talk=False,
        rename=rename,
        has_confident_answer=False,
    ) is False
    assert rename_new_chat_from_question(
        state,
        3,
        "A real FAQ question?",
        is_small_talk=False,
        rename=rename,
        title_is_custom=True,
    ) is False
    assert state["_conversation_needs_title"] is True
    assert rename_calls == []


def test_suggested_question_and_sidebar_reruns_reuse_conversation_id():
    created_titles = []
    state = {"current_conversation_id": None}

    def create_conversation(title):
        created_titles.append(title)
        return len(created_titles)

    # A fallback suggested question is submitted on a later Streamlit rerun.
    first_id = get_or_create_conversation(
        state,
        create_conversation,
    )
    suggested_question_id = get_or_create_conversation(
        state,
        create_conversation,
    )
    assert suggested_question_id == first_id
    assert created_titles == ["New chat"]

    # Selecting a saved sidebar conversation changes the active ID, not creation.
    activate_conversation(state, 42)
    selected_chat_id = get_or_create_conversation(
        state,
        create_conversation,
    )
    rerun_chat_id = get_or_create_conversation(
        state,
        create_conversation,
    )
    assert selected_chat_id == rerun_chat_id == 42
    assert created_titles == ["New chat"]


def test_sidebar_title_uses_ellipsis_for_long_names():
    assert sidebar_title("A short title") == "A short title"
    assert sidebar_title("a" * 40, max_length=10) == "a" * 9 + "…"
