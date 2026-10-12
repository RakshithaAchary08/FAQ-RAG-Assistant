import hashlib
import json

import pytest

import database


def test_load_faqs_validates_and_trims_question_and_answer(tmp_path, monkeypatch):
    faq_path = tmp_path / "faqs.json"
    faq_path.write_text(
        json.dumps([{"question": "  Question?  ", "answer": "  Answer.  "}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)

    assert database.load_faqs() == [{"question": "Question?", "answer": "Answer."}]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "non-empty JSON array"),
        ([None], "must be an object"),
        ([{"question": " ", "answer": "Answer"}], "needs a question"),
        ([{"question": "Question", "answer": ""}], "needs an answer"),
    ],
)
def test_load_faqs_rejects_invalid_entries(tmp_path, monkeypatch, payload, error):
    faq_path = tmp_path / "faqs.json"
    faq_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)

    with pytest.raises(ValueError, match=error):
        database.load_faqs()


def test_save_faqs_validates_and_atomically_writes_json(tmp_path, monkeypatch):
    faq_path = tmp_path / "faqs.json"
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)

    saved = database.save_faqs(
        [{"question": "  New question?  ", "answer": "  New answer.  "}]
    )

    assert saved == [{"question": "New question?", "answer": "New answer."}]
    assert json.loads(faq_path.read_text(encoding="utf-8")) == saved
    assert not list(tmp_path.glob("*.tmp"))


def test_admin_feedback_and_unanswered_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_database()
    database.save_feedback("Question?", "Answer.", [1], 1)
    database.save_feedback("Another?", "Answer.", [], -1)
    database.log_unanswered("Unknown question?")

    assert database.get_feedback_stats() == {
        "total": 2,
        "positive": 1,
        "negative": 1,
    }
    unanswered = database.get_unanswered_questions()
    assert len(unanswered) == 1
    assert unanswered[0]["id"] == 1
    assert unanswered[0]["question"] == "Unknown question?"
    assert unanswered[0]["created_at"]


def test_created_faq_ids_are_stable_by_file_order(tmp_path, monkeypatch):
    faq_path = tmp_path / "faqs.json"
    faq_path.write_text(
        json.dumps(
            [
                {"question": "First?", "answer": "First answer."},
                {"question": "Second?", "answer": "Second answer."},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    class FakeEmbeddingModel:
        calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [[0.0] * database.EMBEDDING_DIMENSION for _ in texts]

    model = FakeEmbeddingModel()
    monkeypatch.setattr(database, "get_embedding_model", lambda: model)
    first_summary = database.create_embeddings()
    second_summary = database.create_embeddings()
    with database.closing(database.get_connection()) as connection:
        rows = connection.execute("SELECT id, question FROM faqs ORDER BY id").fetchall()
        stored_hashes = connection.execute(
            "SELECT content_hash FROM faqs ORDER BY id"
        ).fetchall()

    assert rows == [(1, "First?"), (2, "Second?")]
    assert first_summary == {"added": 2, "updated": 0, "removed": 0, "unchanged": 0}
    assert second_summary == {"added": 0, "updated": 0, "removed": 0, "unchanged": 2}
    assert len(model.calls) == 1
    assert all(stored_hash[0] for stored_hash in stored_hashes)


def test_incremental_sync_embeds_changes_and_removes_deleted_faqs(
    tmp_path,
    monkeypatch,
):
    faq_path = tmp_path / "faqs.json"
    original_faqs = [
        {"question": "First?", "answer": "First answer."},
        {"question": "Second?", "answer": "Second answer."},
    ]
    faq_path.write_text(json.dumps(original_faqs), encoding="utf-8")
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    class FakeEmbeddingModel:
        calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [
                [float(len(self.calls))] * database.EMBEDDING_DIMENSION
                for _ in texts
            ]

    model = FakeEmbeddingModel()
    monkeypatch.setattr(database, "get_embedding_model", lambda: model)
    database.create_embeddings()
    original_faqs[0]["answer"] = "Updated first answer."
    original_faqs[1] = {"question": "Third?", "answer": "Third answer."}
    faq_path.write_text(json.dumps(original_faqs), encoding="utf-8")

    summary = database.create_embeddings()

    with database.closing(database.get_connection()) as connection:
        rows = connection.execute(
            "SELECT id, question, answer FROM faqs ORDER BY id"
        ).fetchall()
        vector_ids = connection.execute(
            "SELECT rowid FROM faq_vectors ORDER BY rowid"
        ).fetchall()

    assert summary == {"added": 0, "updated": 2, "removed": 0, "unchanged": 0}
    assert len(model.calls[-1]) == 2
    assert rows == [
        (1, "First?", "Updated first answer."),
        (2, "Third?", "Third answer."),
    ]
    assert vector_ids == [(1,), (2,)]


def test_incremental_sync_deletes_removed_faqs(tmp_path, monkeypatch):
    faq_path = tmp_path / "faqs.json"
    faqs = [
        {"question": "First?", "answer": "First answer."},
        {"question": "Second?", "answer": "Second answer."},
    ]
    faq_path.write_text(json.dumps(faqs), encoding="utf-8")
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    class FakeEmbeddingModel:
        def embed(self, texts):
            return [[0.0] * database.EMBEDDING_DIMENSION for _ in texts]

    monkeypatch.setattr(database, "get_embedding_model", lambda: FakeEmbeddingModel())
    database.create_embeddings()
    faq_path.write_text(json.dumps(faqs[:1]), encoding="utf-8")

    summary = database.create_embeddings()

    with database.closing(database.get_connection()) as connection:
        rows = connection.execute("SELECT id, question FROM faqs").fetchall()
        vector_ids = connection.execute("SELECT rowid FROM faq_vectors").fetchall()

    assert summary == {"added": 0, "updated": 0, "removed": 1, "unchanged": 1}
    assert rows == [(1, "First?")]
    assert vector_ids == [(1,)]


def test_removing_first_faq_reuses_unchanged_embeddings(tmp_path, monkeypatch):
    faq_path = tmp_path / "faqs.json"
    faqs = [
        {"question": "First?", "answer": "First answer."},
        {"question": "Second?", "answer": "Second answer."},
        {"question": "Third?", "answer": "Third answer."},
    ]
    faq_path.write_text(json.dumps(faqs), encoding="utf-8")
    monkeypatch.setattr(database, "FAQ_PATH", faq_path)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    class FakeEmbeddingModel:
        calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [
                [float(index)] * database.EMBEDDING_DIMENSION
                for index, _ in enumerate(texts, start=1)
            ]

    model = FakeEmbeddingModel()
    monkeypatch.setattr(database, "get_embedding_model", lambda: model)
    database.create_embeddings()
    faq_path.write_text(json.dumps(faqs[1:]), encoding="utf-8")

    summary = database.create_embeddings()

    with database.closing(database.get_connection()) as connection:
        rows = connection.execute(
            "SELECT id, question FROM faqs ORDER BY id"
        ).fetchall()

    assert summary == {"added": 0, "updated": 0, "removed": 1, "unchanged": 2}
    assert len(model.calls) == 1
    assert rows == [(1, "Second?"), (2, "Third?")]


def test_conversation_messages_round_trip_sources_and_feedback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_database()
    conversation_id = database.create_conversation("  Help with account  ")
    user_message_id = database.add_message(
        conversation_id,
        "user",
        "How do I reset my password?",
    )
    assistant_message_id = database.add_message(
        conversation_id,
        "assistant",
        "Use the password reset link.",
        sources=[
            {
                "id": 7,
                "question": "How do I reset my password?",
                "distance": 0.12,
            }
        ],
        used_fallback=True,
        suggested_questions=["How do I update my password?", "How do I sign in?"],
        related_questions=["How do I secure my account?"],
    )
    database.save_feedback(
        "How do I reset my password?",
        "Use the password reset link.",
        [7],
        1,
        message_id=assistant_message_id,
    )

    messages = database.get_messages(conversation_id)
    conversations = database.list_conversations()

    assert [message["id"] for message in messages] == [
        user_message_id,
        assistant_message_id,
    ]
    assert messages[1]["sources"][0]["id"] == 7
    assert messages[1]["question"] == "How do I reset my password?"
    assert messages[1]["faq_ids"] == [7]
    assert messages[1]["feedback_rating"] == 1
    assert messages[1]["used_fallback"] is True
    assert messages[1]["suggested_questions"] == [
        "How do I update my password?",
        "How do I sign in?",
    ]
    assert messages[1]["related_questions"] == ["How do I secure my account?"]
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["title"] == "Help with account"

    database.rename_conversation(conversation_id, "Account help")
    assert database.list_conversations()[0]["title"] == "Account help"
    assert database.list_conversations()[0]["title_is_custom"] is True


def test_analytics_queries_return_daily_questions_fallbacks_and_totals(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    conversation_id = database.create_conversation("Analytics chat")
    first_question_id = database.add_message(
        conversation_id,
        "user",
        "How do I sign in?",
    )
    database.add_message(conversation_id, "user", "How do I sign in?")
    fallback_message_id = database.add_message(
        conversation_id,
        "assistant",
        "I could not find an answer.",
        used_fallback=True,
    )
    confident_message_id = database.add_message(
        conversation_id,
        "assistant",
        "Use your account email.",
    )
    database.add_message(conversation_id, "user", "Where is support?")
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "UPDATE messages SET created_at = '2026-10-08 10:00:00' "
                "WHERE id IN (?, ?, ?, ?)",
                (
                    first_question_id,
                    fallback_message_id,
                    confident_message_id,
                    first_question_id + 1,
                ),
            )
            connection.execute(
                "UPDATE messages SET created_at = '2026-10-09 10:00:00' "
                "WHERE id = ?",
                (first_question_id + 4,),
            )
    database.log_unanswered("Unknown question?")

    analytics = database.get_analytics_stats()

    assert analytics["questions_per_day"] == [
        {"date": "2026-10-08", "questions": 2},
        {"date": "2026-10-09", "questions": 1},
    ]
    assert analytics["fallback_rate"] == 0.5
    assert analytics["top_questions"][0] == {
        "question": "How do I sign in?",
        "count": 2,
    }
    assert analytics["total_conversations"] == 1
    assert analytics["total_messages"] == 5
    assert analytics["total_unanswered"] == 1


def test_analytics_queries_handle_an_empty_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    assert database.get_analytics_stats() == {
        "questions_per_day": [],
        "fallback_rate": 0.0,
        "top_questions": [],
        "total_conversations": 0,
        "total_messages": 0,
        "total_unanswered": 0,
    }


def test_delete_unanswered_questions_removes_only_selected_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_database()
    database.log_unanswered("First?")
    database.log_unanswered("Second?")

    assert database.delete_unanswered_questions([1]) == 1
    assert [
        row["question"] for row in database.get_unanswered_questions()
    ] == ["Second?"]


def test_auto_rename_does_not_overwrite_custom_conversation_title(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    conversation_id = database.create_conversation("New chat")

    assert database.rename_conversation(
        conversation_id,
        "User's title",
        is_custom=True,
    )
    assert not database.rename_conversation(
        conversation_id,
        "Automatic title",
        is_custom=False,
    )
    conversation = database.list_conversations()[0]
    assert conversation["title"] == "User's title"
    assert conversation["title_is_custom"] is True


@pytest.mark.parametrize(
    ("title", "error"),
    [
        ("   ", "cannot be empty"),
        ("x" * 41, "cannot exceed 40 characters"),
    ],
)
def test_custom_conversation_rename_validates_title(
    tmp_path,
    monkeypatch,
    title,
    error,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    conversation_id = database.create_conversation("Original title")

    with pytest.raises(ValueError, match=error):
        database.rename_conversation(conversation_id, title)

    assert database.list_conversations()[0]["title"] == "Original title"


def test_custom_conversation_rename_trims_and_sets_custom_flag(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    conversation_id = database.create_conversation("Original title")

    assert database.rename_conversation(conversation_id, "  New title  ")

    conversation = database.list_conversations()[0]
    assert conversation["title"] == "New title"
    assert conversation["title_is_custom"] is True


def test_signup_then_login_uses_normalized_email_and_exact_password(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    user = database.create_user("  Person@Example.com  ", "strong-password")

    assert user["email"] == "person@example.com"
    assert database.verify_login(
        "  PERSON@example.com  ",
        "strong-password",
    ) == user
    assert database.verify_login("person@example.com", "wrong-password") is None
    assert database.verify_login("person@example.com", " strong-password") is None
    with database.closing(database.get_connection()) as connection:
        stored_hash = connection.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone()[0]
    assert stored_hash != "strong-password"
    with pytest.raises(ValueError, match="already exists"):
        database.create_user("person@example.com", "another-password")
    with pytest.raises(ValueError, match="at least 8 characters"):
        database.create_user("short@example.com", "short")
    with pytest.raises(ValueError, match="valid email"):
        database.create_user("not-an-email", "strong-password")


def test_signup_validates_full_name_and_password_confirmation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")

    email, full_name = database.validate_signup(
        " Person@Example.com ",
        "secure-password",
        "secure-password",
        "  Sam Example  ",
    )
    assert (email, full_name) == ("person@example.com", "Sam Example")
    user = database.create_user(email, "secure-password", full_name)
    assert user["full_name"] == "Sam Example"
    assert database.verify_login(email, "secure-password") == user
    assert database.get_user_by_email(email)["full_name"] == "Sam Example"

    with pytest.raises(ValueError, match="Passwords do not match"):
        database.validate_signup(
            email,
            "secure-password",
            "different-password",
            full_name,
        )
    with pytest.raises(ValueError, match="2 and 60"):
        database.create_user("short-name@example.com", "secure-password", "A")
    with pytest.raises(ValueError, match="2 and 60"):
        database.create_user(
            "long-name@example.com",
            "secure-password",
            "N" * 61,
        )


def test_password_reset_tokens_are_single_use_and_expire(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_user("person@example.com", "original-password")

    token = database.create_password_reset_token(" Person@Example.com ")
    assert token is not None
    assert database.create_password_reset_token("unknown@example.com") is None
    assert database.reset_password(token, "replacement-password")
    assert database.verify_login("person@example.com", "replacement-password")
    assert database.verify_login("person@example.com", "original-password") is None
    assert not database.reset_password(token, "another-password")

    expired_token = database.create_password_reset_token("person@example.com")
    assert expired_token is not None
    token_hash = hashlib.sha256(expired_token.encode("utf-8")).hexdigest()
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "UPDATE password_reset_tokens SET expires_at = ? WHERE token_hash = ?",
                ("2000-01-01T00:00:00+00:00", token_hash),
            )
    assert not database.reset_password(expired_token, "another-password")


def test_unknown_and_wrong_password_login_failures_share_generic_message(
    tmp_path,
    monkeypatch,
):
    from auth import AUTH_LOGIN_ERROR_MESSAGE

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_user("person@example.com", "strong-password")

    wrong_password = database.verify_login("person@example.com", "wrong-password")
    unknown_email = database.verify_login("unknown@example.com", "wrong-password")

    assert wrong_password is None
    assert unknown_email is None
    assert AUTH_LOGIN_ERROR_MESSAGE == "Incorrect email or password."


def test_verify_login_locks_after_five_failures_and_resets_after_success(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    user = database.create_user("person@example.com", "strong-password")

    for _ in range(5):
        assert database.verify_login("person@example.com", "wrong-password") is None
    assert database.verify_login("person@example.com", "strong-password") is None

    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "UPDATE users SET locked_until = '2000-01-01T00:00:00+00:00' "
                "WHERE id = ?",
                (user["id"],),
            )

    assert database.verify_login("person@example.com", "strong-password") == user
    with database.closing(database.get_connection()) as connection:
        assert connection.execute(
            "SELECT failed_attempts, locked_until FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone() == (0, None)
        with connection:
            connection.execute(
                "UPDATE users SET failed_attempts = 3 WHERE id = ?",
                (user["id"],),
            )

    assert database.verify_login("person@example.com", "strong-password") == user
    with database.closing(database.get_connection()) as connection:
        assert connection.execute(
            "SELECT failed_attempts, locked_until FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone() == (0, None)


def test_user_conversation_queries_enforce_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    user_a = database.create_user("a@example.com", "password-a")
    user_b = database.create_user("b@example.com", "password-b")
    conversation_id = database.create_conversation("Private chat", int(user_b["id"]))
    database.create_conversation("Legacy chat")
    database.add_message(
        conversation_id,
        "assistant",
        "Private answer.",
        user_id=int(user_b["id"]),
    )

    assert database.list_conversations(user_id=int(user_a["id"])) == []
    assert database.get_messages(
        conversation_id,
        user_id=int(user_a["id"]),
    ) == []
    with pytest.raises(ValueError, match="does not exist"):
        database.rename_conversation(
            conversation_id,
            "Stolen title",
            user_id=int(user_a["id"]),
        )
    with pytest.raises(ValueError, match="does not exist"):
        database.delete_conversation(conversation_id, user_id=int(user_a["id"]))

    assert [
        chat["title"]
        for chat in database.list_conversations(user_id=int(user_b["id"]))
    ] == ["Private chat"]
    assert database.get_messages(
        conversation_id,
        user_id=int(user_b["id"]),
    )[0]["content"] == "Private answer."
    assert database.delete_all_conversations(user_id=int(user_a["id"])) == 0
    assert len(database.list_conversations(user_id=int(user_b["id"]))) == 1
    database.delete_conversation(conversation_id, user_id=int(user_b["id"]))


def test_title_is_custom_migration_is_safe_to_run_twice(tmp_path, monkeypatch):
    database_path = tmp_path / "database" / "faq.db"
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE conversations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "title TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "INSERT INTO conversations (title) VALUES ('Legacy chat')"
            )

    database.create_database()
    database.create_database()

    with database.closing(database.get_connection()) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        ]
        title_is_custom = connection.execute(
            "SELECT title_is_custom FROM conversations WHERE id = 1"
        ).fetchone()
        user_id = connection.execute(
            "SELECT user_id FROM conversations WHERE id = 1"
        ).fetchone()
    assert columns.count("title_is_custom") == 1
    assert columns.count("user_id") == 1
    assert title_is_custom == (0,)
    assert user_id == (None,)


def test_full_name_migration_is_safe_to_run_twice(tmp_path, monkeypatch):
    database_path = tmp_path / "database" / "faq.db"
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE users ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "email TEXT NOT NULL UNIQUE COLLATE NOCASE, "
                "password_hash TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "failed_attempts INTEGER NOT NULL DEFAULT 0, "
                "locked_until TEXT)"
            )
            connection.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                ("legacy@example.com", "already-hashed"),
            )

    database.create_database()
    database.create_database()

    with database.closing(database.get_connection()) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(users)")
        ]
        row = connection.execute(
            "SELECT full_name FROM users WHERE email = ?",
            ("legacy@example.com",),
        ).fetchone()
    assert columns.count("full_name") == 1
    assert row == ("",)


def test_delete_conversation_cascades_messages_and_unlinks_feedback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    database.create_database()
    conversation_id = database.create_conversation("A conversation")
    message_id = database.add_message(conversation_id, "assistant", "Answer.")
    database.save_feedback("Question?", "Answer.", [], -1, message_id=message_id)

    database.delete_conversation(conversation_id)

    assert database.list_conversations() == []
    assert database.get_messages(conversation_id) == []
    with database.closing(database.get_connection()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
        assert connection.execute(
            "SELECT message_id FROM feedback"
        ).fetchone() == (None,)


def test_delete_all_conversations_cascades_messages_and_unlinks_feedback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    first_id = database.create_conversation("First chat")
    second_id = database.create_conversation("Second chat")
    first_message_id = database.add_message(first_id, "assistant", "First answer.")
    second_message_id = database.add_message(second_id, "assistant", "Second answer.")
    database.save_feedback("First question?", "First answer.", [], 1, first_message_id)
    database.save_feedback("Second question?", "Second answer.", [], -1, second_message_id)

    assert database.delete_all_conversations() == 2
    assert database.list_conversations() == []
    assert database.get_messages(first_id) == []
    assert database.get_messages(second_id) == []
    with database.closing(database.get_connection()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM feedback WHERE message_id IS NULL"
        ).fetchone() == (2,)
    assert database.delete_all_conversations() == 0


def test_touch_conversation_updates_activity_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "database" / "faq.db")
    conversation_id = database.create_conversation("A conversation")
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "UPDATE conversations SET updated_at = '2000-01-01 00:00:00' "
                "WHERE id = ?",
                (conversation_id,),
            )

    database.touch_conversation(conversation_id)

    assert database.list_conversations()[0]["updated_at"] != "2000-01-01 00:00:00"


def test_create_database_migrates_legacy_feedback_table(tmp_path, monkeypatch):
    database_path = tmp_path / "database" / "faq.db"
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    faq_ids TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "INSERT INTO feedback (question, answer, faq_ids, rating) "
                "VALUES ('Question?', 'Answer.', '[]', 1)"
            )

    database.create_database()

    with database.closing(database.get_connection()) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(feedback)")
        }
        existing_feedback = connection.execute(
            "SELECT question, rating, message_id FROM feedback"
        ).fetchone()

    assert "message_id" in columns
    assert existing_feedback == ("Question?", 1, None)


def test_create_database_preserves_legacy_messages_without_fallback_status(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "database" / "faq.db"
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    with database.closing(database.get_connection()) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE conversations ("
                "id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "INSERT INTO conversations (id, title) VALUES (1, 'Legacy chat')"
            )
            connection.execute(
                "CREATE TABLE messages ("
                "id INTEGER PRIMARY KEY, "
                "conversation_id INTEGER NOT NULL REFERENCES conversations(id), "
                "role TEXT NOT NULL, content TEXT NOT NULL, "
                "sources_json TEXT NOT NULL DEFAULT '[]', "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources_json) "
                "VALUES (1, 1, 'assistant', 'Legacy answer.', "
                "'[{\"id\": 1, \"question\": \"Legacy FAQ?\", \"distance\": 0.1}]')"
            )

    database.create_database()

    messages = database.get_messages(1)

    assert messages[0]["sources"][0]["question"] == "Legacy FAQ?"
    assert messages[0]["used_fallback"] is None
