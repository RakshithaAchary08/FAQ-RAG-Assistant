import hashlib
import json
import re
import sqlite3
import secrets
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import sqlite_vec
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "faq.db"
FAQ_PATH = BASE_DIR / "data" / "faqs.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
PASSWORD_HASHER = PasswordHasher()
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash("not-a-valid-user-password")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
    except Exception:
        connection.close()
        raise
    return connection


def create_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS faqs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT ''
                )
                """
            )
            faq_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(faqs)").fetchall()
            }
            if "content_hash" not in faq_columns:
                connection.execute(
                    "ALTER TABLE faqs ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS faq_vectors
                USING vec0(embedding float[{EMBEDDING_DIMENSION}])
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS faq_fts
                USING fts5(question, answer)
                """
            )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS faqs_fts_insert AFTER INSERT ON faqs
                BEGIN
                    INSERT INTO faq_fts(rowid, question, answer)
                    VALUES (new.id, new.question, new.answer);
                END;
                CREATE TRIGGER IF NOT EXISTS faqs_fts_delete AFTER DELETE ON faqs
                BEGIN
                    DELETE FROM faq_fts WHERE rowid = old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS faqs_fts_update AFTER UPDATE ON faqs
                BEGIN
                    DELETE FROM faq_fts WHERE rowid = old.id;
                    INSERT INTO faq_fts(rowid, question, answer)
                    VALUES (new.id, new.question, new.answer);
                END;
                """
            )
            connection.execute(
                """
                INSERT INTO faq_fts(rowid, question, answer)
                SELECT faqs.id, faqs.question, faqs.answer
                FROM faqs
                WHERE NOT EXISTS (
                    SELECT 1 FROM faq_fts WHERE faq_fts.rowid = faqs.id
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL DEFAULT '',
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT
                )
                """
            )
            user_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "full_name" not in user_columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN full_name TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    title_is_custom INTEGER NOT NULL DEFAULT 0,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conversation_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(conversations)"
                ).fetchall()
            }
            if "title_is_custom" not in conversation_columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN title_is_custom "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            if "user_id" not in conversation_columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN user_id INTEGER "
                    "REFERENCES users(id) ON DELETE CASCADE"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC, id DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    used_fallback INTEGER NOT NULL DEFAULT 0,
                    suggestions_json TEXT NOT NULL DEFAULT '[]',
                    related_questions_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            message_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "used_fallback" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN used_fallback INTEGER"
                )
            if "suggestions_json" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN suggestions_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            if "related_questions_json" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN related_questions_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    faq_ids TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            feedback_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
            }
            if "message_id" not in feedback_columns:
                connection.execute(
                    """
                    ALTER TABLE feedback ADD COLUMN message_id INTEGER
                    REFERENCES messages(id) ON DELETE SET NULL
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON messages(conversation_id, id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_message
                ON feedback(message_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS unanswered (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def save_feedback(
    question: str,
    answer: str,
    faq_ids: list[int],
    rating: int,
    message_id: int | None = None,
) -> None:
    if rating not in (-1, 1):
        raise ValueError("Feedback rating must be 1 (positive) or -1 (negative).")

    with closing(get_connection()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO feedback (question, answer, faq_ids, rating, message_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (question, answer, json.dumps(faq_ids), rating, message_id),
            )


def _validate_email(email: str) -> str:
    """Normalize and validate a user email address."""
    clean_email = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(clean_email):
        raise ValueError("Enter a valid email address.")
    return clean_email


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")


def _validate_full_name(full_name: str) -> str:
    clean_name = full_name.strip()
    if not 2 <= len(clean_name) <= 60:
        raise ValueError("Full name must be between 2 and 60 characters.")
    return clean_name


def validate_signup(
    email: str,
    password: str,
    confirmation: str,
    full_name: str,
) -> tuple[str, str]:
    """Validate signup fields without modifying the password."""
    clean_email = _validate_email(email)
    _validate_password(password)
    clean_name = _validate_full_name(full_name)
    if password != confirmation:
        raise ValueError("Passwords do not match.")
    return clean_email, clean_name


def create_user(
    email: str,
    password: str,
    full_name: str | None = None,
) -> dict[str, object]:
    """Create a user with an Argon2 password hash."""
    clean_email = _validate_email(email)
    _validate_password(password)
    clean_name = _validate_full_name(full_name) if full_name is not None else ""
    password_hash = PASSWORD_HASHER.hash(password)
    create_database()
    try:
        with closing(get_connection()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, full_name, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (clean_email, clean_name, password_hash),
                )
                user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with that email already exists.") from error
    if user_id is None:
        raise RuntimeError("Could not create user.")
    return {"id": int(user_id), "email": clean_email, "full_name": clean_name}


def get_user_by_email(email: str) -> dict[str, object] | None:
    """Return a user record without exposing its password hash."""
    clean_email = _validate_email(email)
    create_database()
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT id, email, created_at, full_name FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "email": row[1],
        "created_at": row[2],
        "full_name": row[3],
    }


def verify_login(email: str, password: str) -> dict[str, object] | None:
    """Verify credentials and apply a five-failure, five-minute lockout."""
    try:
        clean_email = _validate_email(email)
    except ValueError:
        try:
            PASSWORD_HASHER.verify(DUMMY_PASSWORD_HASH, password)
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            pass
        return None
    create_database()
    now = datetime.now(timezone.utc)
    with closing(get_connection()) as connection:
        with connection:
            user = connection.execute(
                """
                SELECT id, email, full_name, password_hash, failed_attempts,
                       locked_until
                FROM users WHERE email = ?
                """,
                (clean_email,),
            ).fetchone()
            if user is None:
                try:
                    PASSWORD_HASHER.verify(DUMMY_PASSWORD_HASH, password)
                except (InvalidHashError, VerificationError, VerifyMismatchError):
                    pass
                return None

            (
                user_id,
                stored_email,
                full_name,
                password_hash,
                attempts,
                locked_until,
            ) = user
            if locked_until:
                lock_time = datetime.fromisoformat(locked_until)
                if lock_time.tzinfo is None:
                    lock_time = lock_time.replace(tzinfo=timezone.utc)
                if lock_time > now:
                    return None
                connection.execute(
                    "UPDATE users SET failed_attempts = 0, locked_until = NULL "
                    "WHERE id = ?",
                    (user_id,),
                )
                attempts = 0

            try:
                PASSWORD_HASHER.verify(password_hash, password)
            except (InvalidHashError, VerificationError, VerifyMismatchError):
                attempts += 1
                new_lock = (
                    (now + timedelta(minutes=5)).isoformat()
                    if attempts >= 5
                    else None
                )
                connection.execute(
                    "UPDATE users SET failed_attempts = ?, locked_until = ? "
                    "WHERE id = ?",
                    (attempts, new_lock, user_id),
                )
                return None

            connection.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL "
                "WHERE id = ?",
                (user_id,),
            )
    return {
        "id": int(user_id),
        "email": stored_email,
        "full_name": full_name,
    }


def create_password_reset_token(email: str) -> str | None:
    """Create a single-use, 30-minute reset token for a registered user."""
    clean_email = _validate_email(email)
    create_database()
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    with closing(get_connection()) as connection:
        with connection:
            user = connection.execute(
                "SELECT id FROM users WHERE email = ?",
                (clean_email,),
            ).fetchone()
            if user is None:
                return None
            connection.execute(
                "DELETE FROM password_reset_tokens WHERE user_id = ?",
                (user[0],),
            )
            connection.execute(
                """
                INSERT INTO password_reset_tokens (token_hash, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (token_hash, user[0], expires_at),
            )
    return token


def reset_password(token: str, new_password: str) -> bool:
    """Set a new password with a valid, unexpired, single-use reset token."""
    _validate_password(new_password)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            reset = connection.execute(
                """
                SELECT user_id, expires_at
                FROM password_reset_tokens
                WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if reset is None:
                return False
            expires_at = datetime.fromisoformat(reset[1])
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= now:
                connection.execute(
                    "DELETE FROM password_reset_tokens WHERE token_hash = ?",
                    (token_hash,),
                )
                return False
            password_hash = PASSWORD_HASHER.hash(new_password)
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, failed_attempts = 0, locked_until = NULL
                WHERE id = ?
                """,
                (password_hash, reset[0]),
            )
            connection.execute(
                "DELETE FROM password_reset_tokens WHERE token_hash = ?",
                (token_hash,),
            )
    return True


def create_conversation(title: str, user_id: int | None = None) -> int:
    """Create a conversation and return its database ID."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Conversation title cannot be empty.")
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO conversations (title, user_id) VALUES (?, ?)",
                (clean_title, user_id),
            )
            conversation_id = cursor.lastrowid
    if conversation_id is None:
        raise RuntimeError("Could not create conversation.")
    return int(conversation_id)


def list_conversations(user_id: int | None = None) -> list[dict[str, object]]:
    """List conversations, optionally scoped to their owner."""
    create_database()
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, title, title_is_custom, created_at, updated_at
            FROM conversations
            WHERE user_id IS ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "id": row_id,
            "title": title,
            "title_is_custom": bool(title_is_custom),
            "created_at": created_at,
            "updated_at": updated_at,
        }
        for row_id, title, title_is_custom, created_at, updated_at in rows
    ]


def get_messages(
    conversation_id: int,
    user_id: int | None = None,
) -> list[dict[str, object]]:
    """Get messages only when the requested user owns the conversation."""
    create_database()
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT messages.id, messages.role, messages.content,
                   messages.sources_json, messages.created_at,
                   messages.used_fallback,
                   messages.suggestions_json,
                   messages.related_questions_json,
                   (
                       SELECT feedback.rating
                       FROM feedback
                       WHERE feedback.message_id = messages.id
                       ORDER BY feedback.id DESC
                       LIMIT 1
                   )
            FROM messages
            JOIN conversations
              ON conversations.id = messages.conversation_id
            WHERE messages.conversation_id = ?
              AND conversations.user_id IS ?
            ORDER BY messages.id
            """,
            (conversation_id, user_id),
        ).fetchall()
    messages: list[dict[str, object]] = []
    last_user_question = ""
    for (
        message_id,
        role,
        content,
        sources_json,
        created_at,
        used_fallback,
        suggestions_json,
        related_questions_json,
        rating,
    ) in rows:
        sources = json.loads(sources_json)
        suggestions = json.loads(suggestions_json)
        related_questions = json.loads(related_questions_json)
        if role == "user":
            last_user_question = content
        messages.append({
            "id": message_id,
            "role": role,
            "content": content,
            "sources": sources,
            "created_at": created_at,
            "used_fallback": (
                None if used_fallback is None else bool(used_fallback)
            ),
            "suggested_questions": suggestions,
            "related_questions": related_questions,
            "feedback_rating": rating,
            "question": last_user_question if role == "assistant" else content,
            "faq_ids": [
                source["id"]
                for source in sources
                if isinstance(source, dict) and "id" in source
            ],
        })
    return messages


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict[str, object]] | None = None,
    used_fallback: bool = False,
    suggested_questions: list[str] | None = None,
    related_questions: list[str] | None = None,
    user_id: int | None = None,
) -> int:
    """Persist a message and update its conversation's activity timestamp."""
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be 'user' or 'assistant'.")
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, sources_json, used_fallback,
                    suggestions_json, related_questions_json
                )
                SELECT ?, ?, ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM conversations
                    WHERE id = ? AND user_id IS ?
                )
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources or []),
                    int(used_fallback),
                    json.dumps(suggested_questions or []),
                    json.dumps(related_questions or []),
                    conversation_id,
                    user_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("Conversation does not exist or is not owned by user.")
            connection.execute(
                """
                UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id IS ?
                """,
                (conversation_id, user_id),
            )
            message_id = cursor.lastrowid
    if message_id is None:
        raise RuntimeError("Could not save message.")
    return int(message_id)


def rename_conversation(
    conversation_id: int,
    title: str,
    is_custom: bool = True,
    user_id: int | None = None,
) -> bool:
    """Rename a conversation; manual names are protected from later auto-renames."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Conversation title cannot be empty.")
    if is_custom and len(clean_title) > 40:
        raise ValueError("Conversation title cannot exceed 40 characters.")
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?,
                    title_is_custom = CASE WHEN ? THEN 1 ELSE title_is_custom END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id IS ?
                  AND (? OR title_is_custom = 0)
                """,
                (
                    clean_title,
                    int(is_custom),
                    conversation_id,
                    user_id,
                    int(is_custom),
                ),
            )
    if cursor.rowcount == 0:
        with closing(get_connection()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id IS ?",
                (conversation_id, user_id),
            ).fetchone()
        if exists is None:
            raise ValueError(f"Conversation {conversation_id} does not exist.")
        return False
    return True


def delete_conversation(
    conversation_id: int,
    user_id: int | None = None,
) -> None:
    """Delete a conversation and its messages using the cascading FK."""
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id IS ?",
                (conversation_id, user_id),
            )
    if cursor.rowcount == 0:
        raise ValueError(f"Conversation {conversation_id} does not exist.")


def delete_all_conversations(user_id: int | None = None) -> int:
    """Delete the selected user's conversations and return the number removed."""
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE user_id IS ?",
                (user_id,),
            )
    return cursor.rowcount


def touch_conversation(
    conversation_id: int,
    user_id: int | None = None,
) -> None:
    """Update the conversation's activity timestamp."""
    create_database()
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id IS ?
                """,
                (conversation_id, user_id),
            )
    if cursor.rowcount == 0:
        raise ValueError(f"Conversation {conversation_id} does not exist.")


def log_unanswered(question: str) -> None:
    with closing(get_connection()) as connection:
        with connection:
            connection.execute(
                "INSERT INTO unanswered (question) VALUES (?)",
                (question,),
            )


def validate_faqs(faqs: Any) -> list[dict[str, str]]:
    """Validate an FAQ collection and normalize its question and answer text."""
    if not isinstance(faqs, list) or not faqs:
        raise ValueError("The FAQ file must contain a non-empty JSON array.")

    validated: list[dict[str, str]] = []
    for index, faq in enumerate(faqs):
        if not isinstance(faq, dict):
            raise ValueError(f"FAQ entry {index + 1} must be an object.")
        question = faq.get("question")
        answer = faq.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"FAQ entry {index + 1} needs a question.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"FAQ entry {index + 1} needs an answer.")
        validated.append({"question": question.strip(), "answer": answer.strip()})
    return validated


def load_faqs() -> list[dict[str, str]]:
    """Load and validate FAQs from the source JSON file."""
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        return validate_faqs(json.load(file))


def save_faqs(faqs: Any) -> list[dict[str, str]]:
    """Validate and atomically save FAQs to the source JSON file."""
    validated = validate_faqs(faqs)
    FAQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=FAQ_PATH.parent,
            prefix=f".{FAQ_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(validated, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        temporary_path.replace(FAQ_PATH)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return validated


def get_unanswered_questions(limit: int = 1000) -> list[dict[str, object]]:
    """Return the newest unanswered questions for the admin page."""
    if limit < 1:
        raise ValueError("The unanswered-question limit must be positive.")
    create_database()
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, question, created_at
            FROM unanswered
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {"id": row_id, "question": question, "created_at": created_at}
        for row_id, question, created_at in rows
    ]


def delete_unanswered_questions(question_ids: list[int]) -> int:
    """Remove selected unanswered rows and return the number deleted."""
    if not question_ids:
        return 0
    create_database()
    placeholders = ", ".join("?" for _ in question_ids)
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                f"DELETE FROM unanswered WHERE id IN ({placeholders})",
                question_ids,
            )
    return cursor.rowcount


def get_analytics_stats() -> dict[str, object]:
    """Return daily question counts, fallback rate, popular questions and totals."""
    create_database()
    with closing(get_connection()) as connection:
        daily_rows = connection.execute(
            """
            SELECT date(created_at), COUNT(*)
            FROM messages
            WHERE role = 'user'
            GROUP BY date(created_at)
            ORDER BY date(created_at)
            """
        ).fetchall()
        fallback_row = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(COALESCE(used_fallback, 0) = 1), 0)
            FROM messages
            WHERE role = 'assistant'
            """
        ).fetchone()
        top_rows = connection.execute(
            """
            SELECT content, COUNT(*)
            FROM messages
            WHERE role = 'user'
            GROUP BY content
            ORDER BY COUNT(*) DESC, content COLLATE NOCASE
            LIMIT 10
            """
        ).fetchall()
        totals_row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM conversations),
                (SELECT COUNT(*) FROM messages),
                (SELECT COUNT(*) FROM unanswered)
            """
        ).fetchone()

    if fallback_row is None or totals_row is None:
        raise RuntimeError("Could not retrieve analytics statistics.")
    assistant_count, fallback_count = fallback_row
    return {
        "questions_per_day": [
            {"date": day, "questions": count}
            for day, count in daily_rows
        ],
        "fallback_rate": fallback_count / assistant_count if assistant_count else 0.0,
        "top_questions": [
            {"question": question, "count": count}
            for question, count in top_rows
        ],
        "total_conversations": totals_row[0],
        "total_messages": totals_row[1],
        "total_unanswered": totals_row[2],
    }


def get_feedback_stats() -> dict[str, int]:
    """Return total, positive, and negative feedback counts."""
    create_database()
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(rating = 1), 0),
                   COALESCE(SUM(rating = -1), 0)
            FROM feedback
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("Could not retrieve feedback statistics.")
    return {"total": row[0], "positive": row[1], "negative": row[2]}


def create_embeddings() -> dict[str, int]:
    """Incrementally synchronize FAQ records and embeddings with the source file."""
    create_database()
    faqs = load_faqs()
    with closing(get_connection()) as connection:
        existing_faqs = connection.execute(
            "SELECT id, question, answer, content_hash FROM faqs ORDER BY id"
        ).fetchall()
        existing_vectors = {
            row[0]: bytes(row[1])
            for row in connection.execute(
                "SELECT rowid, embedding FROM faq_vectors"
            ).fetchall()
        }

    existing_by_id = {row[0]: row for row in existing_faqs}
    hashes = [
        hashlib.sha256(
            f"{faq['question']}\0{faq['answer']}".encode("utf-8")
        ).hexdigest()
        for faq in faqs
    ]
    assignments: dict[int, tuple[int | None, str]] = {}
    used_old_ids: set[int] = set()

    for target_id, (faq, content_hash) in enumerate(
        zip(faqs, hashes, strict=True),
        start=1,
    ):
        for old_id, old_question, old_answer, old_hash in existing_faqs:
            if old_id in used_old_ids or old_id not in existing_vectors:
                continue
            if (
                old_question == faq["question"]
                and old_answer == faq["answer"]
                and (old_hash == content_hash or not old_hash)
            ):
                assignments[target_id] = (old_id, "unchanged")
                used_old_ids.add(old_id)
                break

    for target_id, faq in enumerate(faqs, start=1):
        if target_id in assignments:
            continue
        old_faq = existing_by_id.get(target_id)
        if old_faq is not None and target_id not in used_old_ids:
            assignments[target_id] = (target_id, "updated")
            used_old_ids.add(target_id)
        else:
            assignments[target_id] = (None, "added")

    removed_ids = set(existing_by_id) - used_old_ids
    changes = [
        (target_id, faqs[target_id - 1], hashes[target_id - 1], status)
        for target_id, (_, status) in assignments.items()
        if status != "unchanged"
    ]
    unchanged = sum(status == "unchanged" for _, status in assignments.values())
    summary = {
        "added": sum(status == "added" for _, _, _, status in changes),
        "updated": sum(status == "updated" for _, _, _, status in changes),
        "removed": len(removed_ids),
        "unchanged": unchanged,
    }
    remapped = any(
        old_id is not None and old_id != target_id
        for target_id, (old_id, _) in assignments.items()
    )
    needs_sync = bool(changes or removed_ids or remapped)

    embeddings_by_id: dict[int, bytes] = {}
    if needs_sync:
        texts = [
            f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            for _, faq, _, _ in changes
        ]
        vectors = list(get_embedding_model().embed(texts)) if texts else []
        if len(vectors) != len(changes):
            raise RuntimeError(
                "The embedding model returned a different number of vectors than FAQs."
            )
        for (target_id, _, _, _), embedding in zip(changes, vectors, strict=True):
            embeddings_by_id[target_id] = sqlite_vec.serialize_float32(embedding)
        for target_id, (old_id, status) in assignments.items():
            if status == "unchanged" and old_id is not None:
                embeddings_by_id[target_id] = existing_vectors[old_id]

    with closing(get_connection()) as connection:
        with connection:
            if needs_sync:
                connection.execute("DELETE FROM faq_vectors")
                connection.execute("DELETE FROM faqs")
                for faq_id, (faq, content_hash) in enumerate(
                    zip(faqs, hashes, strict=True),
                    start=1,
                ):
                    connection.execute(
                        """
                        INSERT INTO faqs (id, question, answer, content_hash)
                        VALUES (?, ?, ?, ?)
                        """,
                        (faq_id, faq["question"], faq["answer"], content_hash),
                    )
                    connection.execute(
                        "INSERT INTO faq_vectors (rowid, embedding) VALUES (?, ?)",
                        (faq_id, embeddings_by_id[faq_id]),
                    )
            else:
                connection.executemany(
                    "UPDATE faqs SET content_hash = ? WHERE id = ?",
                    (
                        (content_hash, faq_id)
                        for faq_id, content_hash in enumerate(hashes, start=1)
                    ),
                )

    print(
        "FAQ sync summary: "
        f"added={summary['added']}, updated={summary['updated']}, "
        f"removed={summary['removed']}, unchanged={summary['unchanged']}"
    )
    return summary


def ensure_index() -> None:
    create_database()
    with closing(get_connection()) as connection:
        rows = connection.execute(
            "SELECT COUNT(*), COALESCE(MIN(id), 0), COALESCE(MAX(id), 0) FROM faqs"
        ).fetchone()
    if rows is None or rows[0] == 0 or rows[1] != 1 or rows[2] != rows[0]:
        create_embeddings()


if __name__ == "__main__":
    create_embeddings()
    print("FAQ database created successfully.")