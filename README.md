# FAQ Support Desk (RAG Chatbot)

A Retrieval-Augmented Generation chatbot for FAQs. FastEmbed creates FAQ
embeddings, SQLite with `sqlite-vec` retrieves similar entries, and Groq
generates answers grounded in the retrieved FAQs.

## Features

- Semantic FAQ search with a relevance cutoff and a friendly fallback
- Optional hybrid search (vector + BM25), query rewriting, and reranking
- Saved chat history in the sidebar (search, rename, delete, clear all)
- Related-question suggestions and "Based on" attribution under answers
- Small-talk and empathetic replies; crisis messages bypass FAQ search
- Optional signup/login with Argon2 password hashing and signed sessions
- Optional password-protected Admin page (FAQ management, analytics,
  unanswered-question clusters)
- Docker support and a GitHub Actions workflow that runs the tests

## Requirements

- Python 3.11 or newer
- A Groq API key for generated answers (retrieval works without one)

## Windows setup

Run in PowerShell from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and
activate again. In Command Prompt, use `.venv\Scripts\activate.bat` instead.

Put your key in `.env` as `GROQ_API_KEY=...`. The `.env` file contains secrets,
is ignored by Git, and must never be committed or shared. The first launch
downloads the embedding model and builds `database/faq.db` from
`data/faqs.json`.

## Run the app

```powershell
streamlit run main.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. The sidebar always
shows Chat, and shows Admin only when `ADMIN_PASSWORD` is set.

## Run with Docker

Start Docker Desktop and wait for "Engine running", then:

```cmd
docker build -t faq-rag-chatbot .
docker run --rm -p 8501:8501 --env-file .env -v faq_data:/app/database faq-rag-chatbot
```

- The build pre-downloads the FastEmbed model into the image.
- The `-v faq_data:/app/database` volume keeps chats and accounts between runs.
- `--env-file` is read only at container start, so restart after editing `.env`.
- Rebuild the image after changing code or `requirements.txt`.

## Configuration

Copy `.env.example` to `.env`. Do not wrap values in quotes (Docker's
`--env-file` keeps them).

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Enables generated answers |
| `RAG_MAX_DISTANCE` | `1.15` | Relevance cutoff (Euclidean distance) |
| `RAG_USE_QUERY_REWRITE` | see `.env.example` | Rewrite follow-ups into standalone questions with Groq |
| `RAG_USE_HYBRID` | see `.env.example` | Combine vector and BM25 rankings with RRF (k=60) |
| `RAG_USE_RERANK` | `false` | Rerank 10 candidates, keep top 3 |
| `SHOW_SOURCES` | `false` | Show a Sources expander with FAQ matches |
| `SHOW_FEEDBACK` | `false` | Show thumbs up/down under answers |
| `SHOW_RELATED` | `true` | Show up to 3 related questions |
| `UNANSWERED_CLUSTER_THRESHOLD` | `0.8` | Cosine threshold for grouping unanswered questions |
| `ADMIN_PASSWORD` | none | Enables the Admin page |
| `AUTH_ENABLED` | `false` | Enables Login and Create Account pages |
| `AUTH_SECRET` | none | Secret for signing sessions (required if auth is on) |
| `AUTH_ALLOW_GUEST` | `false` | Shows a guest button (guest chats are not saved) |
| `AUTH_PUBLIC_URL` | none | Adds a direct link to password-reset emails |
| `SMTP_HOST` | none | Enables password-reset emails |
| `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_SSL` | none | SMTP settings (STARTTLS by default) |

Generate a strong secret with:

```cmd
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Retrieval options

- **Vector cutoff:** a question with no match under `RAG_MAX_DISTANCE` gets a
  fallback reply instead of being sent to Groq with weak context.
- **Hybrid search:** the cutoff still uses the nearest vector match. Set
  `RAG_USE_HYBRID=false` for vector-only search.
- **Reranking:** uses the FastEmbed `Xenova/ms-marco-MiniLM-L-6-v2`
  cross-encoder, loaded lazily and cached.
- **Query rewriting:** self-contained questions are searched as-is. Without a
  Groq key or when disabled, the app uses recent conversation context instead.

## Chat history

Conversations are stored in SQLite. Use the sidebar to search and reopen chats,
rename them (up to 40 characters), or delete them. "Clear all chats" asks for
confirmation. Deleting a chat removes its messages; feedback records are kept
with their message links cleared. Related questions, sources and feedback status
are restored when a chat is reopened.

New chats stay titled "New chat" through small talk or fallback replies, then are
renamed after the first successful FAQ answer, unless the title was set manually.

## Small talk and safety

- Greetings and brief small talk get friendly replies without FAQ search.
  Messages that also ask about the service still use FAQ search.
- Feeling tired, stressed, sad, bored or overwhelmed gets a short empathetic
  reply with clickable FAQ suggestions.
- Crisis messages bypass FAQ search and generated answers, and encourage
  contacting local emergency services or a trusted person immediately.

## Authentication

Authentication is off by default. Set `AUTH_ENABLED=true` and `AUTH_SECRET` to
enable it.

- Signup needs a 2-60 character name, a valid email, and matching passwords of
  at least 8 characters.
- Passwords are stored only as Argon2 hashes.
- Failed logins trigger a temporary lockout.
- Sessions are signed and last seven days.
- Signed-in users can only access their own chats.
- Password reset emails require `SMTP_HOST`. Tokens are single-use and expire
  after 30 minutes. Without `AUTH_PUBLIC_URL`, users paste the emailed token into
  the app.

## Admin page

Optional and password-protected. It appears only when `ADMIN_PASSWORD` is set. It
supports FAQ management, reindexing, unanswered-question review, feedback totals,
and analytics (daily questions, fallback rate, common questions, totals).
An unanswered cluster can be added as a new FAQ: saving validates it, updates the
search index incrementally, and removes the cluster's unanswered entries.

## Rebuild the FAQ index

After editing `data/faqs.json`:

```powershell
python database.py
```

Content hashes skip embedding for unchanged entries. The command reports added,
updated, removed and unchanged counts.

## Tests and evaluation

```powershell
pip install -r requirements-dev.txt
pytest
python evaluate.py
```

`evaluate.py` runs `tests/eval_questions.json` and reports hit@1, hit@3, fallback
accuracy and retrieval failures. FAQ IDs in the evaluation set follow their order
in `data/faqs.json`. GitHub Actions runs `pytest` on every push.

## Project layout

- `main.py`: multipage entry point and navigation
- `app.py`: chat interface
- `pages/Admin.py`: admin page
- `rag.py`: retrieval, relevance cutoff, prompt building
- `database.py`: SQLite setup, migrations, FAQ validation, indexing
- `chat_helpers.py`, `small_talk.py`: chat and small-talk helpers
- `evaluate.py`, `tests/`: evaluation and unit tests
- `data/faqs.json`: FAQ source data
- `database/faq.db`: generated database (not committed)
- `styles.css`, `assets/`, `.streamlit/`: styling, images and theme
- `requirements.txt`, `requirements-dev.txt`: dependencies
- `.env.example`: environment variable template

## Notes

- The sample FAQ content is illustrative. Replace it with your own.
- Without a Groq key, retrieval still works but generated answers are disabled
  and a notice is shown.