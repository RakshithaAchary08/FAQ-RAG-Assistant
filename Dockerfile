FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/database "$FASTEMBED_CACHE_PATH" \
    && chown -R appuser:appuser /app /home/appuser/.cache

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app.py main.py rag.py database.py small_talk.py chat_helpers.py auth.py ./
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser pages/ ./pages/
COPY --chown=appuser:appuser styles.css ./
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

USER appuser

RUN python -c "from fastembed import TextEmbedding; model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); list(model.embed(['FAQ model warmup']))"

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
