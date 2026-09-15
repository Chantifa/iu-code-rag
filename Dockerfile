# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch keeps the image small (the default PyPI wheel pulls CUDA libraries).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the embedding model into the image so the container works offline.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY config ./config
COPY tests ./tests

ENV DATA_DIR=/app/data
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["iu-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
