# ---- build stage: resolve dependencies into a venv --------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# ---- runtime stage: just Python and the venv --------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Numeric wheels are the bulk of this image (~220 MB of scipy, pandas,
# scikit-learn and numpy). That is the cost of the bundle being a pickle:
# unpickling a scikit-learn estimator requires scikit-learn present. See
# docs/methodology.md.

RUN useradd --create-home --uid 10001 app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app/ /app/app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/data/iris-classifier.zip

WORKDIR /app
USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
