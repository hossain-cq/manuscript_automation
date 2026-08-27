# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/nonroot \
    APP_HOME=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonroot --create-home app

WORKDIR ${APP_HOME}

# Copy a locked dependency file in the real project.
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --chown=10001:10001 src/ ./src/
COPY --chown=10001:10001 migrations/ ./migrations/

RUN mkdir -p /var/lib/manuscript/artifacts /var/lib/manuscript/checkpoints \
    && chown -R 10001:10001 /var/lib/manuscript

USER 10001:10001
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "src.api"]
