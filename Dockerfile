FROM node:24-alpine AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY praxis/ ./praxis/
RUN pip install --no-cache-dir .
COPY --from=web /app/web/dist ./web/dist

ENV PRAXIS_WORK_DIR=/data \
    PRAXIS_WEB_DIST=/app/web/dist
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "praxis.api:app", "--host", "0.0.0.0", "--port", "8000"]
