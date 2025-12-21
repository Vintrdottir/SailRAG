FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for curl (healthchecks) + optional future OCR deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

COPY src /app/src

EXPOSE 8000

CMD ["uvicorn", "sailrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
