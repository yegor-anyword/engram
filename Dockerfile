FROM python:3.13-slim

WORKDIR /app

# Install system deps (libpq-dev needed for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY README.md .

COPY pyproject.toml .
COPY engram/ engram/

# Install the package
RUN pip install --no-cache-dir .

EXPOSE 5820

CMD ["python", "-m", "engram.server.app"]