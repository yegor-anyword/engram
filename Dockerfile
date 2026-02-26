FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY engram/ engram/

# Install the package
RUN pip install --no-cache-dir .

# Default port
EXPOSE 5820

# Run the server
CMD ["python", "-m", "engram.server.app"]
