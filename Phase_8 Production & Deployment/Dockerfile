# Dockerfile — Production FastAPI Backend
#
# Build:  docker build -t ai-service .
# Run:    docker run -p 8000:8000 --env-file .env ai-service

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer, cached unless requirements file changes
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy application code — changes here don't invalidate the pip install cache
COPY rag_service.py observability.py main.py evaluation.py ./

# Document the port (informational — actual publishing happens at `docker run -p`)
EXPOSE 8000

# Run as a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Basic container-level health check (Docker/Compose can use this directly)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
