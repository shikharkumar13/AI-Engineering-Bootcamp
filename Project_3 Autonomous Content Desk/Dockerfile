# Dockerfile — Content Desk backend
#
# Build:  docker build -t content-desk .
# Run:    docker run -p 8000:8000 --env-file .env content-desk

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY crew.py evaluation.py main.py ./

EXPOSE 8000

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
