# Dockerfile — Zepto Support Assistant
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Run document ingestion to build the ChromaDB vector store at build time
RUN python ingest.py

# Default: mock mode (no API key required, graded baseline)
ENV MOCK_LLM=1

# Expose FastAPI port
EXPOSE 7860

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
