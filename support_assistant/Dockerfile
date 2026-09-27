# Build and serve the Zepto GenAI Support Assistant
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch and packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY support_assistant/ ./support_assistant/

# Run indexer to ensure vector DB is built inside container
RUN python support_assistant/indexer.py

# Expose port 7860
EXPOSE 7860

# Default environment toggle to mock mode (graded baseline)
ENV MOCK_LLM=1
ENV PYTHONPATH=/app

# Start FastAPI application
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
