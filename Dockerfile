FROM python:3.12-slim

WORKDIR /app

# Install uv for dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code
COPY gluellm/ ./gluellm/
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GLUELLM_LOG_LEVEL=INFO

# Default command (override in docker-compose or run command)
CMD ["python", "-m", "gluellm.cli", "demo"]
