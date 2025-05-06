FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  librdkafka-dev \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir aiokafka

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Set the agent service to run
ARG AGENT_SERVICE
ENV AGENT_SERVICE=${AGENT_SERVICE}

# Default port
ARG PORT=8000
ENV PORT=${PORT}

# Run the service
CMD ["sh", "-c", "python -m intelligent_chat_service.agents.${AGENT_SERVICE}"]
