# Intelligent Chat Service

An AI-powered chat service with agentic capabilities, orchestrating different AI agents to provide comprehensive responses.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the Service

```bash
# Start the service
./scripts/execute.sh
```

### Example API Call

```bash
curl --location 'http://localhost:8000/ai/chat' \
--header 'Content-Type: application/json' \
--data '{
"workflow_name": "idiscovery_orchestrator",
"user_input": "What is an agentic AI framework",
"selected_sources": ["Internet"],
"config": {
  "thread_id": "1",
  "client_id": "2",
  "user_id": "3",
  "session_id": "4"
},
"stream": true
}'
```

## API Documentation

Access the API documentation at `http://localhost:8000/docs` when the service is running.

## Architecture

The service uses an orchestrator pattern to coordinate different AI agents:

- **Analyzer Agent**: Analyzes user queries to understand intent and context
- **Planner Agent**: Creates a plan to address the user's query using available tools

## Troubleshooting

If you encounter errors, check the logs at `logs/app.log` for detailed information.

## PDF Ingestion

#### Example PDF Upload

```bash
python rag/ingestion/ingestion.py /home/venkatnm94/prototypes/agentic-ai/intelligent-chat-service/MDL-120.pdf
```
