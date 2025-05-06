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

## qdrant tool

# Semantic search with only the query

curl -X POST http://localhost:8001/search/semantic \
 -H "Content-Type: application/json" \
 -d '{
"query": "What are the main features of the product?"
}'

# Keyword search with query and custom collection name

curl -X POST http://localhost:8001/search/keyword \
 -H "Content-Type: application/json" \
 -d '{
"query": "technical specifications",
"collection_name": "document_collection"
}'

# Hybrid search - simplified API with optimal defaults applied automatically

curl -X POST http://localhost:8001/search/hybrid \
 -H "Content-Type: application/json" \
 -d '{
"query": "NAIC rules"
}'

# Health check

curl -X GET http://localhost:8001/health

# tool onboarding

Usage Examples
To register the Qdrant retrieval tool using the JSON definition:

To list all registered tools:

To get information about a specific tool:

To delete a tool:

This implementation provides a complete solution for registering, updating, and managing tool information in MongoDB which can be used by your intelligent chat service.
