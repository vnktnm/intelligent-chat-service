# Intelligent Chat Service Postman Collection

This directory contains a Postman collection for testing the Intelligent Chat Service API.

## Setup Instructions

1. **Install Postman**

   - Download and install Postman from [postman.com](https://www.postman.com/downloads/)

2. **Import the Collection**

   - Open Postman
   - Click "Import" button
   - Select the `intelligent_chat_service.postman_collection.json` file

3. **Set Up Environment**
   - Create a new environment in Postman (e.g., "Local Development")
   - Set the following variables:
     - `baseUrl`: `http://localhost:8000` (or wherever your API is hosted)
     - Leave the other variables empty - they will be populated during testing

## Testing Workflow

### 1. Health Check

- Run the "Health Check" request to verify that the API is up and running

### 2. Create a Tool

- Run the "Create Tool" request
- In the "Tests" tab, add the following script to automatically capture the tool ID:
  ```javascript
  const responseJson = pm.response.json();
  if (responseJson.id) {
    pm.environment.set('toolId', responseJson.id);
    console.log('Tool ID set to: ' + responseJson.id);
  }
  ```
- This will save the tool ID to use in subsequent requests

### 3. Testing Human-in-the-Loop

- Run the "Execute Chat Workflow" request with a prompt that might require human judgment
- Check the "Get Pending Interactions" request
- If there are pending interactions, copy an interaction ID and set it as the `interactionId` variable
- Submit a response using the "Submit Human Response" request

#### For Step-Based Workflows:

1. Run the "Step-based Workflow with Human Interaction" request
2. Check the "Check Pending Human Interactions" request
3. Copy an interaction ID from the response and set it as the `interactionId` variable
4. Submit a response using the "Submit Human Response (Step-based)" request

#### For Graph-Based Workflows:

1. Run the "Graph-based Workflow with Human Interaction" request
2. Check the "Check Pending Human Interactions" request
3. Copy an interaction ID from the response and set it as the `graphInteractionId` variable
4. Submit a response using the "Submit Human Response (Graph-based)" request

### 4. Tool Management

After creating a tool:

1. Run "Get Tool by ID" to verify it was created (using the updated `/tool/{id}` endpoint)
2. Run "List Tools" to see all tools (using the `/list` endpoint)
3. Run "Search Tools" to find tools by semantic search
4. Run "Update Tool" to modify the tool (using the updated `/tool/{id}` endpoint)
5. Run "Delete Tool" to remove the tool when done testing (using the updated `/tool/{id}` endpoint)

### 5. Graph Orchestrator Testing

To test the graph-based orchestrator:

1. Run "Execute Graph Workflow" to test the built-in graph orchestrator
2. Run "Register Graph" to create a custom graph definition
3. In the "Tests" tab, add the following script to capture the graph ID:
   ```javascript
   const responseJson = pm.response.json();
   if (responseJson.graph_id) {
     pm.environment.set('graphId', responseJson.graph_id);
     console.log('Graph ID set to: ' + responseJson.graph_id);
   }
   ```
4. Run "List Graphs" to see all registered graphs
5. Run "Get Graph Details" to see the details of your registered graph
6. Run "Visualize Graph (JSON)" or "Visualize Graph (DOT)" to get visualization data
7. Run "Delete Graph" when done testing

## Variables

The collection uses the following variables:

- `baseUrl`: The base URL of your API
- `toolId`: ID of a tool (populated after creating a tool)
- `interactionId`: ID of a human interaction for step-based workflow
- `graphInteractionId`: ID of a human interaction for graph-based workflow
- `sessionId`: Session ID for testing interaction history
- `threadId`: Thread ID for testing interaction history
- `graphId`: ID of a graph orchestrator (populated after registering a graph)

## Additional Notes

- The Chat endpoint uses Server-Sent Events (SSE), which may not display properly in Postman. Consider using the UI test page for interactive testing.
- Set appropriate values for environment variables before running requests that require them.
- The updated tool endpoints use the following paths:
  - List tools: GET `/ai/tools/list`
  - Get tool: GET `/ai/tools/tool/{id}`
  - Update tool: PUT `/ai/tools/tool/{id}`
  - Delete tool: DELETE `/ai/tools/tool/{id}`
  - Search tools: POST `/ai/tools/search`
- The graph orchestrator endpoints are under the `/graph` prefix:
  - Execute graph workflow: POST `/ai/chat` (with workflow_name="graph_basic_orchestrator")
  - Register graph: POST `/graph/register`
  - List graphs: GET `/graph/list`
  - Get graph details: GET `/graph/{graphId}`
  - Visualize graph: GET `/graph/{graphId}/visualize?format=json|dot`
  - Delete graph: DELETE `/graph/{graphId}`
