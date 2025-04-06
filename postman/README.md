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

### 4. Tool Management

After creating a tool:

1. Run "Get Tool by ID" to verify it was created (using the updated `/tool/{id}` endpoint)
2. Run "List Tools" to see all tools (using the `/list` endpoint)
3. Run "Search Tools" to find tools by semantic search
4. Run "Update Tool" to modify the tool (using the updated `/tool/{id}` endpoint)
5. Run "Delete Tool" to remove the tool when done testing (using the updated `/tool/{id}` endpoint)

## Variables

The collection uses the following variables:

- `baseUrl`: The base URL of your API
- `toolId`: ID of a tool (populated after creating a tool)
- `interactionId`: ID of a human interaction (set manually after finding pending interactions)
- `sessionId`: Session ID for testing interaction history
- `threadId`: Thread ID for testing interaction history

## Additional Notes

- The Chat endpoint uses Server-Sent Events (SSE), which may not display properly in Postman. Consider using the UI test page for interactive testing.
- Set appropriate values for environment variables before running requests that require them.
- The updated tool endpoints use the following paths:
  - List tools: GET `/ai/tools/list`
  - Get tool: GET `/ai/tools/tool/{id}`
  - Update tool: PUT `/ai/tools/tool/{id}`
  - Delete tool: DELETE `/ai/tools/tool/{id}`
  - Search tools: POST `/ai/tools/search`
