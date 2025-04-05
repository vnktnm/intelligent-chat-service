// Global variables
let eventSource = null;
let currentInteractionId = null;
let pendingHelpMarkerFound = false;
let checkingInterval = null;

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const connectionStatus = document.getElementById('connection-status');
const eventLog = document.getElementById('event-log');
const apiUrlInput = document.getElementById('api-url');
const humanToggle = document.getElementById('human-toggle');
const humanInputPanel = document.getElementById('human-input-panel');
const humanQuestion = document.getElementById('human-question');
const humanAnswer = document.getElementById('human-answer');
const submitHumanInput = document.getElementById('submit-human-input');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  // Set up event listeners
  sendButton.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  submitHumanInput.addEventListener('click', submitHumanResponse);

  // Load API URL from localStorage if available
  const savedApiUrl = localStorage.getItem('apiUrl');
  if (savedApiUrl) {
    apiUrlInput.value = savedApiUrl;
  }

  // Save API URL when changed
  apiUrlInput.addEventListener('change', () => {
    localStorage.setItem('apiUrl', apiUrlInput.value);
  });

  // Add example prompts
  addExamplePrompts();

  // Add window resize handling
  window.addEventListener('resize', () => {
    scrollToBottom();
  });

  // Add a mutation observer to monitor DOM changes in the chat area
  const observer = new MutationObserver(() => {
    scrollToBottom();
  });

  observer.observe(chatMessages, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  // Add periodic polling for pending interactions if needed
  checkPendingInteractions();

  // Add Check Pending button functionality
  const checkPendingBtn = document.getElementById('check-pending');
  if (checkPendingBtn) {
    checkPendingBtn.addEventListener('click', async () => {
      logEvent('User', 'Manually checking pending interactions');
      await checkPendingInteractions();
    });
  }

  // Add clear log button functionality
  const clearLogBtn = document.getElementById('clear-log');
  if (clearLogBtn) {
    clearLogBtn.addEventListener('click', clearEventLog);
  }

  // Initialize with instructions
  clearEventLog();

  // Make sure the status panel is visible
  const statusPanel = document.querySelector('.status-panel');
  if (statusPanel) {
    statusPanel.style.display = 'flex';
  }
});

// Add example prompts to help users
function addExamplePrompts() {
  const examples = [
    'What ethical principles should guide AI development?',
    'How should we balance innovation with safety in AI?',
    "What's your opinion on AI regulation and oversight?",
    'Should AI companies prioritize profit over safety concerns?',
  ];

  const examplesContainer = document.createElement('div');
  examplesContainer.className = 'example-prompts';
  examplesContainer.innerHTML =
    '<p>Example questions that might trigger human-in-the-loop:</p>';

  const list = document.createElement('ul');
  examples.forEach((example) => {
    const item = document.createElement('li');
    item.textContent = example;
    item.addEventListener('click', () => {
      userInput.value = example;
      userInput.focus();
    });
    list.appendChild(item);
  });

  examplesContainer.appendChild(list);
  chatMessages.appendChild(examplesContainer);
}

// Send user message
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  // Clear input
  userInput.value = '';

  // Display user message
  appendMessage('user', message);

  // Show thinking indicator
  const thinkingElement = document.createElement('div');
  thinkingElement.className = 'thinking';
  thinkingElement.textContent = 'Thinking...';
  chatMessages.appendChild(thinkingElement);

  // Get API URL
  const apiUrl = apiUrlInput.value;

  try {
    // Close existing connection if any
    closeEventSource();

    // Update connection status
    updateConnectionStatus('Connecting...');

    // Generate a unique thread ID
    const threadId = 'thread-' + Math.random().toString(36).substring(2, 15);

    // Create request payload
    const payload = {
      workflow_name: 'idiscovery_orchestrator',
      user_input: message,
      selected_sources: [],
      config: {
        thread_id: threadId,
        client_id: 'ui_test_client',
        user_id: 'ui_test_user',
        session_id: 'ui_test_session',
        human_in_the_loop: humanToggle.checked,
      },
      stream: true,
    };

    // Log the outgoing request
    logEvent('Request', payload);

    // Start the event source
    startEventSource(apiUrl, payload);

    // Remove thinking indicator after 2 seconds
    setTimeout(() => {
      if (thinkingElement.parentNode) {
        thinkingElement.remove();
      }
    }, 2000);
  } catch (error) {
    console.error('Error sending message:', error);
    appendErrorMessage('Failed to send message: ' + error.message);
    updateConnectionStatus('Error');
  }
}

// Start EventSource for SSE streaming
function startEventSource(apiUrl, payload) {
  try {
    // First, make a POST request to start the chat
    fetch(`${apiUrl}/ai/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`API responded with status ${response.status}`);
        }

        // Create the response element now
        const responseContainer = document.createElement('div');
        responseContainer.className = 'message-container assistant-message';
        const responseHeader = document.createElement('div');
        responseHeader.className = 'message-header';
        responseHeader.textContent = 'Assistant';

        const responseContent = document.createElement('div');
        responseContent.className = 'assistant-response';

        responseContainer.appendChild(responseHeader);
        responseContainer.appendChild(responseContent);
        chatMessages.appendChild(responseContainer);

        // Handle the stream response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        updateConnectionStatus('Connected');

        function processStream() {
          reader
            .read()
            .then(({ done, value }) => {
              if (done) {
                updateConnectionStatus('Disconnected');
                processLastChunk(buffer);

                // If a help marker was found but no official request came through, check for pending interactions
                if (pendingHelpMarkerFound && !currentInteractionId) {
                  logEvent(
                    'Info',
                    'Marker found but no input request received, checking pending interactions'
                  );
                  checkPendingInteractions();

                  // Start checking periodically
                  if (!checkingInterval) {
                    checkingInterval = setInterval(() => {
                      checkPendingInteractions();
                    }, 3000); // Check every 3 seconds

                    // Stop checking after 30 seconds
                    setTimeout(() => {
                      if (checkingInterval) {
                        clearInterval(checkingInterval);
                        checkingInterval = null;
                      }
                    }, 30000);
                  }
                }
                return;
              }

              // Decode and process new chunks
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n\n');
              buffer = lines.pop(); // Keep the last incomplete chunk in buffer

              lines.forEach((line) => {
                if (line.startsWith('data:')) {
                  try {
                    const eventData = JSON.parse(line.substring(5));
                    handleEvent(eventData, responseContent);
                  } catch (e) {
                    console.warn(
                      'Failed to parse event data:',
                      line.substring(5)
                    );
                  }
                }
              });

              // Continue processing the stream
              processStream();
            })
            .catch((error) => {
              console.error('Error reading stream:', error);
              updateConnectionStatus('Error');
              appendErrorMessage('Stream error: ' + error.message);
            });
        }

        function processLastChunk(chunk) {
          if (chunk && chunk.startsWith('data:')) {
            try {
              const eventData = JSON.parse(chunk.substring(5));
              handleEvent(eventData, responseContent);
            } catch (e) {
              console.warn('Failed to parse final event data');
            }
          }
        }

        // Start processing the stream
        processStream();
      })
      .catch((error) => {
        console.error('Error connecting to API:', error);
        updateConnectionStatus('Disconnected');
        appendErrorMessage('Connection error: ' + error.message);
      });
  } catch (error) {
    console.error('Error setting up event source:', error);
    updateConnectionStatus('Error');
    appendErrorMessage('Setup error: ' + error.message);
  }
}

// Close the event source connection
function closeEventSource() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

// Handle incoming events - prioritize ui:human:input_requested event
function handleEvent(event, responseElement) {
  const eventType = event.event;
  const data = event.data;

  // Always log all events first for completeness
  logEvent(eventType, data);

  // Update UI debug info
  if (eventType === 'ui:human:input_requested') {
    // Ensure we're getting a clean interaction ID
    const interactionId = data.interaction_id || '';

    document.getElementById('event-status').textContent = 'Yes';
    document.getElementById('interaction-id').textContent = interactionId;
    document.getElementById('event-flow-status').textContent =
      'Request Received';

    // Store the EXACT interaction ID for later use
    if (interactionId) {
      window.lastReceivedInteractionId = interactionId;
      logEvent('Debug', `Saved interaction ID: ${interactionId}`);
    }

    // Set this to true so we know we're getting proper events
    pendingHelpMarkerFound = true;
  } else if (
    eventType === 'ui:content:chunk' &&
    data.chunk &&
    data.chunk.includes('HUMAN_HELP_NEEDED:')
  ) {
    document.getElementById('marker-status').textContent = 'Yes';
  }

  // Handle specific event types
  switch (eventType) {
    case 'ui:content:chunk':
      // Only use content chunks for display, not for functionality
      handleContentChunk(data, responseElement);
      break;

    case 'ui:human:input_requested':
      // Log more visibly for debugging
      console.log('🚨 HUMAN_INPUT_REQUESTED EVENT RECEIVED!', data);
      logEvent('IMPORTANT', 'Human input request event received');

      // This is the primary event we should be responding to
      handleHumanInputRequest(data);
      break;

    case 'ui:human:input_received':
      logEvent('IMPORTANT', 'Human input received confirmation');
      document.getElementById('event-flow-status').textContent =
        'Input Received';
      appendSystemMessage(
        `Human response received by system: "${data.response?.substring(
          0,
          50
        )}${data.response?.length > 50 ? '...' : ''}"`
      );
      break;

    case 'ui:step:update':
      if (data.status === 'updated_with_human_input') {
        logEvent('IMPORTANT', 'Response updated with human input');
        document.getElementById('event-flow-status').textContent =
          'Response Updated';
        appendSystemMessage('AI response has been updated with your feedback');
      }
      break;

    case 'ui:orchestrator:complete':
      updateConnectionStatus('Complete');
      // Clear any pending checks
      if (checkingInterval) {
        clearInterval(checkingInterval);
        checkingInterval = null;
      }
      break;
  }
}

// Handle human input request event
function handleHumanInputRequest(data) {
  const question = data.question || 'Can you help with this question?';
  const interactionId = data.interaction_id;

  if (!interactionId) {
    appendErrorMessage('Missing interaction ID in human input request');
    return;
  }

  // Log the interaction ID we're handling
  logEvent('Debug', `Setting current interaction ID to: ${interactionId}`);
  console.log('Setting current interaction ID:', interactionId);

  // Store the exact interaction ID globally - no modifications or prefixing
  currentInteractionId = interactionId;
  window.lastReceivedInteractionId = interactionId; // Save in window for recovery if needed

  humanQuestion.textContent = question;
  humanAnswer.value = '';

  // Show the human input panel
  humanInputPanel.classList.add('active');

  // Update UI state
  document.getElementById('event-flow-status').textContent =
    'Awaiting Human Response';
  document.getElementById('interaction-id').textContent = interactionId;

  // Focus the textarea
  humanAnswer.focus();

  // Add a notification sound
  try {
    const audio = new Audio(
      'https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3'
    );
    audio.volume = 0.3;
    audio.play();
  } catch (e) {
    // Ignore audio errors
  }
}

// Handle content chunks from the assistant - DISPLAY ONLY
function handleContentChunk(data, responseElement) {
  const chunk = data.chunk || '';

  // Just note if we see a help marker, but don't take action directly
  if (chunk.includes('HUMAN_HELP_NEEDED:')) {
    pendingHelpMarkerFound = true;
    logEvent('Marker', 'HUMAN_HELP_NEEDED marker detected in content');

    // Inform the user that we're waiting for the proper event
    const helpMarker = document.createElement('div');
    helpMarker.className = 'human-help-marker';
    helpMarker.innerHTML =
      '⚠️ Human help marker detected in content. <span style="font-style: italic; font-size: 0.9em;">Waiting for official request event...</span>';
    responseElement.appendChild(helpMarker);

    // After detecting a marker in content, wait briefly and check for pending interactions
    // This is a fallback mechanism in case the event doesn't arrive
    setTimeout(() => {
      if (!currentInteractionId) {
        logEvent(
          'System',
          'No input_requested event received after marker detection, checking pending interactions'
        );
        checkPendingInteractions();
      }
    }, 3000);
  }

  // Append the content chunk
  const contentNode = document.createTextNode(chunk);
  responseElement.appendChild(contentNode);

  // Auto scroll
  scrollToBottom();
}

// Check for pending interactions (used when event handling fails)
async function checkPendingInteractions() {
  if (currentInteractionId) {
    // Already handling an interaction, skip check
    logEvent('System', 'Skipping pending check - already handling interaction');
    return;
  }

  const apiUrl = apiUrlInput.value;
  try {
    logEvent('System', 'Checking for pending interactions');

    const response = await fetch(`${apiUrl}/ai/human/pending`);
    if (!response.ok) {
      throw new Error(`API responded with status ${response.status}`);
    }

    const data = await response.json();
    const interactions = data.interactions || {};
    const count = Object.keys(interactions).length;

    if (count > 0) {
      logEvent('System', `Found ${count} pending interactions`);
      console.log('Pending interactions:', interactions);

      // Get the first pending interaction
      const interactionId = Object.keys(interactions)[0];
      const details = interactions[interactionId];

      // Ensure we have a valid interaction
      if (!interactionId || interactionId === 'undefined') {
        logEvent('Error', 'Invalid interaction ID from pending interactions');
        return;
      }

      // Handle this interaction
      handleHumanInputRequest({
        interaction_id: interactionId,
        question: details.question || 'No question provided',
        agent: details.agent_id || 'Unknown agent',
      });

      appendSystemMessage(
        `Found pending interaction: ${
          details.question || 'No question provided'
        }`
      );
    } else {
      logEvent('System', 'No pending interactions found');
    }
  } catch (error) {
    logEvent('Error', `Failed to check pending interactions: ${error.message}`);
  }
}

// Submit human response
async function submitHumanResponse() {
  // Make sure we have a current interaction ID
  if (!currentInteractionId) {
    // Try to recover from window property if available
    if (window.lastReceivedInteractionId) {
      currentInteractionId = window.lastReceivedInteractionId;
      logEvent('Debug', `Recovered interaction ID: ${currentInteractionId}`);
    } else {
      appendErrorMessage('No active interaction to respond to');
      return;
    }
  }

  const response = humanAnswer.value.trim();
  if (!response) {
    alert('Please enter a response');
    return;
  }

  const apiUrl = apiUrlInput.value;

  try {
    submitHumanInput.disabled = true;
    submitHumanInput.textContent = 'Sending...';

    // Log the interaction ID and response for debugging
    logEvent(
      'Debug',
      `Submitting response for interaction: ${currentInteractionId}`
    );

    const payload = {
      interaction_id: currentInteractionId,
      response: response,
      metadata: { source: 'ui_test' },
    };

    // Show the details in console
    console.log('Submitting payload:', payload);

    const result = await fetch(`${apiUrl}/ai/human/response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!result.ok) {
      const errorText = await result.text();
      throw new Error(
        `Failed to submit response: ${result.status} - ${errorText}`
      );
    }

    const resultData = await result.json();
    logEvent('Response', resultData);

    // Hide the panel and reset
    humanInputPanel.classList.remove('active');
    appendSystemMessage('Human response submitted: ' + response);

    // Reset UI state
    document.getElementById('marker-status').textContent = 'No';
    document.getElementById('event-status').textContent = 'No';
    document.getElementById('interaction-id').textContent = 'None';
    document.getElementById('event-flow-status').textContent =
      'Response Submitted';

    // Ensure we scroll to see the system message
    scrollToBottom();

    // Clear the current interaction
    currentInteractionId = null;
    pendingHelpMarkerFound = false;
    window.lastReceivedInteractionId = null;

    // Don't clean up immediately - let the agent process the response first
    setTimeout(async () => {
      try {
        await fetch(`${apiUrl}/ai/human/cleanup`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        logEvent('System', 'Cleaned up completed interactions');
      } catch (err) {
        console.warn('Failed to clean up interactions:', err);
      }
    }, 5000); // Increased to 5 seconds
  } catch (error) {
    appendErrorMessage('Error submitting response: ' + error.message);
    logEvent('Error', `Failed to submit response: ${error.message}`);
  } finally {
    submitHumanInput.disabled = false;
    submitHumanInput.textContent = 'Submit Response';
  }
}

// Append a message to the chat
function appendMessage(role, content) {
  const messageContainer = document.createElement('div');
  messageContainer.className =
    role === 'user'
      ? 'message-container user-message'
      : 'message-container assistant-message';

  const messageHeader = document.createElement('div');
  messageHeader.className = 'message-header';
  messageHeader.textContent = role === 'user' ? 'You' : 'Assistant';

  const messageContent = document.createElement('div');
  messageContent.className = 'message';
  messageContent.textContent = content;

  messageContainer.appendChild(messageHeader);
  messageContainer.appendChild(messageContent);

  chatMessages.appendChild(messageContainer);
  scrollToBottom();
}

// Append a system message
function appendSystemMessage(content) {
  const systemMsg = document.createElement('div');
  systemMsg.className = 'system-message';
  systemMsg.textContent = content;
  chatMessages.appendChild(systemMsg);
  scrollToBottom();
}

// Append an error message
function appendErrorMessage(content) {
  const errorMsg = document.createElement('div');
  errorMsg.className = 'error-message';
  errorMsg.textContent = content;
  chatMessages.appendChild(errorMsg);
  scrollToBottom();
}

// Update connection status
function updateConnectionStatus(status) {
  connectionStatus.textContent = status;
  connectionStatus.className =
    status === 'Connected' ? 'connected' : 'disconnected';
}

// Enhanced log event function for better readability and completeness
function logEvent(type, data) {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = document.createElement('div');

  // Add timestamp span with specific styling
  const timestampSpan = document.createElement('span');
  timestampSpan.className = 'event-timestamp';
  timestampSpan.textContent = `[${timestamp}]`;
  logEntry.appendChild(timestampSpan);

  // Add type with specific styling based on event type
  const typeSpan = document.createElement('span');

  // Categorize event types for better visual distinction
  if (type === 'IMPORTANT' || type === 'Marker') {
    typeSpan.className =
      type === 'IMPORTANT' ? 'event-type-important' : 'event-type-marker';
  } else if (type === 'Request') {
    typeSpan.className = 'event-type-request';
  } else if (type === 'Response' || type === 'System') {
    typeSpan.className = 'event-type-response';
  } else if (type === 'Error') {
    typeSpan.className = 'event-type-error';
  }

  typeSpan.textContent = ` ${type}: `;
  logEntry.appendChild(typeSpan);

  // Format the data for display
  let displayData;
  let fullData = data;

  if (typeof data === 'string') {
    displayData = data;
  } else {
    try {
      // Create a simplified view for the log display
      if (data && typeof data === 'object') {
        const simplifiedData = { ...data };

        // Handle content chunks specially
        if (simplifiedData.chunk && typeof simplifiedData.chunk === 'string') {
          if (simplifiedData.chunk.length > 50) {
            simplifiedData.chunk =
              simplifiedData.chunk.substring(0, 50) + '...';
          }
        }

        // Handle specific event types
        if (type === 'ui:human:input_requested') {
          // Highlight the important fields for human input requests
          displayData = `Interaction ID: ${
            data.interaction_id
          }, Question: "${data.question?.substring(0, 100)}${
            data.question?.length > 100 ? '...' : ''
          }"`;
        } else {
          // For other objects, show concise JSON
          displayData = JSON.stringify(simplifiedData, null, 0);
        }
      } else {
        displayData = JSON.stringify(data);
      }
    } catch (e) {
      displayData = '[Object]';
    }
  }

  // Add the data content
  const dataSpan = document.createElement('span');
  dataSpan.textContent = displayData;
  logEntry.appendChild(dataSpan);

  // Add the full object data as a tooltip and expandable section
  if (fullData && typeof fullData === 'object') {
    logEntry.title = 'Click to expand/collapse full details';
    logEntry.style.cursor = 'pointer';

    const detailsDiv = document.createElement('div');
    detailsDiv.className = 'event-details';
    detailsDiv.style.display = 'none';
    detailsDiv.style.padding = '5px';
    detailsDiv.style.marginTop = '5px';
    detailsDiv.style.backgroundColor = '#f8f8f8';
    detailsDiv.style.border = '1px solid #eee';
    detailsDiv.style.borderRadius = '3px';
    detailsDiv.style.whiteSpace = 'pre-wrap';
    detailsDiv.textContent = JSON.stringify(fullData, null, 2);

    logEntry.addEventListener('click', () => {
      detailsDiv.style.display =
        detailsDiv.style.display === 'none' ? 'block' : 'none';
    });

    logEntry.appendChild(detailsDiv);
  }

  // Add to the log
  eventLog.appendChild(logEntry);
  eventLog.scrollTop = eventLog.scrollHeight;

  // Ensure event log container is properly sized
  const eventLogContainer = document.querySelector('.event-log-container');
  if (eventLogContainer) {
    if (eventLog.scrollHeight > eventLog.clientHeight) {
      eventLogContainer.style.minHeight = '180px';
    }
  }
}

// Add this function to clear the log but keep a header
function clearEventLog() {
  eventLog.innerHTML = '';
  logEvent('System', 'Event log cleared');
  logEvent(
    'Info',
    'All events will be logged here. Click on events with objects to expand details.'
  );
}

// Improved scroll function
function scrollToBottom() {
  // First attempt immediate scroll
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Then do another scroll after a short delay to ensure content is rendered
  setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 50);

  // Final scroll after all content should be rendered
  setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 300);
}
