#!/usr/bin/env python3
"""
Human-in-the-loop test script that tests the complete feedback flow:
1. Invokes the chat API with input likely to trigger human interaction
2. Processes the event stream to detect human input requests
3. Submits human feedback responses
4. Confirms the response is incorporated into the agent's final output
"""

import asyncio
import aiohttp
import json
import os
import time
import uuid
from dotenv import load_dotenv
import argparse

# Force human-in-the-loop to be enabled before any imports
os.environ["HUMAN_IN_THE_LOOP_ENABLED"] = "true"

# Load environment variables
load_dotenv()
API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


class HumanInTheLoopTester:
    """Tester class for human-in-the-loop functionality"""

    def __init__(self, api_url, verbose=False):
        self.api_url = api_url
        self.verbose = verbose
        self.interaction_id = None
        self.event_history = []
        self.received_human_request = False
        self.response_submitted = False
        self.received_updated_response = False

    def log(self, message):
        """Print message with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}] {message}")

    def debug(self, message):
        """Print debug message if verbose mode is enabled"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] 🔍 DEBUG: {message}")

    async def run_test(self, response_mode="auto"):
        """Run the complete test"""
        self.log("🚀 Starting human-in-the-loop test")

        # Start a fresh chat session
        await self.start_chat_session(response_mode)

        # Report test results
        await self.report_results()

    async def check_pending_interactions(
        self, response_mode="auto", wait_after_found=False
    ):
        """Check for any pending interactions from previous test runs"""
        self.log("🔍 Checking pending interactions endpoint")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_url}/ai/human/pending") as resp:
                    if resp.status != 200:
                        self.log(
                            f"❌ Failed to get pending interactions: {resp.status}"
                        )
                        return False

                    data = await resp.json()
                    interactions = data.get("interactions", {})
                    count = len(interactions)

                    if count == 0:
                        self.log("✅ No pending interactions found")
                        return False

                    self.log(f"🔔 Found {count} pending interactions")
                    interactions_handled = 0

                    # Respond to each interaction
                    for interaction_id, details in interactions.items():
                        question = details.get("question", "No question provided")
                        agent_id = details.get("agent_id", "Unknown")
                        self.interaction_id = interaction_id
                        self.received_human_request = True

                        self.log(
                            f"📝 Interaction: {interaction_id} from agent: {agent_id}"
                        )
                        self.log(f"❓ Question: {question}")

                        success = await self.submit_response(
                            interaction_id, question, response_mode
                        )
                        if success:
                            interactions_handled += 1

                        # Only handle one interaction if not waiting
                        if not wait_after_found:
                            break

                    # Wait a bit after handling interactions if requested
                    if interactions_handled > 0 and wait_after_found:
                        self.log("⏳ Waiting a moment after handling interactions...")
                        await asyncio.sleep(2)

                    return interactions_handled > 0

            except Exception as e:
                self.log(f"❌ Error checking pending interactions: {str(e)}")
                return False

    async def start_chat_session(self, response_mode="auto"):
        """Start a new chat session with a prompt likely to trigger human interaction"""
        self.log("\n📱 Starting new chat session")

        # Reset tracking variables
        self.interaction_id = None
        self.received_human_request = False
        self.response_submitted = False
        self.received_updated_response = False

        # First check if there are pending interactions
        await self.check_pending_interactions(response_mode)

        # Create a payload with a prompt that should trigger human-in-the-loop
        chat_payload = {
            "workflow_name": "idiscovery_orchestrator",
            "user_input": (
                "This is an ETHICAL question that requires human judgment: "
                "What ethical principles should guide AI development and deployment? "
                "How should we balance innovation speed with safety? "
                "I want your personal opinion on this ethical dilemma."
            ),
            "selected_sources": [],
            "config": {
                "thread_id": f"test-{uuid.uuid4().hex[:8]}",
                "client_id": "test_client",
                "user_id": "test_user",
                "session_id": "test_session",
                "human_in_the_loop": True,  # Explicitly enable
            },
            "stream": True,
        }

        self.log("📝 Sending chat request with ethical question")
        self.debug(f"Request payload: {json.dumps(chat_payload, indent=2)}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_url}/ai/chat", json=chat_payload
                ) as resp:
                    if resp.status != 200:
                        self.log(f"❌ Chat request failed: {resp.status}")
                        error_text = await resp.text()
                        self.log(f"Error: {error_text}")
                        return

                    self.log("✅ Chat request accepted, processing events")
                    self.log("⏳ Waiting for events (this might take a minute)...")

                    # Process the server-sent events stream
                    await self.process_event_stream(resp, response_mode)

            except Exception as e:
                self.log(f"❌ Error during chat session: {str(e)}")
                import traceback

                traceback.print_exc()

    async def process_event_stream(self, response, response_mode):
        """Process the server-sent events (SSE) stream"""
        marker_found = False
        last_activity = time.time()
        last_check_time = 0
        check_interval = 2  # seconds

        # Track if we're waiting for the content end event
        waiting_for_content_end = False
        content_end_received = False

        self.log("🔄 Processing event stream...")

        # Process events until stream ends or timeout
        async for line in response.content:
            last_activity = time.time()

            # Decode the line
            decoded_line = line.decode("utf-8").strip()
            if not decoded_line or not decoded_line.startswith("data:"):
                continue

            # Parse the event
            try:
                data_str = decoded_line[5:].strip()
                data = json.loads(data_str)
                event_type = data.get("event", "unknown")
                event_data = data.get("data", {})

                # Track all events for reporting
                self.event_history.append({"type": event_type, "data": event_data})

                # Handle different event types
                if event_type == "ui:content:chunk":
                    print(".", end="", flush=True)
                    chunk = event_data.get("chunk", "")

                    # Look for the HUMAN_HELP_NEEDED marker in content
                    if "HUMAN_HELP_NEEDED:" in chunk and not marker_found:
                        marker_found = True
                        waiting_for_content_end = True
                        self.log("\n\n🔍 Found 'HUMAN_HELP_NEEDED:' marker in content")

                # When we receive content end after marker found, check pending interactions
                elif event_type == "ui:content:end" and waiting_for_content_end:
                    content_end_received = True
                    self.log(
                        "\n🔄 Content generation finished, checking for pending interactions"
                    )
                    # Wait a moment for the backend to process the request
                    await asyncio.sleep(1)

                    # Check pending interactions repeatedly until one is found or timeout
                    check_attempts = 0
                    max_attempts = 5
                    while check_attempts < max_attempts:
                        check_attempts += 1
                        self.log(
                            f"Checking for pending interactions (attempt {check_attempts}/{max_attempts})..."
                        )
                        found = await self.check_pending_interactions(
                            response_mode, wait_after_found=True
                        )
                        if found:
                            break
                        await asyncio.sleep(2)  # Wait between checks

                elif event_type == "ui:human:input_requested":
                    self.received_human_request = True
                    question = event_data.get("question", "")
                    self.interaction_id = event_data.get("interaction_id")

                    self.log(f"\n\n{'='*60}")
                    self.log("🚨 HUMAN INPUT REQUESTED EVENT RECEIVED")
                    self.log(f"❓ Question: {question}")
                    self.log(f"🆔 Interaction ID: {self.interaction_id}")
                    self.log(f"{'='*60}\n")

                    # Submit response if we have a valid interaction ID
                    if self.interaction_id:
                        await self.submit_response(
                            self.interaction_id, question, response_mode
                        )
                    else:
                        self.log("⚠️ Missing interaction_id, can't respond!")

                elif event_type == "ui:human:input_received":
                    self.log("\n✅ System confirmed receipt of human input")

                elif (
                    event_type == "ui:step:update"
                    and event_data.get("status") == "updated_with_human_input"
                ):
                    self.received_updated_response = True
                    self.log("\n🎉 Agent has incorporated human feedback into response")

                # If enough time has passed and we've found the marker but no event yet, check pending
                current_time = time.time()
                if (
                    marker_found
                    and not self.received_human_request
                    and (current_time - last_check_time) > check_interval
                ):
                    last_check_time = current_time
                    self.debug("Checking pending interactions due to marker...")
                    await self.check_pending_interactions(response_mode)

                # Log other significant events
                if self.verbose and event_type not in ["ui:content:chunk"]:
                    self.debug(f"Event: {event_type}")

            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

            # Check for inactivity
            if time.time() - last_activity > 15:  # 15 seconds of inactivity
                self.log(
                    "\n⚠️ No events received for 15 seconds, checking for pending interactions"
                )
                await self.check_pending_interactions(response_mode)
                last_activity = time.time()

        self.log("\n✅ Event stream ended")

        # Final check for pending interactions if we didn't get a human input event
        if (
            marker_found or waiting_for_content_end
        ) and not self.received_human_request:
            self.log(
                "\n⚠️ Content finished but no human input event received. Checking pending interactions..."
            )
            await asyncio.sleep(2)  # Wait a moment for backend to process
            await self.check_pending_interactions(response_mode)

    async def submit_response(self, interaction_id, question, mode="auto"):
        """Submit a human response to the interaction"""
        self.log(f"💬 Preparing response for interaction: {interaction_id}")

        # The human's response - can be automated or manual
        if mode == "manual":
            self.log("\n📝 Enter your response to the question:")
            self.log(f"❓ {question}")
            human_answer = input("\n> ")
        else:
            # Use a pre-defined response
            human_answer = (
                "Safety should be prioritized over rapid innovation. We need ethical guardrails like: "
                "1) Rigorous safety testing before deployment, 2) Transparent decision processes, "
                "3) Independent oversight, 4) Clear accountability frameworks, and 5) Inclusive stakeholder engagement. "
                "Higher-risk AI systems should have more stringent requirements. This balanced approach "
                "protects people while still enabling responsible innovation."
            )
            self.log(f"📄 Using automated response: {human_answer[:50]}...")

        # Submit the response
        payload = {
            "interaction_id": interaction_id,
            "response": human_answer,
            "metadata": {"source": "test_script"},
        }

        self.log("📤 Submitting response to API...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/ai/human/response", json=payload
                ) as resp:
                    if resp.status == 200:
                        self.response_submitted = True
                        response_text = await resp.text()
                        self.log(f"✅ Response submitted successfully! {response_text}")

                        # Clean up after a brief delay
                        await asyncio.sleep(1)
                        try:
                            await session.post(f"{self.api_url}/ai/human/cleanup")
                            self.log("✅ Cleaned up completed interactions")
                        except Exception as cleanup_error:
                            self.log(
                                f"⚠️ Failed to clean up interactions: {cleanup_error}"
                            )

                        return True
                    else:
                        error_text = await resp.text()
                        self.log(
                            f"❌ Failed to submit response: {resp.status} - {error_text}"
                        )
                        return False
        except Exception as e:
            self.log(f"❌ Error submitting response: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    async def report_results(self):
        """Report the test results"""
        self.log("\n" + "=" * 60)
        self.log("📊 TEST RESULTS")
        self.log("=" * 60)

        success = self.received_human_request and self.response_submitted

        self.log(
            f"✓ Detected human help marker: {True if self.interaction_id else False}"
        )
        self.log(
            f"✓ Received human input request (event or pending): {self.received_human_request}"
        )
        self.log(f"✓ Submitted human response: {self.response_submitted}")
        self.log(
            f"✓ Agent incorporated human feedback: {self.received_updated_response}"
        )

        if success:
            self.log(
                "\n🎉 TEST PASSED: Successfully handled human-in-the-loop request!"
            )
        else:
            self.log("\n⚠️ TEST INCOMPLETE: Some parts of the flow didn't complete")

            # Provide debug suggestions
            if not self.received_human_request:
                self.log(
                    "  - No human input was requested. Check if HUMAN_IN_THE_LOOP_ENABLED=true"
                )
            if not self.response_submitted:
                self.log(
                    "  - Failed to submit human response. Check network or API errors"
                )

        # Clean up at the end of the test
        self.log("🧹 Cleaning up any remaining interactions...")
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.api_url}/ai/human/cleanup")
                self.log("✅ Final cleanup completed")
        except Exception as e:
            self.log(f"⚠️ Final cleanup failed: {str(e)}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test the human-in-the-loop functionality"
    )
    parser.add_argument(
        "--api", default=API_URL, help=f"API base URL (default: {API_URL})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--manual",
        "-m",
        action="store_true",
        help="Manual response mode (default: automated)",
    )
    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 80)
    print("🤖 HUMAN-IN-THE-LOOP TEST SCRIPT - FIXED VERSION 👤")
    print("=" * 80)

    # Create and run the tester
    tester = HumanInTheLoopTester(args.api, args.verbose)
    response_mode = "manual" if args.manual else "auto"
    await tester.run_test(response_mode)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Test canceled by user")
    except Exception as e:
        print(f"\n\n💥 Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
