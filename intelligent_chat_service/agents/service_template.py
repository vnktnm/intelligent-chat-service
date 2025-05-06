import argparse
import os
import sys
from typing import Type
import asyncio
import signal

from .agent import Agent
from .kafka_agent_adapter import KafkaAgentAdapter
import config
from utils import logger


def create_agent_service(agent_class: Type[Agent], default_port: int = 8000):
    """
    Creates and runs a service for the specified agent class
    """
    parser = argparse.ArgumentParser(description=f"{agent_class.__name__} Service")
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("PORT", default_port),
        help="Port to run the service on",
    )
    parser.add_argument(
        "--kafka",
        action="store_true",
        default=os.environ.get("USE_KAFKA", "").lower() == "true"
        or config.USE_KAFKA_FOR_AGENTS,
        help="Enable Kafka communication (default: depends on configuration)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        default=True,
        help="Enable HTTP API (default: True)",
    )
    args = parser.parse_args()

    # Determine agent name from class name
    agent_name = agent_class.__name__.lower().replace("agent", "")

    # Start Kafka adapter if enabled
    kafka_adapter = None
    if args.kafka:
        logger.info(f"Starting {agent_name} with Kafka enabled")
        kafka_adapter = KafkaAgentAdapter(agent_class, agent_name)

        # Setup async event loop for Kafka if we're only using Kafka
        if not args.http:
            loop = asyncio.get_event_loop()

            async def start_kafka():
                await kafka_adapter.start()

                # Setup signal handlers for graceful shutdown
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(
                        sig, lambda: asyncio.create_task(shutdown(kafka_adapter))
                    )

                # Keep the service running
                while True:
                    await asyncio.sleep(1)

            async def shutdown(adapter):
                logger.info("Shutting down Kafka agent service...")
                await adapter.stop()
                loop.stop()

            try:
                loop.run_until_complete(start_kafka())
                loop.run_forever()
            finally:
                loop.close()
                logger.info("Kafka agent service shutdown complete")

    # Start HTTP API if enabled
    if args.http:
        # Create the agent service
        app = Agent.create_service(agent_class)

        # Add Kafka startup/shutdown if both are enabled
        if args.kafka:

            @app.on_event("startup")
            async def startup_event():
                await kafka_adapter.start()

            @app.on_event("shutdown")
            async def shutdown_event():
                await kafka_adapter.stop()

        # Run the service
        Agent.run_service(app, port=args.port)
