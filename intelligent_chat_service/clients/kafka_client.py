import json
import asyncio
from typing import Dict, Any, Callable, Optional, List
import aiokafka
from utils import logger
import config


class KafkaClient:
    """Client for interacting with Kafka for agent communication"""

    def __init__(self, bootstrap_servers=None):
        self.bootstrap_servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
        self.producer = None
        self.consumer = None
        self.consumer_task = None
        self.running = False

    async def initialize(self):
        """Initialize the Kafka producer"""
        if not self.producer:
            self.producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self.producer.start()
            logger.info(
                f"Kafka producer initialized with servers: {self.bootstrap_servers}"
            )

    async def send_message(
        self, topic: str, message: Dict[str, Any], key: str = None
    ) -> bool:
        """Send a message to a Kafka topic"""
        try:
            if not self.producer:
                await self.initialize()

            encoded_key = key.encode("utf-8") if key else None
            await self.producer.send_and_wait(topic, message, key=encoded_key)
            logger.debug(f"Message sent to topic {topic}: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to Kafka topic {topic}: {str(e)}")
            return False

    async def consume_messages(
        self,
        topics: List[str],
        group_id: str,
        callback: Callable[[str, Dict[str, Any]], None],
        auto_offset_reset: str = "latest",
    ):
        """Consume messages from Kafka topics"""
        try:
            self.consumer = aiokafka.AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset=auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            await self.consumer.start()
            logger.info(f"Kafka consumer started for topics {topics}, group {group_id}")

            self.running = True
            try:
                while self.running:
                    try:
                        async for msg in self.consumer:
                            try:
                                await callback(msg.topic, msg.value)
                            except Exception as e:
                                logger.error(
                                    f"Error processing Kafka message: {str(e)}"
                                )
                    except asyncio.CancelledError:
                        logger.info("Kafka consumer task cancelled")
                        break
            finally:
                await self.stop_consumer()

        except Exception as e:
            logger.error(f"Error in Kafka consumer: {str(e)}")
            if self.consumer:
                await self.stop_consumer()

    def start_consumer(
        self,
        topics: List[str],
        group_id: str,
        callback: Callable[[str, Dict[str, Any]], None],
        auto_offset_reset: str = "latest",
    ):
        """Start consuming messages in a background task"""
        if self.consumer_task:
            logger.warning("Consumer task already running")
            return

        self.consumer_task = asyncio.create_task(
            self.consume_messages(topics, group_id, callback, auto_offset_reset)
        )
        return self.consumer_task

    async def stop_consumer(self):
        """Stop consuming messages"""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None

        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
            self.consumer_task = None

    async def shutdown(self):
        """Shutdown the Kafka client"""
        await self.stop_consumer()
        if self.producer:
            await self.producer.stop()
            self.producer = None
        logger.info("Kafka client shutdown complete")
