#!/usr/bin/env python3
"""
Script to test Kafka setup using producer and consumer functionality.
"""

import json
import time
from datetime import datetime
import argparse
from typing import Dict, Any

from kafka import KafkaProducer, KafkaConsumer


def create_producer(bootstrap_servers: str = "localhost:29092"):
    """Create a Kafka producer instance."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def create_consumer(
    topic: str, bootstrap_servers: str = "localhost:29092", group_id: str = "test-group"
):
    """Create a Kafka consumer instance."""
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )


def produce_messages(producer: KafkaProducer, topic: str, num_messages: int = 5):
    """Produce test messages to the specified topic."""
    for i in range(num_messages):
        message = {
            "message_id": i,
            "text": f"Test message {i}",
            "timestamp": datetime.now().isoformat(),
        }

        future = producer.send(topic, value=message, key=str(i))
        print(f"Producing message: {message}")

        # Block until the message is sent (optional)
        future.get(timeout=10)
        time.sleep(0.5)  # Small delay between messages

    producer.flush()
    print(f"Successfully produced {num_messages} messages to topic: {topic}")


def consume_messages(consumer: KafkaConsumer, limit: int = 5, timeout: int = 10):
    """Consume messages from the subscribed topic with a timeout."""
    count = 0
    start_time = time.time()

    print(f"Starting to consume messages (timeout: {timeout}s, limit: {limit})...")

    while count < limit and time.time() - start_time < timeout:
        for message in consumer:
            print(f"Consumed message {count}:")
            print(f"  Topic: {message.topic}")
            print(f"  Partition: {message.partition}")
            print(f"  Offset: {message.offset}")
            print(f"  Key: {message.key.decode('utf-8') if message.key else None}")
            print(f"  Value: {message.value}")
            print("-" * 50)

            count += 1
            if count >= limit:
                break

    if count == 0:
        print("No messages consumed within the timeout period.")
    else:
        print(f"Successfully consumed {count} messages.")


def run_producer_test(args):
    """Run the producer test."""
    producer = create_producer(args.bootstrap_servers)
    try:
        produce_messages(producer, args.topic, args.num_messages)
    finally:
        producer.close()


def run_consumer_test(args):
    """Run the consumer test."""
    consumer = create_consumer(args.topic, args.bootstrap_servers, args.group_id)
    try:
        consume_messages(consumer, args.num_messages, args.timeout)
    finally:
        consumer.close()


def main():
    """Parse arguments and run the appropriate test."""
    parser = argparse.ArgumentParser(description="Test Kafka setup")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Producer command
    producer_parser = subparsers.add_parser("produce", help="Produce messages to Kafka")
    producer_parser.add_argument(
        "--bootstrap-servers", default="localhost:29092", help="Kafka bootstrap servers"
    )
    producer_parser.add_argument(
        "--topic", default="test-topic", help="Topic to produce messages to"
    )
    producer_parser.add_argument(
        "--num-messages", type=int, default=5, help="Number of messages to produce"
    )

    # Consumer command
    consumer_parser = subparsers.add_parser(
        "consume", help="Consume messages from Kafka"
    )
    consumer_parser.add_argument(
        "--bootstrap-servers", default="localhost:29092", help="Kafka bootstrap servers"
    )
    consumer_parser.add_argument(
        "--topic", default="test-topic", help="Topic to consume messages from"
    )
    consumer_parser.add_argument(
        "--group-id", default="test-group", help="Consumer group ID"
    )
    consumer_parser.add_argument(
        "--num-messages",
        type=int,
        default=5,
        help="Maximum number of messages to consume",
    )
    consumer_parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout in seconds"
    )

    args = parser.parse_args()

    if args.command == "produce":
        run_producer_test(args)
    elif args.command == "consume":
        run_consumer_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
