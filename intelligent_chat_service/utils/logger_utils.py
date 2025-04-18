import sys
import json
import os
from datetime import datetime
from loguru import logger
import logging
import config
import contextvars
import socket

# Create context variables to store request, thread, and session IDs
request_id_var = contextvars.ContextVar("request_id", default=None)
thread_id_var = contextvars.ContextVar("thread_id", default=None)
session_id_var = contextvars.ContextVar("session_id", default=None)


def set_request_id(request_id):
    """Set the request ID in the current context"""
    request_id_var.set(request_id)


def get_request_id():
    """Get the request ID from the current context"""
    return request_id_var.get()


def set_thread_id(thread_id):
    """Set the thread ID in the current context"""
    thread_id_var.set(thread_id)


def get_thread_id():
    """Get the thread ID from the current context"""
    return thread_id_var.get()


def set_session_id(session_id):
    """Set the session ID in the current context"""
    session_id_var.set(session_id)


def get_session_id():
    """Get the session ID from the current context"""
    return session_id_var.get()


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging and redirects to loguru
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            # Convert levelname to uppercase to handle case-insensitivity
            level = logger.level(record.levelname.upper()).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class JsonSink:
    """
    Custom sink for Loguru that formats logs as JSON for DataDog compatibility via FluentBit
    """

    def __init__(self, serialize=True, file_path=None):
        self.serialize = serialize
        self.file_path = file_path
        self.hostname = socket.gethostname()
        self.service_name = getattr(config, "SERVICE_NAME", "intelligent-chat-service")
        self.env = getattr(config, "ENVIRONMENT", "development")

    def __call__(self, message):
        record = message.record

        # Base log data with DataDog compatible structure
        log_data = {
            "timestamp": datetime.utcfromtimestamp(
                record["time"].timestamp()
            ).isoformat()
            + "Z",
            "level": record["level"].name,
            "message": record["message"],
            "process_id": record["process"].id,
            "thread_name": record["thread"].name,
            # DataDog specific attributes
            "hostname": self.hostname,
            "service": self.service_name,
            "ddsource": "python",
            "ddtags": f"env:{self.env}",
        }

        # Include context IDs if available
        request_id = get_request_id()
        if request_id is not None:
            log_data["request_id"] = request_id
            # Add as a DataDog trace ID if appropriate
            log_data["dd.trace_id"] = request_id

        thread_id = get_thread_id()
        if thread_id is not None:
            log_data["chat_thread_id"] = thread_id

        session_id = get_session_id()
        if session_id is not None:
            log_data["session_id"] = session_id

        # Include exception info if available
        if record["exception"]:
            log_data["error.message"] = str(record["exception"])
            log_data["error.stack"] = record["exception"].traceback

        # Include any extra attributes
        for key, value in record["extra"].items():
            log_data[key] = value

        formatted_log = json.dumps(log_data) if self.serialize else str(log_data)

        if self.file_path:
            with open(self.file_path, "a") as f:
                f.write(formatted_log + "\n")
        else:
            print(formatted_log)


def setup_logging(
    json_logs=True,
    log_level="INFO",
    intercept_std_logging=True,
    log_to_file=False,
    log_file_path=None,
    log_file_rotation="20 MB",
    log_file_retention=5,
    log_file_compression="zip",
):
    """
    Configure logging with Loguru

    Args:
        json_logs (bool): Whether to output logs in JSON format
        log_level (str): Minimum log level to display
        intercept_std_logging (bool): Whether to intercept standard Python logging
        log_to_file (bool): Whether to write logs to a file
        log_file_path (str): Path to the log file (if None, logs/{timestamp}.log will be used)
        log_file_rotation (str): When to rotate the log file (e.g., "20 MB", "1 day")
        log_file_retention (int): Number of log files to retain
        log_file_compression (str): Compression format for rotated log files
    """
    # Remove default loguru handler
    logger.remove()

    # Ensure log_level is uppercase to prevent errors with loguru
    if isinstance(log_level, str):
        log_level = log_level.upper()

    # Configure JSON logging if requested
    if json_logs:
        # Always add the console handler
        logger.configure(handlers=[{"sink": JsonSink(), "level": log_level}])

        # Add file handler if requested
        if log_to_file:
            if log_file_path is None:
                # Create logs directory if it doesn't exist
                os.makedirs("logs", exist_ok=True)
                log_file_path = f"logs/app.log"

            logger.add(
                JsonSink(file_path=log_file_path),
                level=log_level,
                enqueue=True,  # Makes logging thread-safe
            )
    else:
        # For non-JSON logs, use the standard loguru format
        # Console handler
        logger.configure(
            handlers=[
                {
                    "sink": sys.stderr,
                    "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> - <level>{message}</level>",
                    "level": log_level,
                }
            ]
        )

        # Add file handler if requested
        if log_to_file:
            if log_file_path is None:
                # Create logs directory if it doesn't exist
                os.makedirs("logs", exist_ok=True)
                log_file_path = f"logs/app.log"

            logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} - {message}",
                level=log_level,
                rotation=log_file_rotation,
                retention=log_file_retention,
                compression=log_file_compression,
                enqueue=True,  # Makes logging thread-safe
                backtrace=True,  # Adds traceback to errors
            )

    # Intercept standard logging if requested
    if intercept_std_logging:
        # Intercept standard logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Return configured logger
    return logger


# Create a global logger instance with file logging
# The defaults here can be overridden by environment variables or configuration files in a real application
logger = setup_logging(
    json_logs=True,
    log_level=(
        config.LOG_LEVEL.upper()
        if isinstance(config.LOG_LEVEL, str)
        else config.LOG_LEVEL
    ),
    intercept_std_logging=True,
    log_to_file=config.LOG_TO_FILE,  # Enable file logging by default
    log_file_path=config.LOG_FILE_PATH,  # Use default path if not specified
    log_file_rotation=config.LOG_FILE_ROTATION,  # Rotate at 20MB by default
    log_file_retention=int(config.LOG_FILE_RETENTION),  # Keep 5 files by default
    log_file_compression=config.LOG_FILE_COMPRESSION,
)
