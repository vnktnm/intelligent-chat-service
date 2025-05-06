from dotenv import load_dotenv
import os

load_dotenv()

# fastapi
SERVICE_HOST = os.environ.get("SERVICE_HOST", "localhost")
SERVICE_PORT = os.environ.get("SERVICE_PORT", 8000)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "True").lower() == "true"
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "logs/app.log")
LOG_FILE_ROTATION = os.environ.get("LOG_FILE_ROTATION", "5 MB")
LOG_FILE_RETENTION = os.environ.get("LOG_FILE_RETENTION", "5")
LOG_FILE_COMPRESSION = os.environ.get("LOG_FILE_COMPRESSION", "zip")

# openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in the environment variables.")
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_DEFAULT_TEMPERATURE = float(os.environ.get("OPENAI_DEFAULT_TEMPERATURE", 0.7))
OPENAI_TOOL_CHOICE = os.environ.get("OPENAI_TOOL_CHOICE", "auto")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", 1000))

# debug
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# database
MONGO_USER = os.environ.get("MONGO_USER", "admin")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "password")
MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_PORT = int(os.environ.get("MONGO_PORT", 27017))
MONGO_AUTH_MECH = os.environ.get("MONGO_AUTH_MECH", "PLAIN")
MONGO_REPLICA_SET = os.environ.get("MONGO_REPLICA_SET", "rs0")
MONGO_READ_PREFERENCE = os.environ.get("MONGO_READ_PREFERENCE", "primary")
MONGO_DATABASE_NAME = os.environ.get("MONGO_DATABASE_NAME", "intelligent_chat_service")
MONGO_ROOT_CHAIN = os.environ.get("MONGO_ROOT_CHAIN", "root_chain.pem")

# Agent service URLs
ANALYZER_AGENT_URL = os.environ.get("ANALYZER_AGENT_URL", "http://localhost:8001")
PLANNER_AGENT_URL = os.environ.get("PLANNER_AGENT_URL", "http://localhost:8002")

# MongoDB tool collection
MONGO_TOOL_COLLECTION_NAME = os.environ.get("MONGO_TOOL_COLLECTION_NAME", "tools")

# Prompts
PROMPT_PATH = os.environ.get(
    "PROMPT_PATH",
    "/home/venkatnm94/prototypes/agentic-ai/intelligent-chat-service/intelligent_chat_service/prompts/prompts.yaml",
)
PROMPT_AGENT_TYPE = os.environ.get("PROMPT_AGENT_TYPE", "agents")
PROMPT_ANALYZER_AGENT = os.environ.get("PROMPT_ANALYZER_AGENT", "analyzer")
PROMPT_CLARIFICATION_AGENT = os.environ.get(
    "PROMPT_CLARIFICATION_AGENT", "clarification"
)
PROMPT_EXECUTOR_AGENT = os.environ.get("PROMPT_EXECUTOR_AGENT", "executor")
PROMPT_PLANNER_AGENT = os.environ.get("PROMPT_PLANNER_AGENT", "planner")

# Tools
TOOL_QDRANT = os.environ.get("TOOL_QDRANT", "tools-qdrant")
AVAILABLE_TOOLS = os.environ.get("AVAILABLE_TOOLS", None)

# Human in the loop configuration
HITL_ENABLED = os.environ.get("HITL_ENABLED", "False").lower() == "true"
HITL_RESPONSE_TIMEOUT = int(os.environ.get("HITL_RESPONSE_TIMEOUT", "300"))  # seconds
HITL_MAX_ROUNDS = int(os.environ.get("HITL_MAX_ROUNDS", "2"))

# Kafka configuration
USE_KAFKA_FOR_AGENTS = os.environ.get("USE_KAFKA_FOR_AGENTS", "False").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_REQUEST_TOPIC_PREFIX = os.environ.get(
    "KAFKA_REQUEST_TOPIC_PREFIX", "agent-request-"
)
KAFKA_RESPONSE_TOPIC = os.environ.get("KAFKA_RESPONSE_TOPIC", "agent-responses")
KAFKA_REQUEST_TIMEOUT = float(os.environ.get("KAFKA_REQUEST_TIMEOUT", "60.0"))
KAFKA_PROCESS_ALL_MESSAGES = (
    os.environ.get("KAFKA_PROCESS_ALL_MESSAGES", "False").lower() == "true"
)
