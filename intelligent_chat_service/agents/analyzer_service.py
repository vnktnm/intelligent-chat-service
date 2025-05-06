from .analyzer_agent import AnalyzerAgent
from .service_template import create_agent_service

if __name__ == "__main__":
    create_agent_service(AnalyzerAgent, default_port=8001)
