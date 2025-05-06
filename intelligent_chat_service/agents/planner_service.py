from .planner_agent import PlannerAgent
from .service_template import create_agent_service

if __name__ == "__main__":
    create_agent_service(PlannerAgent, default_port=8002)
