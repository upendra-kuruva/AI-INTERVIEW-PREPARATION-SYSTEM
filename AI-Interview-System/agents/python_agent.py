from agents.base_agent import DomainAgent


class PythonAgent(DomainAgent):
    domain = "python"
    persona = "a senior engineer interviewing for Python technical depth"
    eval_emphasis = "correctness and technical_depth"
