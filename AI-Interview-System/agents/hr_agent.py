from agents.base_agent import DomainAgent


class HRAgent(DomainAgent):
    domain = "hr"
    persona = "an HR interviewer assessing communication, confidence, and behavioral fit"
    eval_emphasis = "clarity and confidence (technical_depth matters less here)"
