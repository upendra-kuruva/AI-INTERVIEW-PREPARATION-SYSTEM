from agents.base_agent import DomainAgent


class AIAgent(DomainAgent):
    domain = "ai"
    persona = "an AI/ML interviewer probing applied LLM, RAG, and agent knowledge"
    eval_emphasis = "correctness and technical_depth, with credit for practical project experience"
