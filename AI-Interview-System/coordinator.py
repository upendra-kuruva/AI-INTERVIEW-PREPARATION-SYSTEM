"""
Interview Coordinator Agent.

Two responsibilities:
1. A LangGraph graph that, for a single Q&A turn, routes to the right
   domain specialist (HR / Python / AI) and then to the Feedback Agent.
   This is the literal "multi-agent system" piece.
2. A higher-level InterviewCoordinator class that drives the whole session:
   resume analysis -> question generation (RAG-grounded) -> turn-by-turn
   evaluation via the graph -> final report.
"""
import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

import utils
from agents.resume_agent import ResumeAgent
from agents.hr_agent import HRAgent
from agents.python_agent import PythonAgent
from agents.ai_agent import AIAgent
from agents.feedback_agent import FeedbackAgent
from rag.vector_store import load_or_build_store
from rag.retriever import retrieve_relevant_questions

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_CACHE_DIR = os.path.join(DATA_DIR, "vector_index")
QUESTION_BANK_PATH = os.path.join(DATA_DIR, "question_bank.json")

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class TurnState(TypedDict):
    domain: str
    question: str
    key_points: list
    answer: str
    specialist_eval: Optional[dict]
    final_eval: Optional[dict]


def _route(state: TurnState) -> str:
    return state["domain"]


def build_turn_graph(domain_agents: dict, feedback_agent: FeedbackAgent):
    graph = StateGraph(TurnState)

    def make_specialist_node(agent):
        def node(state: TurnState) -> TurnState:
            evaluation = agent.evaluate_answer(state["question"], state["answer"], state["key_points"])
            return {"specialist_eval": evaluation}
        return node

    for domain, agent in domain_agents.items():
        graph.add_node(f"{domain}_node", make_specialist_node(agent))

    def feedback_node(state: TurnState) -> TurnState:
        final = feedback_agent.finalize(state["domain"], state["question"], state["answer"], state["specialist_eval"])
        return {"final_eval": final}

    graph.add_node("feedback_node", feedback_node)

    graph.add_conditional_edges(
        START, _route, {domain: f"{domain}_node" for domain in domain_agents}
    )
    for domain in domain_agents:
        graph.add_edge(f"{domain}_node", "feedback_node")
    graph.add_edge("feedback_node", END)

    return graph.compile()


class InterviewCoordinator:
    def __init__(self, api_key: str):
        utils.configure(api_key)

        self.resume_agent = ResumeAgent()
        self.domain_agents = {
            "hr": HRAgent(),
            "python": PythonAgent(),
            "ai": AIAgent(),
        }
        self.feedback_agent = FeedbackAgent()
        self.graph = build_turn_graph(self.domain_agents, self.feedback_agent)
        self.vector_store = load_or_build_store(QUESTION_BANK_PATH, VECTOR_CACHE_DIR)

        self.resume_analysis: dict = {}
        self.queue: list[dict] = []
        self.index: int = 0
        self.transcript: list[dict] = []

    # ---- session lifecycle ----

    def start(self, resume_text: str, num_per_domain: int = 4) -> dict:
        self.resume_analysis = self.resume_agent.analyze(resume_text)
        skills = self.resume_analysis.get("skills", [])

        per_domain_questions = {}
        for domain, agent in self.domain_agents.items():
            context = retrieve_relevant_questions(self.vector_store, skills, domain, top_k=4)
            per_domain_questions[domain] = agent.generate_questions(skills, context, num=num_per_domain)
            for q in per_domain_questions[domain]:
                q.setdefault("domain", domain)

        # Interleave domains round-robin, sorted within each domain by difficulty,
        # so the interview alternates HR/Python/AI rather than blocking by domain.
        for domain in per_domain_questions:
            per_domain_questions[domain].sort(key=lambda q: DIFFICULTY_ORDER.get(q.get("difficulty", "medium"), 1))

        queue = []
        domains_cycle = list(self.domain_agents.keys())
        max_len = max(len(v) for v in per_domain_questions.values())
        for i in range(max_len):
            for domain in domains_cycle:
                lst = per_domain_questions[domain]
                if i < len(lst):
                    queue.append(lst[i])

        self.queue = queue
        self.index = 0
        self.transcript = []
        return self.resume_analysis

    def total_questions(self) -> int:
        return len(self.queue)

    def is_finished(self) -> bool:
        return self.index >= len(self.queue)

    def current_question(self) -> Optional[dict]:
        if self.is_finished():
            return None
        return self.queue[self.index]

    def submit_answer(self, answer: str) -> dict:
        """Evaluates the current question's answer via the LangGraph multi-agent
        pipeline. Does NOT advance the index - call advance() after showing feedback."""
        q = self.current_question()
        if q is None:
            raise RuntimeError("No current question to answer.")

        state: TurnState = {
            "domain": q["domain"],
            "question": q["question"],
            "key_points": q.get("key_points", []),
            "answer": answer,
            "specialist_eval": None,
            "final_eval": None,
        }
        result = self.graph.invoke(state)
        final_eval = result["final_eval"]

        entry = {
            "domain": q["domain"],
            "question": q["question"],
            "difficulty": q.get("difficulty", "medium"),
            "answer": answer,
            **final_eval,
        }
        self.transcript.append(entry)
        self._adaptive_reorder(q["domain"], final_eval["overall_score"])
        return entry

    def advance(self):
        self.index += 1

    def _adaptive_reorder(self, domain: str, overall_score: float):
        """Light adaptive-difficulty touch: if the candidate is doing well,
        pull a harder not-yet-asked question of the same domain forward;
        if struggling, pull an easier one forward instead."""
        remaining = self.queue[self.index + 1:]
        if not remaining:
            return
        target_difficulty = "hard" if overall_score >= 8 else ("easy" if overall_score <= 4 else None)
        if not target_difficulty:
            return
        for offset, q in enumerate(remaining):
            if q["domain"] == domain and q.get("difficulty") == target_difficulty:
                actual_idx = self.index + 1 + offset
                self.queue.insert(self.index + 1, self.queue.pop(actual_idx))
                break

    def finalize(self) -> dict:
        summary = self.feedback_agent.compute_summary(self.transcript)
        return {
            "summary": summary,
            "resume_analysis": self.resume_analysis,
            "transcript": self.transcript,
        }

    def export_pdf(self, path: str) -> str:
        report = self.finalize()
        return self.feedback_agent.generate_pdf(
            report["summary"], report["resume_analysis"], report["transcript"], path
        )
