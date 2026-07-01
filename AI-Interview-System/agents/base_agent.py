"""
Base class for the three domain specialist agents (HR, Python, AI/ML).
Each subclass just supplies a persona + which evaluation dimensions
it should weigh most heavily - the prompting and JSON contract stay shared,
which is also what makes them easy to wire into LangGraph nodes uniformly.
"""
from utils import get_model, call_json


class DomainAgent:
    domain: str = "generic"
    persona: str = "an interviewer"
    eval_emphasis: str = "correctness and clarity"

    def __init__(self, model_name: str | None = None):
        self.model = get_model(model_name) if model_name else get_model()

    def generate_questions(self, skills: list[str], context_questions: list[dict], num: int = 4) -> list[dict]:
        bank_str = "\n".join(
            f"- ({q.get('difficulty','medium')}) {q['question']}" for q in context_questions
        ) or "(no reference questions available, generate from scratch)"

        prompt = f"""You are {self.persona} in an AI interview preparation system.

Candidate's resume skills: {', '.join(skills) if skills else 'general candidate'}

Here are some reference questions from our question bank for this domain:
{bank_str}

Generate {num} interview questions for this candidate, personalized to their actual
skills/projects where possible (e.g. if they mention a specific tool, ask about it
instead of a generic question). Include a mix of difficulties.

Return ONLY valid JSON, a list of exactly {num} objects with this shape:
[
  {{"question": "...", "difficulty": "easy|medium|hard", "key_points": ["point1", "point2", "point3"]}}
]
"""
        questions = call_json(self.model, prompt)
        for q in questions:
            q["domain"] = self.domain
            q.setdefault("key_points", [])
            q.setdefault("difficulty", "medium")
        return questions

    def evaluate_answer(self, question: str, answer: str, key_points: list[str]) -> dict:
        prompt = f"""You are {self.persona}, evaluating a candidate's spoken/written answer.
When scoring, weigh {self.eval_emphasis} most heavily for this domain, but still score
every dimension below.

Question: {question}
Reference points a strong answer would cover: {', '.join(key_points) if key_points else 'use your judgement'}
Candidate's answer: "{answer}"

Score each dimension 0-10 (integers):
- correctness: factual/technical accuracy
- clarity: how clearly the answer was communicated
- confidence: how confident/assured the answer reads
- technical_depth: depth of understanding shown

Also list which reference points were missing from the answer.

Return ONLY valid JSON:
{{
  "correctness": 0,
  "clarity": 0,
  "confidence": 0,
  "technical_depth": 0,
  "missing_points": ["..."],
  "notes": "one short sentence on what was good or missing"
}}
"""
        return call_json(self.model, prompt)
