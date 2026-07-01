"""
Resume Analyzer Agent
Extracts skills, projects, education, and weak areas from raw resume text.
"""
from utils import get_model, call_json


class ResumeAgent:
    def __init__(self, model_name: str | None = None):
        self.model = get_model(model_name) if model_name else get_model()

    def analyze(self, resume_text: str) -> dict:
        prompt = f"""You are the Resume Analyzer Agent in an AI interview preparation system.

Read the resume text below and extract structured information.

Resume:
---
{resume_text[:8000]}
---

Return ONLY valid JSON with this exact shape:
{{
  "skills": ["skill1", "skill2", ...],
  "projects": ["short project description", ...],
  "education": ["degree / institution", ...],
  "strengths": ["skill or area the candidate is clearly strong in", ...],
  "weak_areas": ["skill commonly expected for this profile but missing or weak", ...]
}}

Guidelines:
- "skills" should include technical skills, tools, and frameworks actually mentioned.
- "weak_areas" should be inferred gaps (e.g. if they list ML projects but no mention of
  statistics, SQL, or system design, flag those) - things a real interviewer would probe.
- Keep each list to at most 8 items.
"""
        result = call_json(self.model, prompt)
        result.setdefault("skills", [])
        result.setdefault("projects", [])
        result.setdefault("education", [])
        result.setdefault("strengths", [])
        result.setdefault("weak_areas", [])
        return result
