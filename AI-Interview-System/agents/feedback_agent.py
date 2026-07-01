"""
Feedback Agent.
Takes the raw specialist evaluation (from HR/Python/AI agent) for a single
Q&A turn and turns it into final, candidate-facing feedback. Also builds the
end-of-interview summary report and exports it to PDF.
"""
from collections import defaultdict
from utils import get_model, call_json
from fpdf import FPDF


class FeedbackAgent:
    def __init__(self, model_name: str | None = None):
        self.model = get_model(model_name) if model_name else get_model()

    def finalize(self, domain: str, question: str, answer: str, specialist_eval: dict) -> dict:
        prompt = f"""You are the Feedback Agent in an AI interview preparation system.
A specialist {domain} interviewer already evaluated this answer. Your job is to turn
that raw evaluation into encouraging, honest, candidate-facing feedback.

Question: {question}
Candidate answer: "{answer}"
Specialist evaluation (JSON): {specialist_eval}

Return ONLY valid JSON with this shape:
{{
  "correctness": 0,
  "clarity": 0,
  "confidence": 0,
  "technical_depth": 0,
  "overall_score": 0.0,
  "feedback": "2-3 sentences, encouraging but honest, mention what was good and what was missing",
  "missing_points": ["..."]
}}
You may keep the specialist's scores as-is unless you have a strong reason to adjust them.
overall_score should be the average of the four scores, rounded to 1 decimal place.
"""
        result = call_json(self.model, prompt)
        for key in ("correctness", "clarity", "confidence", "technical_depth"):
            result.setdefault(key, specialist_eval.get(key, 5))
        result.setdefault(
            "overall_score",
            round(
                sum(result[k] for k in ("correctness", "clarity", "confidence", "technical_depth")) / 4,
                1,
            ),
        )
        result.setdefault("missing_points", specialist_eval.get("missing_points", []))
        result.setdefault("feedback", specialist_eval.get("notes", ""))
        return result

    @staticmethod
    def compute_summary(transcript: list[dict]) -> dict:
        """Deterministic aggregation across the whole interview - no LLM call,
        so the final numbers are always reliable and reproducible."""
        by_domain = defaultdict(list)
        all_missing = []
        for turn in transcript:
            by_domain[turn["domain"]].append(turn["overall_score"])
            all_missing.extend(turn.get("missing_points", []))

        domain_scores = {d: round(sum(v) / len(v), 1) for d, v in by_domain.items() if v}
        overall = round(sum(domain_scores.values()) / len(domain_scores), 1) if domain_scores else 0.0

        weak_areas = [d for d, score in domain_scores.items() if score < 6]

        # Most frequently missing concepts, deduplicated, capped at 6.
        seen = []
        for point in all_missing:
            if point and point not in seen:
                seen.append(point)
        recommended_topics = seen[:6]

        return {
            "overall_score": overall,
            "domain_scores": domain_scores,
            "weak_areas": weak_areas,
            "recommended_topics": recommended_topics,
        }

    @staticmethod
    def generate_pdf(summary: dict, resume_analysis: dict, transcript: list[dict], path: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AI Interview Preparation Report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Overall Score: {summary['overall_score']} / 10", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Domain Scores", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for domain, score in summary["domain_scores"].items():
            pdf.cell(0, 7, f"  {domain.upper()}: {score} / 10", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Weak Areas", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        if summary["weak_areas"]:
            for area in summary["weak_areas"]:
                pdf.cell(0, 7, f"  - {area}", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 7, "  None - solid performance across domains.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Recommended Topics to Review", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for topic in summary["recommended_topics"]:
            pdf.multi_cell(0, 7, f"  - {topic}")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Per-Question Breakdown", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for i, turn in enumerate(transcript, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, f"{i}. [{turn['domain'].upper()}] {turn['question']}")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, f"   Answer: {turn['answer']}")
            pdf.multi_cell(
                0, 6,
                f"   Score: {turn['overall_score']}/10 | Feedback: {turn['feedback']}"
            )
            pdf.ln(1)

        pdf.output(path)
        return path
