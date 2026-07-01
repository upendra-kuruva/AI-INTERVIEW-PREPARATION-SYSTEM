"""
Shared helpers for talking to Gemini and parsing its JSON output safely.
Every agent in this project goes through here so there's exactly one place
that configures the API key and one place that hardens JSON parsing.
"""
import os
import json
import re
import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.5-flash"
#EMBED_MODEL = "models/text-embedding-004"
EMBED_MODEL = "models/gemini-embedding-001"

_configured = False


def configure(api_key: str | None = None):
    """Configure the Gemini SDK once with the given (or env) API key."""
    global _configured
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY env var or enter it in the sidebar."
        )
    genai.configure(api_key=key)
    _configured = True


def get_model(model_name: str = DEFAULT_MODEL):
    if not _configured:
        raise RuntimeError("Gemini not configured yet. Call utils.configure(api_key) first.")
    return genai.GenerativeModel(model_name)


def embed_text(text: str, task_type: str = "retrieval_document"):
    """Returns an embedding vector (list[float]) for the given text."""
    if not _configured:
        raise RuntimeError("Gemini not configured yet. Call utils.configure(api_key) first.")
    result = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    return result["embedding"]


def safe_json_parse(text: str):
    """
    LLMs love wrapping JSON in ```json fences or adding stray prose.
    This strips that noise and tries hard to recover a valid JSON value.
    """
    if text is None:
        raise ValueError("Empty response from model")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first {...} or [...] block in the text.
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"Could not parse JSON from model output: {text[:200]}")


def call_json(model, prompt: str, retries: int = 2):
    """Calls the model expecting a JSON response, retrying once on parse failure."""
    last_err = None
    for _ in range(retries + 1):
        response = model.generate_content(prompt)
        try:
            return safe_json_parse(response.text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            prompt = prompt + "\n\nIMPORTANT: Your last response was not valid JSON. Return ONLY valid JSON, no markdown fences, no commentary."
    raise RuntimeError(f"Model never returned valid JSON: {last_err}")
