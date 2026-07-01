# AI Interview Preparation System

A multi-agent AI interview coach: upload your resume, get personalized
HR / Python / AI-ML interview questions, answer them, and get instant
scored feedback plus a final PDF report.

**Stack:** Google Gemini (LLM + embeddings) · LangGraph (multi-agent routing)
· FAISS (RAG over a question bank) · Streamlit (UI)

## Architecture

```
              User
                │
                ▼
     Streamlit App (app.py)
                │
                ▼
   InterviewCoordinator (coordinator.py)
        │            │
        ▼            ▼
  ResumeAgent    RAG retriever ── FAISS vector store (question bank)
        │            │
        └─────┬──────┘
              ▼
   Question generation (HR / Python / AI agents)
              │
              ▼
   ── per answer ──▶  LangGraph turn graph:
                       START → [route by domain] → HR/Python/AI specialist node
                              → Feedback Agent node → END
              │
              ▼
        Transcript → FeedbackAgent.compute_summary() → Final report + PDF
```

The LangGraph graph (`coordinator.py::build_turn_graph`) is the literal
multi-agent system: a router sends each answer to the right domain
specialist, which produces a raw evaluation; that evaluation is then passed
to the Feedback Agent node, which produces the final candidate-facing score
and feedback. One graph invocation = one fully evaluated Q&A turn.

## Setup

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Paste your Gemini API key into the sidebar, upload a resume, and go.

(Optional) Instead of pasting the key each time, copy `.env.example` to
`.env` and set `GEMINI_API_KEY`, or `export GEMINI_API_KEY=...` before
running Streamlit.

## Project structure

```
AI-Interview-System/
├── app.py                  # Streamlit UI - the whole user-facing flow
├── coordinator.py          # LangGraph turn-graph + InterviewCoordinator
├── utils.py                # Gemini config, JSON-safe LLM calls
├── agents/
│   ├── resume_agent.py     # Extracts skills/strengths/weak areas from resume
│   ├── base_agent.py       # Shared logic for the 3 domain agents below
│   ├── hr_agent.py         # Behavioral questions
│   ├── python_agent.py     # Python technical questions
│   ├── ai_agent.py         # AI/ML technical questions
│   └── feedback_agent.py   # Aggregates scores, builds report, exports PDF
├── rag/
│   ├── embeddings.py       # Gemini embedding calls
│   ├── vector_store.py     # FAISS index build/save/load/search
│   └── retriever.py        # Resume skills -> relevant question bank entries
├── data/
│   ├── question_bank.json  # Seed questions (HR/Python/AI, easy/medium/hard)
│   └── vector_index/       # Auto-created FAISS cache (gitignore this)
└── requirements.txt
```

## How personalization works (RAG)

The question bank (`data/question_bank.json`) is embedded once into a FAISS
index. When you upload a resume, each domain agent retrieves the most
relevant seed questions for your specific skills, then asks Gemini to
generate new questions inspired by (not copied from) those seeds **and**
tailored to what's actually in your resume - e.g. if you mention using
Gemini for a Q&A project, you'll get asked "why Gemini over GPT?" instead
of a generic RAG question.

## Adaptive difficulty

After each answer, `InterviewCoordinator._adaptive_reorder` nudges a
harder not-yet-asked question of the same domain forward if you scored
well (≥8), or an easier one forward if you struggled (≤4).

## Extending it

- **Add a new domain agent:** subclass `DomainAgent` in `agents/base_agent.py`
  (see `hr_agent.py` for the ~3-line pattern), add it to
  `InterviewCoordinator.domain_agents` in `coordinator.py`, and add matching
  entries to `data/question_bank.json`.
- **Swap the LLM provider:** everything funnels through `utils.py` - swap
  `genai.GenerativeModel` / `genai.embed_content` for another provider's SDK
  there and nothing else needs to change.
- **Voice mode:** pipe `st.audio_input` through a speech-to-text call before
  `coordinator.submit_answer`, and TTS the `feedback` string back.
