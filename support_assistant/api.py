"""
api.py — FastAPI application for the Zepto Support Assistant.

Endpoints:
    GET  /      — health check
    POST /ask   — submit a query, get a structured answer

Run locally (mock mode is the default — no API key needed):
    uvicorn api:app --reload --port 7860

With real LLM (optional, ungraded):
    MOCK_LLM=0 GROQ_API_KEY=your_key uvicorn api:app --reload --port 7860
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import ask, AssistantResponse


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "A RAG-based customer support assistant that answers questions "
        "grounded in Zepto's policy documents. Uses LangGraph for "
        "intent routing and ChromaDB for semantic retrieval."
    ),
    version="1.0.0",
)


# ── Request schema ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is your refund policy for damaged items?"},
                {"query": "How long does delivery take?"},
                {"query": "What is the weather like today?"},
            ]
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def health():
    """Return service status."""
    return {"status": "ok", "service": "Zepto Support Assistant"}


@app.post("/ask", response_model=AssistantResponse, summary="Ask a policy question")
def ask_question(request: QueryRequest) -> AssistantResponse:
    """
    Submit a customer support query.

    The assistant will:
    1. Classify the intent (policy_question vs general_question)
    2. Retrieve relevant policy chunks if policy-related
    3. Generate a grounded answer

    Returns a JSON object with `answer`, `sources`, and `confidence`.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = ask(request.query.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return response
