"""
agent.py — LangGraph-based support assistant for Zepto policy questions.

Architecture:
  StateGraph with three nodes:
    1. classify_intent     — decides if the question is policy-related or general
    2. retrieve_and_answer — RAG: retrieve docs + generate answer (policy questions)
    3. direct_answer       — direct response for non-policy / small-talk queries

  Conditional edge from classify_intent:
    "policy_question"  -> retrieve_and_answer
    "general_question" -> direct_answer

  Environment variable MOCK_LLM:
    - Unset or "1" (default, graded baseline): deterministic mock mode, no LLM call.
    - "0": calls a real LLM (optional, ungraded extension).

Output schema (Pydantic):
    answer:     str
    sources:    list[str]   — chunk IDs for policy_question, empty for general_question
    confidence: float       — 0.0 to 1.0 (fixed 1.0 in mock mode)
"""

import json
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ValidationError

from retriever import retrieve


# ── Helper: MOCK_LLM toggle ──────────────────────────────────────────────────

def _is_mock() -> bool:
    """
    Return True when mock mode is active (the graded baseline).

    Mock mode is the DEFAULT: MOCK_LLM unset or MOCK_LLM=1 → mock.
    Only MOCK_LLM=0 activates the real-LLM path.
    """
    val = os.environ.get("MOCK_LLM", "1")
    return val != "0"


# ── Output schema ─────────────────────────────────────────────────────────────

class AssistantResponse(BaseModel):
    """Structured response returned by the graph."""
    answer:     str
    sources:    list[str]
    confidence: float


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:     str          # The user's query
    intent:    str          # "policy_question" or "general_question"
    chunks:    list[dict]   # Retrieved document chunks (empty for general)
    response:  dict         # Final answer dict matching AssistantResponse schema


# ── Structured prompt template ────────────────────────────────────────────────
# Follows the role–context–task–format–length skeleton.
# Includes a negative constraint and one few-shot example.
# Used only in the optional MOCK_LLM=0 extension path.

STRUCTURED_PROMPT_TEMPLATE = """
## Role
You are a helpful and friendly customer support assistant for Zepto, a quick-commerce grocery delivery app in India. You answer customer queries accurately and concisely.

## Context
The following excerpts are retrieved from Zepto's official policy documents. Use ONLY these excerpts to answer the customer's question.

{context}

## Task
Answer the customer's question below based solely on the provided context.

## Negative Constraint
Do NOT answer using information not present in the provided context. If the context does not contain enough information to fully answer the question, say so explicitly rather than guessing or using outside knowledge.

## Format
Return your answer as a valid JSON object with exactly these fields:
- "answer": a concise, friendly answer string
- "sources": a JSON array of the chunk IDs used (e.g. ["doc_01_chunk_0", "doc_02_chunk_0"])
- "confidence": a float between 0.0 and 1.0 indicating how confident you are

## Length
Keep the answer to 2-4 sentences maximum.

## Few-Shot Example
Customer question: "How long does delivery take?"
Assistant response:
{{"answer": "Zepto delivers grocery and household essentials within 10 to 30 minutes of order confirmation, depending on your delivery zone and current order volume.", "sources": ["doc_01_chunk_0"], "confidence": 0.95}}

## Customer Question
{question}

## Assistant Response (JSON only):
""".strip()

CLASSIFICATION_PROMPT = """
Classify the following customer query as either "policy_question" (if it asks about Zepto's delivery, returns, refunds, membership, tracking, cancellation, gift cards, or support hours) or "general_question" (if it does not).

Query: {question}

Respond with exactly one word: policy_question or general_question
""".strip()

DIRECT_ANSWER_PROMPT = """
You are a friendly customer support assistant for Zepto grocery delivery.
The customer has asked a general question that is not about Zepto's policies.
Respond helpfully and let them know you can help with questions about Zepto's delivery, returns, refunds, membership, tracking, cancellation, gift cards, or support hours.

Customer: {question}

Respond as a JSON object: {{"answer": "...", "sources": [], "confidence": 0.8}}
""".strip()


# ── LLM helper (optional MOCK_LLM=0 path) ────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """
    Call the real LLM. Only used when MOCK_LLM=0.
    Uses Groq's free tier by default; falls back to error message.
    """
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return '[LLM unavailable — set GROQ_API_KEY when using MOCK_LLM=0]'
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


def _parse_llm_response_with_retry(
    prompt: str, chunk_ids: list[str], max_retries: int = 2
) -> AssistantResponse:
    """
    Call the LLM and parse its output as AssistantResponse JSON.
    Retries up to max_retries times with a corrective instruction on failure.
    Returns a clearly marked error response if all retries fail.
    """
    raw = _call_llm(prompt)

    for attempt in range(max_retries + 1):
        try:
            # Try to extract JSON from the raw response
            text = raw.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            parsed = json.loads(text)
            return AssistantResponse(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                corrective_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response was not valid JSON matching the required schema. "
                    f"Error: {e}\n"
                    f"Please respond with ONLY a valid JSON object with fields: "
                    f'"answer" (string), "sources" (list of strings), "confidence" (float 0-1).'
                )
                raw = _call_llm(corrective_prompt)
            else:
                # All retries exhausted — return error response
                return AssistantResponse(
                    answer=f"[Error: LLM output failed schema validation after {max_retries + 1} attempts. Raw: {raw[:200]}]",
                    sources=chunk_ids,
                    confidence=0.0,
                )

    # Should not reach here, but just in case
    return AssistantResponse(
        answer="[Error: unexpected retry loop exit]",
        sources=chunk_ids,
        confidence=0.0,
    )


# ── Node 1: classify_intent ──────────────────────────────────────────────────

# Exact keywords from the spec
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]


def classify_intent(state: AgentState) -> AgentState:
    """
    Decide whether the query is about Zepto policy or a general query.

    Mock mode (default): keyword heuristic — if the lowercased query contains
    any of the spec keywords, classify as policy_question; else general_question.
    No LLM call is made.

    MOCK_LLM=0: call the LLM to classify.
    """
    query = state["query"]

    if _is_mock():
        # Keyword heuristic — graded baseline
        q_lower = query.lower()
        is_policy = any(kw in q_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        # Optional real-LLM classification
        prompt = CLASSIFICATION_PROMPT.format(question=query)
        result = _call_llm(prompt).strip().lower()
        if "policy" in result:
            intent = "policy_question"
        else:
            intent = "general_question"

    return {**state, "intent": intent}


# ── Node 2: retrieve_and_answer ───────────────────────────────────────────────

def retrieve_and_answer(state: AgentState) -> AgentState:
    """
    Retrieve the most relevant policy chunks and produce an answer.

    Retrieval (ChromaDB cosine similarity) ALWAYS runs in both modes.
    Only the answer-generation step branches on MOCK_LLM.
    """
    query = state["query"]

    # Retrieval always runs (embedding + ChromaDB need no API key)
    chunks = retrieve(query)

    if not chunks:
        response = {
            "answer":     "I could not find relevant information in the policy documents.",
            "sources":    [],
            "confidence": 0.0,
        }
        return {**state, "chunks": [], "response": response}

    chunk_ids = [c["id"] for c in chunks]

    if _is_mock():
        # Mock mode (graded baseline): canned template answer
        top_chunk_snippet = chunks[0]["document"][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        response = {
            "answer":     answer,
            "sources":    chunk_ids,
            "confidence": 1.0,
        }
    else:
        # Optional MOCK_LLM=0: call real LLM with structured prompt
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Chunk ID: {c['id']}, Source: {c['source']}]\n{c['document']}")
        context = "\n\n".join(context_parts)

        prompt = STRUCTURED_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        result = _parse_llm_response_with_retry(prompt, chunk_ids)
        response = {
            "answer":     result.answer,
            "sources":    result.sources,
            "confidence": result.confidence,
        }

    return {**state, "chunks": chunks, "response": response}


# ── Node 3: direct_answer ────────────────────────────────────────────────────

def direct_answer(state: AgentState) -> AgentState:
    """
    Handle non-policy questions with a direct, canned response.
    No documents are retrieved.
    """
    query = state["query"]

    if _is_mock():
        # Mock mode (graded baseline): fixed canned string
        response = {
            "answer":     "I can only answer questions about Zepto policies right now.",
            "sources":    [],
            "confidence": 1.0,
        }
    else:
        # Optional MOCK_LLM=0: call real LLM
        prompt = DIRECT_ANSWER_PROMPT.format(question=query)
        result = _parse_llm_response_with_retry(prompt, [])
        response = {
            "answer":     result.answer,
            "sources":    result.sources,
            "confidence": result.confidence,
        }

    return {**state, "chunks": [], "response": response}


# ── Graph construction ────────────────────────────────────────────────────────

def _route_intent(state: AgentState) -> str:
    """Conditional edge: route based on classified intent."""
    return state["intent"]


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent",     classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer",       direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "policy_question":  "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer",       END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return the compiled graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str) -> AssistantResponse:
    """
    Ask the support assistant a question.

    Args:
        question: The customer's question string.

    Returns:
        AssistantResponse with answer, sources, and confidence.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "query":    question,
        "intent":   "",
        "chunks":   [],
        "response": {},
    }
    final_state = graph.invoke(initial_state)
    resp = final_state["response"]
    return AssistantResponse(
        answer=resp["answer"],
        sources=resp["sources"],
        confidence=resp["confidence"],
    )
