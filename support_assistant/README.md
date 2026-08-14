# Support Assistant — Zepto Data & AI Platform (Module 3)

## Overview

This module implements a **RAG (Retrieval-Augmented Generation)** customer support assistant for Zepto, grounded in 8 official policy documents. It uses a **LangGraph** `StateGraph` for intent routing and answer orchestration, **ChromaDB** with **all-MiniLM-L6-v2** embeddings for semantic retrieval, and a **FastAPI** endpoint to serve queries.

**All grading runs with the default `MOCK_LLM` mode** (unset or `MOCK_LLM=1`) — fully deterministic, no API key, no network call.

---

## Quick Start

```bash
cd support_assistant

# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest policy documents into ChromaDB
python ingest.py

# 3. Start the API server (mock mode — default, no API key needed)
python -m uvicorn api:app --port 7860

# 4. Test it
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your delivery policy?"}'
```

---

## RAG Pipeline Architecture

### Pipeline Stages

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        RAG Pipeline Flow                           │
 ├─────────────────────────────────────────────────────────────────────┤
 │                                                                     │
 │  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌────────────┐  │
 │  │ INGEST   │───▶│ EMBEDDING │───▶│ RETRIEVAL │───▶│ GENERATION │  │
 │  │ ingest.py│    │ ChromaDB  │    │retriever. │    │ agent.py   │  │
 │  │          │    │ ONNX EF   │    │   py      │    │ (3 nodes)  │  │
 │  └──────────┘    └───────────┘    └───────────┘    └────────────┘  │
 │   docs/*.txt      all-MiniLM       cosine sim       MOCK_LLM      │
 │   8 files         -L6-v2           top-3 chunks     branching      │
 └─────────────────────────────────────────────────────────────────────┘
```

### Stage-by-Stage Breakdown

#### 1. Ingestion (`ingest.py`)

- **What it does**: Reads all 8 `.txt` files from `docs/`, splits each into overlapping character-level chunks (500 chars, 50 overlap), and stores them in ChromaDB.
- **Function**: `ingest()` and `_chunk_text()`
- **Output**: 14 chunks from 8 documents stored in the `zepto_policies` ChromaDB collection at `chroma_db/`.
- **MOCK_LLM impact**: None — ingestion is always real.

#### 2. Embedding (ChromaDB `DefaultEmbeddingFunction`)

- **What it does**: Both at ingest time and at query time, text is embedded using **all-MiniLM-L6-v2** via ChromaDB's built-in ONNX Runtime backend (`DefaultEmbeddingFunction`).
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` — runs locally, no API key.
- **MOCK_LLM impact**: None — embedding always runs locally.

#### 3. Retrieval (`retriever.py`)

- **What it does**: Embeds the incoming query and performs cosine-similarity search against the ChromaDB collection, returning the top-3 most similar chunks with their IDs, text, source filenames, and distances.
- **Function**: `retrieve(query, top_k=3)`
- **MOCK_LLM impact**: None — retrieval always runs for real in both modes.

#### 4. Generation (`agent.py` — LangGraph StateGraph)

- **What it does**: A 3-node `StateGraph` classifies intent, routes the query, and produces a structured answer.
- **MOCK_LLM branching**: This is the **only stage that branches on MOCK_LLM**:
  - **Default (mock, MOCK_LLM unset or "1")**: All three nodes use deterministic, rule-based logic — no LLM call is ever made.
  - **Optional (MOCK_LLM=0)**: The generation steps inside each node call a real LLM via Groq's free tier.

### Data Flow

```
User query (string)
  │
  ▼
┌───────────────────────────┐
│ POST /ask {"query": ...}  │  ← api.py (FastAPI)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│     classify_intent       │  ← agent.py: Node 1
│                           │
│  Mock: keyword heuristic  │  Keywords: delivery, return, refund,
│  Real: LLM classification │  membership, tracking, cancel,
│                           │  gift card, support hours
│  Output: intent =         │
│    "policy_question" or   │
│    "general_question"     │
└────────────┬──────────────┘
             │
     ┌───────┴────────┐   conditional edge (_route_intent)
     │                │
     ▼                ▼
┌────────────┐  ┌─────────────┐
│ retrieve_  │  │ direct_     │
│ and_answer │  │ answer      │
│            │  │             │
│ 1. Embed   │  │ Mock: canned│
│    query   │  │ string      │
│ 2. ChromaDB│  │ Real: LLM   │
│    top-3   │  │ response    │
│ 3. Mock:   │  └──────┬──────┘
│    canned  │         │
│    template│         │
│    Real:   │         │
│    LLM +   │         │
│    retry   │         │
└─────┬──────┘         │
      └────────┬───────┘
               ▼
   AssistantResponse (Pydantic)
   { answer, sources, confidence }
```

### Which Stages Branch on MOCK_LLM

| Stage | Branches? | Mock Behavior | Real-LLM Behavior |
|---|---|---|---|
| Ingestion | No | Always real | Always real |
| Embedding | No | Always local (ONNX) | Always local (ONNX) |
| Retrieval | No | Always real (ChromaDB) | Always real (ChromaDB) |
| classify_intent | **Yes** | Keyword heuristic | LLM classification |
| retrieve_and_answer (generation) | **Yes** | Canned template from top chunk | LLM with structured prompt + retry |
| direct_answer (generation) | **Yes** | Fixed canned string | LLM response |

---

## Structured Prompt Template

The prompt follows the **role–context–task–format–length** skeleton and includes:
- ✅ **Negative constraint**: "Do NOT answer using information not present in the provided context."
- ✅ **Few-shot example**: A sample Q&A about delivery timing with expected JSON output.

This template is defined in `agent.py` as `STRUCTURED_PROMPT_TEMPLATE` and is used only in the optional `MOCK_LLM=0` extension path.

---

## LangGraph Architecture

**State (`AgentState`):**

```python
class AgentState(TypedDict):
    query:     str          # User query
    intent:    str          # "policy_question" or "general_question"
    chunks:    list[dict]   # Retrieved document chunks
    response:  dict         # Final answer dict
```

**Nodes:**

| Node | Description |
|---|---|
| `classify_intent` | Keyword heuristic (mock) or LLM (real) intent classification |
| `retrieve_and_answer` | Retrieves top-3 chunks → canned template (mock) or LLM synthesis (real) |
| `direct_answer` | Fixed canned string (mock) or LLM response (real) for non-policy queries |

**Conditional Edge from `classify_intent`:**
- `"policy_question"` → `retrieve_and_answer`
- `"general_question"` → `direct_answer`

**Retry Logic (MOCK_LLM=0 only):**
When the real LLM's raw output fails to validate against the `AssistantResponse` Pydantic schema, the system retries up to 2 additional times with a corrective instruction before returning a clearly marked error response.

---

## Policy Documents (Exact Spec Corpus)

| File | Topic |
|---|---|
| `doc_01.txt` | Delivery Policy |
| `doc_02.txt` | Returns & Refunds |
| `doc_03.txt` | Membership Tiers |
| `doc_04.txt` | Order Tracking |
| `doc_05.txt` | Order Cancellation Policy |
| `doc_06.txt` | Damaged or Missing Items |
| `doc_07.txt` | Gift Cards |
| `doc_08.txt` | Customer Support Hours |

---

## Example Call Transcripts (MOCK_LLM default)

### Example 1: Policy Question (triggers retrieval)

**Request:**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your delivery policy?"}'
```

**Response:**
```json
{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
        "doc_01_chunk_0",
        "doc_02_chunk_0",
        "doc_05_chunk_0"
    ],
    "confidence": 1.0
}
```

- ✅ Routed to `retrieve_and_answer` (keyword "delivery" matched)
- ✅ Answer follows the canned `"Based on the retrieved context: ..."` template
- ✅ Sources are chunk IDs from the correct document (`doc_01_chunk_0`)
- ✅ Confidence is exactly `1.0`
- ✅ No LLM call was made

### Example 2: General Question (no retrieval)

**Request:**
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather like today?"}'
```

**Response:**
```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

- ✅ Routed to `direct_answer` (no policy keywords matched)
- ✅ Answer is the fixed canned string
- ✅ Sources is empty
- ✅ Confidence is exactly `1.0`
- ✅ No LLM call was made

---

## Pydantic Output Schema

```python
class AssistantResponse(BaseModel):
    answer:     str          # The assistant's answer
    sources:    list[str]    # Chunk IDs used (empty for general_question)
    confidence: float        # 0.0–1.0 (fixed 1.0 in mock mode)
```

In mock mode, this schema is populated deterministically:
- `sources` = chunk IDs of retrieved chunks (for `policy_question`), or `[]` (for `general_question`)
- `confidence` = `1.0`

---

## Running with Docker

```bash
cd support_assistant

# Build the image (ingestion runs at build time)
docker build -t zepto-support-assistant .

# Run with mock LLM (default — no API key needed)
docker run -p 7860:7860 zepto-support-assistant

# Test it
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I get a refund?"}'
```

The Dockerfile:
- Installs dependencies from `requirements.txt`
- Runs `python ingest.py` at build time to pre-populate ChromaDB
- Sets `ENV MOCK_LLM=1` by default
- Exposes port 7860 and starts uvicorn

---

## Tests

```bash
cd support_assistant
python test_agent.py       # 15 tests, all in MOCK_LLM mode
# or
pytest test_agent.py -v
```

Tests validate:
- All 8 spec keywords route to `policy_question`
- General queries route to `direct_answer` with canned string
- Mock sources are chunk IDs (not filenames)
- Mock confidence is exactly `1.0`
- Retrieval returns chunks from the correct source document

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MOCK_LLM` | `1` (unset = mock) | Default/`1` = deterministic mock mode; `0` = call real LLM |
| `GROQ_API_KEY` | — | Only needed when `MOCK_LLM=0` (optional, ungraded) |

---

## Files

```
support_assistant/
├── docs/
│   ├── doc_01.txt                 # Delivery Policy
│   ├── doc_02.txt                 # Returns & Refunds
│   ├── doc_03.txt                 # Membership Tiers
│   ├── doc_04.txt                 # Order Tracking
│   ├── doc_05.txt                 # Order Cancellation Policy
│   ├── doc_06.txt                 # Damaged or Missing Items
│   ├── doc_07.txt                 # Gift Cards
│   └── doc_08.txt                 # Customer Support Hours
├── chroma_db/                     # Created by ingest.py (not committed)
├── agent.py                       # LangGraph StateGraph (3 nodes + routing)
├── api.py                         # FastAPI application (POST /ask)
├── constants.py                   # Shared configuration
├── ingest.py                      # Document ingestion pipeline
├── retriever.py                   # ChromaDB semantic retriever
├── test_agent.py                  # 15 smoke tests (MOCK_LLM mode)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
└── README.md                      # This file
```
