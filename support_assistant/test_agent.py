"""
test_agent.py — Tests for the Zepto Support Assistant.

Tests the full pipeline in mock mode (MOCK_LLM unset or "1" — no external API key required).
Validates all acceptance criteria from the spec.

Run:
    python test_agent.py
    # or
    pytest test_agent.py -v
"""

import os
# Ensure mock mode is active (the graded baseline)
os.environ["MOCK_LLM"] = "1"

import pytest
from agent import ask, AssistantResponse


# ── Intent routing tests ──────────────────────────────────────────────────────

def test_policy_question_delivery():
    """A query containing 'delivery' should be classified as policy_question."""
    resp = ask("What is your delivery policy?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0, "Policy question should return sources"
    assert resp.answer.startswith("Based on the retrieved context:")


def test_policy_question_refund():
    """A query containing 'refund' should be classified as policy_question."""
    resp = ask("How do I get a refund for a damaged item?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_return():
    """A query containing 'return' should be classified as policy_question."""
    resp = ask("What is the return window for groceries?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_membership():
    """A query containing 'membership' should be classified as policy_question."""
    resp = ask("Tell me about membership tiers.")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_tracking():
    """A query containing 'tracking' should be classified as policy_question."""
    resp = ask("How does order tracking work?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_cancel():
    """A query containing 'cancel' should be classified as policy_question."""
    resp = ask("Can I cancel my order?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_gift_card():
    """A query containing 'gift card' should be classified as policy_question."""
    resp = ask("What denominations are gift card available in?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_policy_question_support_hours():
    """A query containing 'support hours' should be classified as policy_question."""
    resp = ask("What are the support hours?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


def test_general_question_no_sources():
    """A general (non-policy) question should return empty sources and canned string."""
    resp = ask("What is the weather like today?")
    assert isinstance(resp, AssistantResponse)
    assert resp.sources == [], "General question should have no sources"
    assert resp.answer == "I can only answer questions about Zepto policies right now."
    assert resp.confidence == 1.0


def test_general_question_hello():
    """A greeting should be routed to direct_answer."""
    resp = ask("Hello, how are you?")
    assert isinstance(resp, AssistantResponse)
    assert resp.sources == []
    assert resp.answer == "I can only answer questions about Zepto policies right now."


# ── Schema validation tests ──────────────────────────────────────────────────

def test_response_schema():
    """Response must always match the AssistantResponse schema."""
    resp = ask("How do I cancel my Zepto Pass membership?")
    assert hasattr(resp, "answer")
    assert hasattr(resp, "sources")
    assert hasattr(resp, "confidence")
    assert isinstance(resp.answer, str)
    assert isinstance(resp.sources, list)
    assert isinstance(resp.confidence, float)


def test_mock_sources_are_chunk_ids():
    """In mock mode, sources should be chunk IDs (not filenames)."""
    resp = ask("What is your delivery policy?")
    for src in resp.sources:
        assert "_chunk_" in src, f"Source '{src}' should be a chunk ID like 'doc_01_chunk_0'"


def test_mock_confidence_is_one():
    """In mock mode, confidence should be exactly 1.0."""
    resp = ask("Tell me about refund timelines.")
    assert resp.confidence == 1.0


# ── Retrieval correctness tests ───────────────────────────────────────────────

def test_delivery_retrieves_doc_01():
    """A delivery question should retrieve chunks from doc_01."""
    resp = ask("How long does delivery take?")
    assert any("doc_01" in src for src in resp.sources), \
        f"Expected doc_01 in sources, got {resp.sources}"


def test_refund_retrieves_doc_02():
    """A refund question should retrieve chunks from doc_02."""
    resp = ask("What is the refund window for perishable items?")
    assert any("doc_02" in src for src in resp.sources), \
        f"Expected doc_02 in sources, got {resp.sources}"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running support assistant tests in MOCK_LLM mode...\n")

    tests = [
        test_policy_question_delivery,
        test_policy_question_refund,
        test_policy_question_return,
        test_policy_question_membership,
        test_policy_question_tracking,
        test_policy_question_cancel,
        test_policy_question_gift_card,
        test_policy_question_support_hours,
        test_general_question_no_sources,
        test_general_question_hello,
        test_response_schema,
        test_mock_sources_are_chunk_ids,
        test_mock_confidence_is_one,
        test_delivery_retrieves_doc_01,
        test_refund_retrieves_doc_02,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
