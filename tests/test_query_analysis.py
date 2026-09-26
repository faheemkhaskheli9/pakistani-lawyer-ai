"""Tests for legal query analysis (issue #19)."""
from legal_core.generation import AnswerGenerator
from legal_core.llm import LLMProvider
from legal_core.query_analysis import analyze_query, classify_intent, detect_jurisdiction
from legal_core.retrieval import RetrievedChunk
from legal_core.service import QuestionAnsweringService


def test_detects_supported_pakistani_jurisdictions():
    assert detect_jurisdiction("Punjab tenancy law") == "Punjab"
    assert detect_jurisdiction("Sindh High Court case") == "Sindh"
    assert detect_jurisdiction("KPK employment rules") == "Khyber Pakhtunkhwa"
    assert detect_jurisdiction("Balochistan local law") == "Balochistan"
    assert detect_jurisdiction("ICT rent law in Islamabad") == "Islamabad Capital Territory"
    assert detect_jurisdiction("Federal Constitution of Pakistan") == "Federal"


def test_multiple_jurisdictions_are_left_unspecified():
    assert detect_jurisdiction("Compare Punjab and Sindh tenancy law") is None


def test_classifies_common_legal_query_intents():
    assert classify_intent("What does section 5 of the Act provide?") == "statute_lookup"
    assert classify_intent("Find Supreme Court case law about bail") == "case_search"
    assert classify_intent("PLD 2024 SC 123") == "citation_lookup"
    assert classify_intent("What is the definition of free consent?") == "legal_definition"
    assert classify_intent("Summarize this contract clause") == "document_question"
    assert classify_intent("Prepare legal research on authorities on bail") == "research_request"
    assert classify_intent("Can a person obtain bail?") == "legal_question"


def test_analysis_normalizes_query_whitespace():
    result = analyze_query("  Punjab   section 5   ")
    assert result.normalized_query == "Punjab section 5"
    assert result.intent == "statute_lookup"
    assert result.jurisdiction == "Punjab"


class Provider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        return "Grounded answer."


class Retriever:
    def retrieve(self, query):
        return [
            RetrievedChunk(
                rank=1,
                chunk_id="c1",
                score=0.9,
                text="Source text.",
                source="law.txt",
                section="Section 5",
                chunk_index=0,
            )
        ]


def test_qa_response_exposes_query_analysis():
    result = QuestionAnsweringService(
        Retriever(),
        generator=AnswerGenerator(Provider()),
        min_score=0.2,
    ).ask("Punjab section 5")

    assert result.status == "answered"
    assert result.query_analysis is not None
    assert result.query_analysis.intent == "statute_lookup"
    assert result.query_analysis.jurisdiction == "Punjab"
