from types import SimpleNamespace

import pytest

from legal_core.llm import AnthropicProvider, LocalExtractiveProvider, get_llm_provider


class FakeClient:
    def __init__(self, text):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def test_anthropic_provider_sends_grounded_prompt():
    client = FakeClient("Answer [SOURCE 1: a, Section 1]")
    provider = AnthropicProvider(client=client)
    out = provider.generate(question="Q?", context="[SOURCE 1: a, Section 1]\ntext")
    assert out.startswith("Answer")
    call = client.calls[0]
    assert "ONLY the numbered sources" in call["system"]
    assert "Question: Q?" in call["messages"][0]["content"]


def test_insufficient_sources_is_an_error():
    provider = AnthropicProvider(client=FakeClient("INSUFFICIENT_SOURCES"))
    with pytest.raises(ValueError):
        provider.generate(question="Q?", context="ctx")


def test_empty_context_rejected():
    with pytest.raises(ValueError):
        AnthropicProvider(client=FakeClient("x")).generate(question="Q", context=" ")


def test_env_selects_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert isinstance(get_llm_provider(), LocalExtractiveProvider)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises((ValueError, ImportError)):
        get_llm_provider()


def test_service_retries_then_withholds_on_unverifiable_citation():
    from legal_core.generation import AnswerGenerator
    from legal_core.retrieval import RetrievedChunk
    from legal_core.service import QuestionAnsweringService

    class R:
        def retrieve(self, q, **kw):
            return [RetrievedChunk(rank=1, chunk_id="c", score=0.9, text="Contracts need consent.",
                                   source="a.txt", section="Section 1", chunk_index=0, metadata={})]

    class P:
        calls = 0
        def generate(self, *, question, context):
            P.calls += 1
            return "Per [SOURCE 9: fake.txt, Section 99] anything."

    result = QuestionAnsweringService(R(), generator=AnswerGenerator(P()), min_score=0.1).ask("contract consent?")
    assert P.calls == 2
    assert result.status == "citation_verification_failed"


def test_service_maps_insufficient_sources_to_no_source():
    from legal_core.generation import AnswerGenerator
    from legal_core.retrieval import RetrievedChunk
    from legal_core.service import QuestionAnsweringService

    class R:
        def retrieve(self, q, **kw):
            return [RetrievedChunk(rank=1, chunk_id="c", score=0.9, text="x", source="a.txt",
                                   section="Section 1", chunk_index=0, metadata={})]

    class P:
        def generate(self, *, question, context):
            raise ValueError("insufficient")

    result = QuestionAnsweringService(R(), generator=AnswerGenerator(P()), min_score=0.1).ask("q?")
    assert result.status == "no_relevant_source"
