"""Shared app for the legal-information assistant.

Phase 1 covers corpus ingestion: parsing source documents, chunking them,
and tagging each chunk with a source citation
(`legal_core.chunking`, `legal_core.parsing`, `legal_core.ingest`).
Embedding/retrieval/generation land in later phases on top of this layer.
"""
