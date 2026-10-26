"""Minimal MCP (Model Context Protocol) server over stdio, no extra dependencies.

Exposes the offline legal tools -- corpus search, citation check, limitation
calculator, drafting and document summary -- to MCP clients such as Claude
Desktop.  Run with ``python -m legal_core.mcp_server``.  Speaks newline-delimited
JSON-RPC 2.0.  Tools never call an LLM, so every result is grounded in the
corpus or deterministic rules.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from typing import Any, Callable, TextIO

from .doc_summary import summarize_document
from .drafting import TEMPLATES, render_draft
from .limitation import calculate_limitation
from .safeguards import LEGAL_INFORMATION_DISCLAIMER

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "pakistani-lawyer-ai", "version": "0.1.0"}


def _schema(props: dict[str, dict], required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}


class LegalMCPServer:
    def __init__(self, *, search_service: Any | None = None, citation_verifier: Any | None = None):
        self._search = search_service
        self._verifier = citation_verifier
        self.tools: dict[str, tuple[str, dict, Callable[[dict], Any]]] = {
            "search_corpus": (
                "Search the Pakistani legal corpus; returns cited excerpts.",
                _schema({"query": _S, "top_k": {"type": "integer"}}, ["query"]),
                self._search_corpus,
            ),
            "verify_citation": (
                "Check a citation (PLD/SCMR/CLC/MLD/YLR or 'Section N Act') against the corpus.",
                _schema({"citation": _S}, ["citation"]),
                self._verify,
            ),
            "limitation_period": (
                "Compute a Limitation Act 1908 expiry date.",
                _schema({"kind": _S, "start": {"type": "string", "description": "YYYY-MM-DD"}}, ["kind", "start"]),
                lambda a: asdict(calculate_limitation(a["kind"], date.fromisoformat(a["start"]))),
            ),
            "draft_document": (
                "Fill a drafting template; omit 'template' to list templates.",
                _schema({"template": _S, "values": {"type": "object"}}, []),
                self._draft,
            ),
            "summarize_document": (
                "Extractive summary of document text with entity extraction.",
                _schema({"text": _S}, ["text"]),
                lambda a: asdict(summarize_document(a["text"])),
            ),
        }

    def _search_corpus(self, args: dict) -> Any:
        if self._search is None:
            from .api import _default_services

            self._search = _default_services()[1]
        top_k = args.get("top_k")
        return asdict(self._search.search(args["query"], top_k=int(top_k) if top_k else None))

    def _verify(self, args: dict) -> Any:
        if self._verifier is None:
            from .verify_tool import CitationVerifier

            self._verifier = CitationVerifier()
        return asdict(self._verifier.verify(args["citation"]))

    def _draft(self, args: dict) -> Any:
        if not args.get("template"):
            return {"templates": [{"key": t.key, "title": t.title, "fields": t.fields} for t in TEMPLATES.values()]}
        return asdict(render_draft(args["template"], args.get("values") or {}))

    def handle(self, msg: dict) -> dict | None:
        """Return the JSON-RPC response for ``msg`` (None for notifications)."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification
            return None

        def ok(result: Any) -> dict:
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        def err(code: int, text: str) -> dict:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}

        params = msg.get("params") or {}
        if method == "initialize":
            return ok({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": LEGAL_INFORMATION_DISCLAIMER,
            })
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": [
                {"name": n, "description": d, "inputSchema": s} for n, (d, s, _) in self.tools.items()
            ]})
        if method == "tools/call":
            name = params.get("name")
            if name not in self.tools:
                return err(-32602, f"Unknown tool: {name}")
            args = params.get("arguments") or {}
            required = self.tools[name][1]["required"]
            missing = [r for r in required if r not in args]
            if missing:
                return err(-32602, f"Missing arguments: {', '.join(missing)}")
            try:
                payload = self.tools[name][2](args)
                text, is_error = json.dumps(payload, ensure_ascii=False, default=str), False
            except (ValueError, KeyError, TypeError) as exc:
                text, is_error = f"Error: {exc}", True
            return ok({"content": [{"type": "text", "text": text}], "isError": is_error})
        return err(-32601, f"Method not found: {method}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError
            except ValueError:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


def main() -> None:
    LegalMCPServer().serve()


if __name__ == "__main__":
    main()
