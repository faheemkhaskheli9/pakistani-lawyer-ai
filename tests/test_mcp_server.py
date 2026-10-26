import io
import json
import subprocess
import sys

from legal_core.mcp_server import LegalMCPServer
from legal_core.verify_tool import CitationVerification


class FakeVerifier:
    def verify(self, c):
        return CitationVerification(c, "reported_case", True, True, c)


def call(server, method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def tool(server, name, args):
    r = call(server, "tools/call", {"name": name, "arguments": args})
    return r


def test_initialize_and_list():
    s = LegalMCPServer()
    assert call(s, "initialize")["result"]["serverInfo"]["name"] == "pakistani-lawyer-ai"
    names = {t["name"] for t in call(s, "tools/list")["result"]["tools"]}
    assert names == {"search_corpus", "verify_citation", "limitation_period", "draft_document", "summarize_document"}
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_work_and_errors_are_flagged():
    s = LegalMCPServer(citation_verifier=FakeVerifier())
    r = tool(s, "limitation_period", {"kind": "breach_of_contract", "start": "2022-03-10"})["result"]
    assert not r["isError"] and "2025-03-10" in r["content"][0]["text"]
    assert tool(s, "limitation_period", {"kind": "nope", "start": "2022-03-10"})["result"]["isError"]
    assert tool(s, "limitation_period", {"kind": "residuary", "start": "bad"})["result"]["isError"]
    assert "found_in_corpus" in tool(s, "verify_citation", {"citation": "PLD 2010 SC 265"})["result"]["content"][0]["text"]
    d = tool(s, "draft_document", {"template": "affidavit", "values": {"deponent_name": "Ali"}})["result"]
    assert "Ali" in d["content"][0]["text"]
    assert "plaint" in tool(s, "draft_document", {})["result"]["content"][0]["text"]


def test_protocol_errors():
    s = LegalMCPServer()
    assert call(s, "nope")["error"]["code"] == -32601
    assert call(s, "tools/call", {"name": "zzz"})["error"]["code"] == -32602
    assert call(s, "tools/call", {"name": "verify_citation", "arguments": {}})["error"]["code"] == -32602


def test_stdio_loop_handles_garbage():
    out = io.StringIO()
    inp = io.StringIO('not json\n\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n[1]\n')
    LegalMCPServer().serve(inp, out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0]["error"]["code"] == -32700 and lines[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}
    assert lines[2]["error"]["code"] == -32700


def test_real_subprocess_roundtrip():
    p = subprocess.run(
        [sys.executable, "-m", "legal_core.mcp_server"],
        input='{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n',
        capture_output=True, text=True, timeout=30, env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert json.loads(p.stdout)["result"]["tools"]
