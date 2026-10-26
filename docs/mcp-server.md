# MCP server

`legal-mcp` (or `python -m legal_core.mcp_server`) runs a Model Context Protocol
server over stdio (newline-delimited JSON-RPC 2.0, no extra dependencies).

Tools: `search_corpus`, `verify_citation`, `limitation_period`, `draft_document`,
`summarize_document`. None call an LLM; the client's model does the reasoning.

The server reads the index from `data/index` relative to its working directory
(build it first with the ingest CLI). Claude Desktop config:

```json
{"mcpServers": {"pakistani-lawyer-ai": {"command": "legal-mcp"}}}
```

Run it from the repository root (or set the client's working directory there).

Outputs are legal information, not legal advice; see the disclaimer returned in
`initialize.instructions`.
