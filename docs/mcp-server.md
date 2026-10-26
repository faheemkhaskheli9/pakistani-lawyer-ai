# MCP server

`legal-mcp` (or `python -m legal_core.mcp_server`) runs a Model Context Protocol
server over stdio (newline-delimited JSON-RPC 2.0, no extra dependencies).

Tools: `search_corpus`, `verify_citation`, `limitation_period`, `draft_document`,
`summarize_document`. None call an LLM; the client's model does the reasoning.

Claude Desktop config:

```json
{"mcpServers": {"pakistani-lawyer-ai": {"command": "legal-mcp", "env": {"INDEX_DIR": "/path/to/index"}}}}
```

Outputs are legal information, not legal advice; see the disclaimer returned in
`initialize.instructions`.
