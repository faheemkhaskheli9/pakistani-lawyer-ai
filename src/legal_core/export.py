"""Render a Q&A result as a Markdown research note with a citations list."""
from __future__ import annotations

from .service import QAResult


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


def export_markdown(result: QAResult) -> str:
    lines = ["# Legal research note", ""]
    if result.answer is None:
        lines += [result.message, ""]
    else:
        ans = result.answer
        lines += [f"**Question:** {ans.question}", "", "## Answer", "", ans.answer, ""]
        if ans.citations:
            lines += ["## Citations", ""]
            checks = {}
            if result.citation_verification is not None:
                checks = {c.chunk_id: c for c in result.citation_verification.checks}
            for i, c in enumerate(ans.citations, start=1):
                check = checks.get(c.chunk_id)
                status = "" if check is None else (
                    " — verified against source" if check.verified else " — NOT verified"
                )
                where = f" ({c.section})" if c.section else ""
                lines += [f"{i}. **{c.citation}**{where}, source `{c.source}`{status}", "", _quote(c.excerpt), ""]
    lines += ["---", f"_{result.disclaimer}_", ""]
    return "\n".join(lines)
