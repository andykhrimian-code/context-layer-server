#!/usr/bin/env python3
"""
Context Layer — MCP server (Step 2: the read path).

Exposes one tool, search_knowledge, that Claude can call directly in any
conversation once this is added as a custom connector. This is the same
query logic as search.py, wrapped so Claude can call it instead of you
running it by hand.

Needs (set as environment variables on the host, e.g. Render):
  DATABASE_URL    — same Supabase session pooler string you've been using
  VOYAGE_API_KEY  — same key from Step 2's local script
  PORT            — set automatically by most hosts (Render sets this)
"""

import os

import psycopg
import voyageai
from fastmcp import FastMCP

MODEL = "voyage-3.5"

mcp = FastMCP(
    "context-layer",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)


@mcp.tool()
def search_knowledge(query: str, limit: int = 3) -> str:
    """
    Search the team's shared onboarding/knowledge base for entries relevant
    to a question. Use this whenever the person asks something that might be
    answered by past onboarding notes, product quirks, or team conventions
    rather than general knowledge.
    """
    vo = voyageai.Client()
    vec = str(vo.embed([query], model=MODEL, input_type="query").embeddings[0])

    sql = """
        select content, 1 - (embedding <=> %s::vector) as similarity
        from entries
        where embedding is not null
        order by embedding <=> %s::vector
        limit %s
    """

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vec, vec, limit))
            rows = cur.fetchall()

    if not rows:
        return "No relevant entries found in the knowledge base."

    return "\n\n".join(f"(similarity {score:.2f}) {content}" for content, score in rows)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
