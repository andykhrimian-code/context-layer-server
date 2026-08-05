#!/usr/bin/env python3
"""
Context Layer — MCP server for the AI Grad knowledge base.

Targets the `ai-grad-knowledge-base` Supabase project (Ice's schema).

Read:    search_knowledge      — hybrid vector + full-text over published entries
Write:   submit_learning       — raw conversation span -> sources (pending)
         list_pending_sources  — what's waiting to be processed
         submit_candidates     — scored candidates -> gate -> auto-publish or review
Review:  list_review_queue     — what didn't auto-publish, and why
         review_candidate      — approve / reject a queued candidate

Gate logic mirrors the database:
  * is_carve_out() -> always human review (update/conflict, sensitive,
    official_policy, or unclassified section)
  * core + high + grounded -> auto-publish
  * everything else -> review queue

Env vars: DATABASE_URL, VOYAGE_API_KEY, PORT
"""

import hashlib
import json
import os
import uuid
from datetime import date, timedelta

import psycopg
import voyageai
from fastmcp import FastMCP

MODEL = "voyage-3.5"          # 1024 dims, matches entries.embedding
DEFAULT_REVIEW_DAYS = 90
VOLATILE_SECTIONS = {"tools_and_access", "processes_and_workflows"}
VOLATILE_REVIEW_DAYS = 45

mcp = FastMCP("context-layer")


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"])


def _embed(texts, input_type):
    return voyageai.Client().embed(texts, model=MODEL, input_type=input_type).embeddings


def _review_by(section: str) -> date:
    days = VOLATILE_REVIEW_DAYS if section in VOLATILE_SECTIONS else DEFAULT_REVIEW_DAYS
    return date.today() + timedelta(days=days)


# ---------------------------------------------------------------- read

@mcp.tool()
def search_knowledge(query: str, limit: int = 4) -> str:
    """
    Search the AI Grad knowledge base for internal Housecall Pro knowledge.

    Use this for ANY question about how things actually work at Housecall Pro
    that would not be reliably documented publicly: tool and environment
    access, product quirks and known limitations, internal team or project
    names, who owns or built what, data warehouse structure, onboarding
    conventions, and lessons captured from past sessions.

    Prefer this over web search or general knowledge for anything
    Housecall-Pro-specific. If it is unclear whether a question is internal,
    search here first — a miss costs nothing.
    """
    vec = str(_embed([query], "query")[0])

    # Hybrid: semantic similarity carries most of the weight, lexical rank
    # catches exact names and terms that vectors handle poorly.
    sql = """
        select title, body, section, aliases, status, review_by,
               0.7 * coalesce(1 - (embedding <=> %s::vector), 0)
             + 0.3 * coalesce(ts_rank(search_vector,
                       plainto_tsquery('english', %s)), 0) as score
        from entries
        where status in ('live', 'live_unreviewed')
        order by score desc
        limit %s
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, query, limit))
        rows = cur.fetchall()

    if not rows:
        return "No entries found in the knowledge base."

    out = []
    for title, body, section, aliases, status, review_by, score in rows:
        stale = " [STALE — past review date]" if review_by < date.today() else ""
        unreviewed = " [auto-published, not human-reviewed]" if status == "live_unreviewed" else ""
        out.append(
            f"## {title}{stale}{unreviewed}\n"
            f"{body}\n"
            f"(section: {section} · relevance score {score:.2f})"
        )
    return "\n\n".join(out)


# ---------------------------------------------------------------- write

@mcp.tool()
def submit_learning(
    captured_by: str,
    text: str = "",
    turns: list[dict] | None = None,
    kind: str = "conversation_span",
    surface: str = "claude_chat",
    capture_phrase: str = "",
    sensitive_hint: bool = False,
) -> str:
    """
    Save raw source material to the knowledge base. ALWAYS raw and first —
    nothing is structured or scored here, so nothing is lost if later steps fail.

    Pass EITHER `text` (a pasted note, transcript, or single learning) OR
    `turns` (a conversation span as a list of
    {role, text, actor|model, turn_ref} objects, matching the existing
    convention).

    kind: conversation_span | pasted_note | transcript | document
    captured_by: full name of the person capturing.
    capture_phrase: what the person actually said to trigger capture.
    sensitive_hint: true if it may touch personnel, customer, or unreleased
    material — flags it for review rather than withholding it.

    Returns the source_id.
    """
    if turns is None:
        if not text.strip():
            return "ERROR: provide either `text` or `turns`. Nothing was saved."
        turns = [{"role": "user", "text": text, "actor": captured_by, "turn_ref": "u-1"}]

    canonical = "\n".join(t.get("text", "") for t in turns)
    span_hash = hashlib.sha256(canonical.encode()).hexdigest()

    with _db() as conn, conn.cursor() as cur:
        # span_hash makes re-capturing the same span detectable
        cur.execute("select id from sources where span_hash = %s", (span_hash,))
        existing = cur.fetchone()
        if existing:
            return f"Already captured — this exact span exists as source {existing[0]}."

        cur.execute(
            """insert into sources
               (capture_request_id, turns, surface, capture_phrase, span_hash,
                captured_by, kind, sensitive_hint, ingest_status)
               values (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, 'pending')
               returning id""",
            (str(uuid.uuid4()), json.dumps(turns), surface, capture_phrase,
             span_hash, captured_by, kind, sensitive_hint),
        )
        source_id = cur.fetchone()[0]
        conn.commit()

    return f"Saved as source {source_id} (pending). Not yet searchable — awaiting processing."


@mcp.tool()
def list_pending_sources(limit: int = 5) -> str:
    """
    List raw sources captured but not yet turned into entries.

    Use this to run a processing pass: read each pending source, distill it
    into candidates, and submit them with submit_candidates.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """select id, captured_by, captured_at, kind, sensitive_hint, turns
               from sources where ingest_status = 'pending'
               order by captured_at limit %s""",
            (limit,),
        )
        rows = cur.fetchall()

    if not rows:
        return "No pending sources. Everything captured has been processed."

    out = []
    for sid, by, at, kind, sens, turns in rows:
        flag = " [SENSITIVE HINT]" if sens else ""
        body = "\n".join(
            f"  [{t.get('turn_ref','?')}] {t.get('role','?')}: {t.get('text','')}"
            for t in turns
        )
        out.append(f"### source {sid}{flag}\n{kind} · {by} · {at:%Y-%m-%d}\n{body}")
    return "\n\n".join(out)


@mcp.tool()
def submit_candidates(source_id: str, candidates: list[dict]) -> str:
    """
    Submit distilled, scored candidates from a processed source. The server
    applies the gate and routes each one; it does not second-guess the scores.

    Each candidate dict REQUIRES:
      title           — the claim itself, phrased as a statement
      body            — 1-3 sentences of detail; preserve any hedges in the source
      section         — tools_and_access | processes_and_workflows | who_to_ask
                        | facts_and_best_practices | company_context | unclassified
      origin_type     — human_stated | model_suggested | joint_synthesis
      epistemic_status— official_policy | team_convention | reported_practice
                        | personal_learning | proposal | unresolved
      relevance       — core | supporting | peripheral
      specificity     — high | medium | low
      groundedness    — grounded | partial | ungrounded
      score_reasons   — {"relevance": "...", "specificity": "...", "groundedness": "..."}

    Optional: aliases (alternate phrasings people would search),
    evidence ([{quote, turn_ref}] direct from the source),
    classification (new | update | conflict | duplicate), sensitive, owner.

    Write entries SELF-CONTAINED — they are retrieved alone, so never say
    "he said" or "as mentioned above". Keep attribution in the text where it
    matters, since retrieval cannot recover it otherwise.
    """
    if not candidates:
        return "No candidates provided."

    required = ["title", "body", "section", "origin_type", "epistemic_status",
                "relevance", "specificity", "groundedness", "score_reasons"]
    for i, c in enumerate(candidates):
        missing = [f for f in required if not c.get(f)]
        if missing:
            return f"ERROR: candidate {i+1} ('{c.get('title','untitled')}') missing {missing}. Nothing saved."

    published, queued, rejected = [], [], []

    with _db() as conn, conn.cursor() as cur:
        for c in candidates:
            section = c["section"]
            sensitive = bool(c.get("sensitive", False))
            classification = c.get("classification", "new")
            owner = c.get("owner") or "unassigned"
            review_by = _review_by(section)

            # Floor: not worth reviewing at all
            gate_result = not (c["groundedness"] == "ungrounded"
                               or c["relevance"] == "peripheral")

            cur.execute(
                "select is_carve_out(%s::classification, %s, %s::epistemic_status, %s::kb_section)",
                (classification, sensitive, c["epistemic_status"], section),
            )
            carve_out = cur.fetchone()[0]

            top_tier = (c["relevance"] == "core"
                        and c["specificity"] == "high"
                        and c["groundedness"] == "grounded")

            if not gate_result:
                status, reason = "rejected", f"below floor: {c['relevance']}/{c['groundedness']}"
            elif carve_out:
                status, reason = "needs_review", "carve-out: requires human judgement"
            elif top_tier:
                status, reason = "auto_published", "passed"
            else:
                status, reason = "needs_review", (
                    f"not top tier: {c['relevance']}/{c['specificity']}/{c['groundedness']}")

            cur.execute(
                """insert into candidates
                   (source_id, title, body, aliases, evidence, section, classification,
                    origin_type, epistemic_status, relevance, specificity, groundedness,
                    score_reasons, sensitive, agent_draft, owner, status,
                    gate_result, gate_reason, review_by)
                   values (%s,%s,%s,%s,%s::jsonb,%s::kb_section,%s::classification,
                           %s::origin_type,%s::epistemic_status,%s::relevance_tier,
                           %s::specificity_tier,%s::groundedness_tier,%s::jsonb,%s,
                           %s::jsonb,%s,%s::candidate_status,%s,%s,%s)
                   returning id""",
                (source_id, c["title"], c["body"], c.get("aliases", []),
                 json.dumps(c.get("evidence", [])), section, classification,
                 c["origin_type"], c["epistemic_status"], c["relevance"],
                 c["specificity"], c["groundedness"], json.dumps(c["score_reasons"]),
                 sensitive, json.dumps(c), owner, status, gate_result, reason, review_by),
            )
            candidate_id = cur.fetchone()[0]

            if status == "auto_published":
                _publish(cur, source_id, candidate_id, c, owner, review_by,
                         "auto_pass", "live_unreviewed", None)
                published.append(c["title"])
            elif status == "rejected":
                rejected.append(f"{c['title']} — {reason}")
            else:
                queued.append(f"{c['title']} — {reason}")

        cur.execute(
            "update sources set ingest_status='processed' where id=%s", (source_id,))
        conn.commit()

    lines = [f"Processed source {source_id}."]
    if published:
        lines.append(f"\nAuto-published ({len(published)}) — now searchable:")
        lines += [f"  - {t}" for t in published]
    if queued:
        lines.append(f"\nQueued for review ({len(queued)}):")
        lines += [f"  - {t}" for t in queued]
    if rejected:
        lines.append(f"\nRejected ({len(rejected)}):")
        lines += [f"  - {t}" for t in rejected]
    return "\n".join(lines)


def _publish(cur, source_id, candidate_id, c, owner, review_by,
             method, status, approved_by):
    """Insert a published entry, with embedding and search vector."""
    text_for_embedding = f"{c['title']}\n{c['body']}\n{' '.join(c.get('aliases', []))}"
    vec = str(_embed([text_for_embedding], "document")[0])

    cur.execute(
        """insert into entries
           (source_id, candidate_id, title, body, aliases, evidence, section,
            origin_type, epistemic_status, owner, publication_method, review_by,
            status, sensitive, approved_by, approved_at, embedding, search_vector)
           values (%s,%s,%s,%s,%s,%s::jsonb,%s::kb_section,%s::origin_type,
                   %s::epistemic_status,%s,%s::publication_method,%s,
                   %s::entry_status,%s,%s,
                   case when %s is null then null else now() end,
                   %s::vector,
                   to_tsvector('english', %s))""",
        (source_id, candidate_id, c["title"], c["body"], c.get("aliases", []),
         json.dumps(c.get("evidence", [])), c["section"], c["origin_type"],
         c["epistemic_status"], owner, method, review_by, status,
         bool(c.get("sensitive", False)), approved_by, approved_by,
         vec, text_for_embedding),
    )


# ---------------------------------------------------------------- review

@mcp.tool()
def list_review_queue(limit: int = 10) -> str:
    """
    Show candidates that did NOT auto-publish and are waiting for a human,
    with the reason each was held. Ordered lowest-specificity first.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """select id, title, body, section, relevance, specificity, groundedness,
                      gate_reason, held_as_carve_out, captured_by, captured_at
               from review_queue limit %s""",
            (limit,),
        )
        rows = cur.fetchall()

    if not rows:
        return "Review queue is empty."

    out = []
    for (cid, title, body, section, rel, spec, grnd, reason,
         carve, by, at) in rows:
        tag = " [CARVE-OUT]" if carve else ""
        out.append(
            f"### {title}{tag}\n{body}\n"
            f"id: {cid}\nsection: {section} · {rel}/{spec}/{grnd}\n"
            f"held because: {reason}\ncaptured by {by} on {at:%Y-%m-%d}"
        )
    return f"{len(rows)} awaiting review.\n\n" + "\n\n".join(out)


@mcp.tool()
def review_candidate(
    candidate_id: str,
    decision: str,
    reviewer: str,
    edited_title: str = "",
    edited_body: str = "",
) -> str:
    """
    Approve or reject a queued candidate.

    decision: 'approve' publishes it as human-reviewed and fully live;
              'reject' records it permanently and it never publishes.

    Optionally pass edited_title / edited_body to correct it before publishing.
    The original model draft is preserved in agent_draft either way.
    """
    if decision not in ("approve", "reject"):
        return "ERROR: decision must be 'approve' or 'reject'."

    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """select source_id, title, body, aliases, evidence, section,
                      origin_type, epistemic_status, sensitive, owner, review_by
               from candidates where id = %s and status = 'needs_review'""",
            (candidate_id,),
        )
        row = cur.fetchone()
        if not row:
            return f"No candidate {candidate_id} awaiting review."

        (source_id, title, body, aliases, evidence, section,
         origin_type, epistemic_status, sensitive, owner, review_by) = row

        if decision == "reject":
            cur.execute(
                """update candidates
                   set status='rejected', gate_result=false, reviewer=%s,
                       gate_reason='rejected by reviewer'
                   where id=%s""",
                (reviewer, candidate_id),
            )
            conn.commit()
            return f"Rejected '{title}'. Recorded permanently; it will not publish."

        c = {
            "title": edited_title or title,
            "body": edited_body or body,
            "aliases": aliases,
            "evidence": evidence,
            "section": section,
            "origin_type": origin_type,
            "epistemic_status": epistemic_status,
            "sensitive": sensitive,
        }
        _publish(cur, source_id, candidate_id, c, owner or "unassigned",
                 review_by, "human_review", "live", reviewer)
        cur.execute(
            "update candidates set status='approved', reviewer=%s where id=%s",
            (reviewer, candidate_id),
        )
        conn.commit()

    return f"Approved and published '{c['title']}' as human-reviewed. Now searchable."


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
