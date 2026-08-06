#!/usr/bin/env python3
"""
Context Layer — MCP server for the AI Grad knowledge base.

DESIGN NOTE — read before changing anything.

The DATABASE owns the pipeline, not this server. Postgres triggers handle:
  candidates_route            gate_result, gate_reason, review_by, status
  candidates_publish          creates the entries row on auto_publish/approve
  candidates_require_reviewer blocks approval without reviewer + owner
  entries_maintain            builds search_vector, bumps updated_at
  sources_flag_duplicate      fills possible_duplicate_of from span_hash

And retrieval ranking lives in the DB function search_kb_hybrid(), which
ORs across the text and vector legs so a lexical miss can still be rescued
by semantic similarity, and reports which leg matched.

So this server MUST NOT compute the gate, decide status, insert into
entries, or reimplement ranking. Insert the row, call the function, let the
database decide. Duplicating any of it guarantees drift.

The one thing Postgres cannot do is call an embedding API, so every
trigger-created entry starts with embedding NULL. backfill_embeddings()
fixes that and runs automatically after any write that may publish.

Capture goes through the capture_source() RPC, not a raw INSERT — that is
where idempotency (capture_request_id) and span-hash duplicate detection
live. Extraction is a separate, asynchronous pass so that capture itself
stays instant.

Embeddings: OpenAI text-embedding-3-small at dimensions=1024, which matches
entries.embedding vector(1024) exactly. Chosen over Voyage because OpenAI's
API does not train on business data by default.

Env: DATABASE_URL, OPENAI_API_KEY, PORT, EXTRACTION_MODEL (optional)
"""

import hashlib
import json
import os
import uuid

import psycopg
from fastmcp import FastMCP
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
# Set EXTRACTION_MODEL to whatever the OpenAI account actually has access to.
EXTRACT_MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-5.5")
EMBED_DIMS = 1024  # must match entries.embedding vector(1024)

mcp = FastMCP("context-layer")


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"])


def _embed(texts: list[str]) -> list[list[float]]:
    """OpenAI has no document/query distinction — one call shape for both."""
    resp = OpenAI().embeddings.create(
        model=EMBED_MODEL, input=texts, dimensions=EMBED_DIMS
    )
    # Sort by index rather than trusting response order.
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def _backfill(conn) -> int:
    """Embed any published entry that has no vector yet."""
    with conn.cursor() as cur:
        cur.execute(
            "select id, title, body, aliases from entries where embedding is null"
        )
        rows = cur.fetchall()
    if not rows:
        return 0

    texts = [f"{t}\n{b}\n{' '.join(a or [])}" for _, t, b, a in rows]
    vectors = _embed(texts)

    with conn.cursor() as cur:
        for (eid, *_), vec in zip(rows, vectors):
            cur.execute(
                "update entries set embedding = %s::vector where id = %s",
                (str(vec), eid),
            )
    conn.commit()
    return len(rows)


# ------------------------------------------------------------------ read

@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> str:
    """
    Search the AI Grad knowledge base for internal Housecall Pro knowledge.

    Use this for ANY question about how things actually work at Housecall Pro
    that wouldn't be reliably documented publicly: tool and environment access,
    product quirks and limitations, internal team or project names, who owns or
    built what, data warehouse structure, onboarding conventions, and lessons
    captured from past sessions.

    Prefer this over web search or general knowledge for anything
    Housecall-Pro-specific. If it's unclear whether a question is internal,
    search here first — a miss costs nothing.
    """
    vec = str(_embed([query])[0])

    with _db() as conn, conn.cursor() as cur:
        # Ranking lives in the DB. Do not reimplement it here.
        cur.execute(
            """select title, body, section, standing, owner, captured_by,
                      is_stale, text_rank, vector_score, matched_by
               from search_kb_hybrid(%s, %s::vector, %s)""",
            (query, vec, limit),
        )
        rows = cur.fetchall()

    if not rows:
        return "No entries found in the knowledge base."

    out = []
    for (title, body, section, standing, owner, captured_by,
         is_stale, t_rank, v_score, matched_by) in rows:
        flags = ""
        if is_stale:
            flags += " [STALE — past review date]"
        if standing and standing.startswith("LIVE_UNREVIEWED"):
            flags += " [auto-published, not human-reviewed]"
        out.append(
            f"## {title}{flags}\n{body}\n"
            f"(section: {section} · owner: {owner} · captured by {captured_by} · "
            f"matched by {matched_by}: text {t_rank:.2f} / vector {v_score:.2f})"
        )
    return "\n\n".join(out)


# ----------------------------------------------------------------- write

@mcp.tool()
def submit_learning(
    captured_by: str,
    text: str = "",
    turns: list[dict] | None = None,
    kind: str = "conversation_span",
    surface: str = "claude_chat",
    capture_phrase: str = "",
    sensitive_hint: bool = False,
    capture_request_id: str = "",
) -> str:
    """
    Save raw source material. Always raw, always first — nothing is structured
    or scored here, so nothing is lost if later steps fail.

    Pass EITHER `text` (a pasted note or single learning) OR `turns` — the
    EXACT conversation turns, as a list of
    {turn_ref, role, actor|model, text}. Send real turns, never a paraphrase:
    extraction quotes the source text and the database verifies those quotes
    against these turns. A paraphrase makes verification pass while proving
    nothing.

    kind: conversation_span | pasted_note | transcript | document
    capture_phrase: your one-line framing of what the person wanted kept.
      Extraction reads this as a pointer to the part that matters.
    sensitive_hint: true if it may touch personnel or unreleased material.
      NEVER send credentials, keys, or connection strings at all — refuse
      instead; the flag is for review routing, not redaction.
    capture_request_id: pass the SAME uuid when retrying a failed call.
      Leave blank on a first attempt and one is minted.

    Returns source_id plus whether it was already captured or looks like a
    duplicate of existing material.
    """
    if turns is None:
        if not text.strip():
            return "ERROR: provide either `text` or `turns`. Nothing saved."
        turns = [{"turn_ref": "u-1", "role": "user", "actor": captured_by,
                  "text": text}]

    # Give every turn a turn_ref — evidence verification looks turns up by it.
    for i, t in enumerate(turns, 1):
        t.setdefault("turn_ref", f"t-{i}")

    req_id = capture_request_id.strip() or str(uuid.uuid4())

    with _db() as conn, conn.cursor() as cur:
        # capture_source() owns idempotency and span hashing. Do not
        # hand-roll an INSERT here — that bypasses both.
        cur.execute(
            """select source_id, turn_count, span_hash,
                      already_captured, possible_duplicate, sensitive_hint
               from capture_source(%s::uuid, %s::jsonb, %s, %s::source_kind,
                                   %s, %s, %s)""",
            (req_id, json.dumps(turns), captured_by, kind,
             surface, capture_phrase or None, sensitive_hint),
        )
        sid, n_turns, _hash, already, dup, sens = cur.fetchone()
        conn.commit()

    if already:
        return (f"Already captured — same capture_request_id was retried. "
                f"Source {sid}, {n_turns} turns. This is success, not an error.")

    notes = []
    if dup:
        notes.append("Looks like material already captured by someone else — "
                     "that's signal, not a problem.")
    if sens:
        notes.append("Flagged sensitive; it will route to human review.")

    return (f"Captured source {sid} ({n_turns} turns), pending extraction. "
            f"Not searchable yet." + ("\n" + " ".join(notes) if notes else ""))


@mcp.tool()
def list_pending_sources(limit: int = 5) -> str:
    """
    List raw sources captured but not yet processed into candidates.

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
        return "No pending sources — everything captured has been processed."

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
    Submit distilled candidates from a processed source. The DATABASE decides
    what publishes — this tool only inserts and reports what the gate did.

    Each candidate REQUIRES:
      title            the claim itself, as a statement
      body             1-3 sentences; preserve any hedges from the source
      evidence         [{"quote": "...", "turn_ref": "s-2"}] — at least one,
                       quoted from the source. No evidence fails the gate.
      section          tools_and_access | processes_and_workflows | who_to_ask
                       | facts_and_best_practices | company_context | unclassified
      origin_type      human_stated | model_suggested | joint_synthesis
      epistemic_status official_policy | team_convention | reported_practice
                       | personal_learning | proposal | unresolved
      relevance        core | supporting | peripheral
      specificity      high | medium | low
      groundedness     grounded | partial | ungrounded
      score_reasons    {"relevance": "...", "specificity": "...",
                        "groundedness": "..."}

    Optional: aliases (alternate phrasings people would search),
    classification (new | update | conflict | duplicate), sensitive, owner.

    Write entries SELF-CONTAINED — each is retrieved alone, so never write
    "he said" or "as mentioned above". Name the speaker IN THE BODY TEXT where
    attribution matters; `owner` is who maintains the entry, not who said it,
    and is not a substitute for attribution in the prose.
    """
    if not candidates:
        return "No candidates provided."

    required = ["title", "body", "evidence", "section", "origin_type",
                "epistemic_status", "relevance", "specificity",
                "groundedness", "score_reasons"]
    for i, c in enumerate(candidates):
        if missing := [f for f in required if not c.get(f)]:
            return (f"ERROR: candidate {i+1} ('{c.get('title','untitled')}') "
                    f"missing {missing}. Nothing saved.")

    results = []
    with _db() as conn:
        with conn.cursor() as cur:
            for c in candidates:
                # status / gate_result / gate_reason / review_by are set by the
                # candidates_route BEFORE-INSERT trigger. Do not send them.
                cur.execute(
                    """insert into candidates
                       (source_id, title, body, aliases, evidence, section,
                        classification, origin_type, epistemic_status,
                        relevance, specificity, groundedness, score_reasons,
                        sensitive, agent_draft, owner)
                       values (%s,%s,%s,%s,%s::jsonb,%s::kb_section,
                               %s::classification,%s::origin_type,
                               %s::epistemic_status,%s::relevance_tier,
                               %s::specificity_tier,%s::groundedness_tier,
                               %s::jsonb,%s,%s::jsonb,%s)
                       returning status, gate_reason""",
                    (source_id, c["title"], c["body"], c.get("aliases", []),
                     json.dumps(c["evidence"]), c["section"],
                     c.get("classification", "new"), c["origin_type"],
                     c["epistemic_status"], c["relevance"], c["specificity"],
                     c["groundedness"], json.dumps(c["score_reasons"]),
                     bool(c.get("sensitive", False)), json.dumps(c),
                     c.get("owner")),
                )
                status, reason = cur.fetchone()
                results.append((c["title"], status, reason))

            cur.execute(
                "update sources set ingest_status='processed' where id=%s",
                (source_id,),
            )
        conn.commit()
        embedded = _backfill(conn)

    published = [t for t, s, _ in results if s == "auto_published"]
    queued = [(t, r) for t, s, r in results if s == "needs_review"]
    rejected = [(t, r) for t, s, r in results if s == "rejected"]

    lines = [f"Submitted {len(results)} candidates from source {source_id}."]
    if published:
        lines.append(f"\nAuto-published ({len(published)}):")
        lines += [f"  - {t}" for t in published]
    if queued:
        lines.append(f"\nHeld for review ({len(queued)}):")
        lines += [f"  - {t} — {r}" for t, r in queued]
    if rejected:
        lines.append(f"\nRejected ({len(rejected)}):")
        lines += [f"  - {t} — {r}" for t, r in rejected]
    if embedded:
        lines.append(f"\nEmbedded {embedded} newly published entries.")
    return "\n".join(lines)


# ------------------------------------------------------------- extraction

EXTRACTION_POLICY = """You extract reusable knowledge from raw material captured at Housecall Pro (a field-service management SaaS) for an internal knowledge base used by new AI Grad hires.

You are given the RAW TURNS of one source. Extract every distinct, reusable learning.

WHAT COUNTS
- Processes and workflows: how things actually get done here
- Tool instructions, access steps, environment quirks, sanctioned paths
- Who owns what and who to ask for what
- Company-specific facts, dates, schemas, endpoints
- Decisions made, or changes to how things used to work
- Gotchas and workarounds surfaced casually — high value, easy to miss

WHAT TO SKIP
- Pleasantries, introductions, scheduling, small talk
- Generic advice true at any company
- Anything answerable in one web search
- Motivational or culture content with no actionable takeaway

THE TEST: would this save a teammate 15+ minutes, or spare them asking someone?
If no, leave it out. A 60-minute session usually yields a handful of real
learnings, not twenty. Never pad to fill a quota; returning few is correct
when the source is thin. If a speaker corrected themselves, use the correction.

HARD RULES
- NEVER extract credentials, API keys, tokens, connection strings, passwords,
  PII, or real customer data. If a learning cannot be stated without one,
  omit the learning entirely.
- Every quote in `evidence` MUST be a VERBATIM substring of the `text` of the
  turn named in `turn_ref`. Do not paraphrase, tidy, or join across turns.
  Quotes are machine-verified against the source; an inexact quote causes the
  whole candidate to be rejected.
- Write each entry SELF-CONTAINED. It will be retrieved alone, so never write
  "he said", "as mentioned above", or dangling pronouns. Put the speaker's
  NAME in the body where attribution matters — retrieval cannot recover it.

FIELDS
title: the claim itself, phrased as a complete statement
body: 1-3 sentences of detail; preserve hedges from the source
aliases: 2-4 alternate phrasings someone would actually search, including
  name-based ones like "what did Sani say about X"
evidence: [{"quote": "<verbatim>", "turn_ref": "<ref>"}] — at least one
section: tools_and_access | processes_and_workflows | who_to_ask |
  facts_and_best_practices | company_context | unclassified
origin_type: human_stated (a person asserted it) | model_suggested |
  joint_synthesis (emerged in conversation)
epistemic_status: official_policy | team_convention | reported_practice |
  personal_learning | proposal | unresolved
  (official_policy always routes to a human — use it only for genuine policy)
relevance: core | supporting | peripheral
specificity: high | medium | low
groundedness: grounded | partial | ungrounded
  (grounded = the quotes fully support the claim; be honest, partial and
   ungrounded are correct answers and downgrade rather than fabricate)
score_reasons: {"relevance": "...", "specificity": "...", "groundedness": "..."}
owner: the speaker the learning came from, if identifiable
sensitive: true if it touches personnel or unreleased plans

Return ONLY a JSON object: {"candidates": [ ... ]}. No prose, no markdown."""


def _extract(turns: list[dict], capture_phrase: str | None) -> list[dict]:
    """Ask the model to turn raw turns into scored candidates."""
    rendered = "\n\n".join(
        f"[{t.get('turn_ref','?')}] {t.get('actor') or t.get('model') or t.get('role','?')}: "
        f"{t.get('text','')}"
        for t in turns
    )
    framing = (f"\n\nThe person capturing this framed it as: {capture_phrase}\n"
               f"Weight that as a pointer to what matters."
               if capture_phrase else "")

    resp = OpenAI().chat.completions.create(
        model=EXTRACT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_POLICY},
            {"role": "user", "content": f"RAW TURNS:\n\n{rendered}{framing}"},
        ],
    )
    return json.loads(resp.choices[0].message.content).get("candidates", [])


@mcp.tool()
def process_pending_sources(limit: int = 1, dry_run: bool = False) -> str:
    """
    Run the extraction pass: turn captured raw sources into scored candidates.

    This is the step between capture and retrieval. Captured material sits at
    ingest_status='pending' and is NOT searchable until this runs.

    For each pending source it extracts candidates, then inserts them — at
    which point the database verifies every quote against the raw turns,
    checks for duplicates, applies the gate, and routes each candidate to
    auto-publish or review. This tool does not make any of those decisions.

    dry_run=True extracts and shows what would be submitted without writing,
    which is the safe way to sanity-check the extraction policy.

    Start with limit=1. Extraction quality is worth eyeballing before batching.
    """
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """select id, turns, capture_phrase, captured_by, kind
                   from sources
                   where ingest_status = 'pending'
                   order by captured_at
                   limit %s""",
                (limit,),
            )
            pending = cur.fetchall()

        if not pending:
            return "No pending sources — everything captured has been processed."

        report = []
        for sid, turns, phrase, by, kind in pending:
            if dry_run:
                try:
                    drafted = _extract(turns, phrase)
                except Exception as exc:
                    report.append(f"### source {sid}\n  extraction failed: {exc}")
                    continue
                report.append(
                    f"### source {sid} — DRY RUN, nothing written\n"
                    + "\n".join(
                        f"  - [{c.get('relevance')}/{c.get('specificity')}/"
                        f"{c.get('groundedness')}] {c.get('title')}"
                        for c in drafted
                    )
                )
                continue

            # Claim the source so a concurrent run doesn't double-process it.
            with conn.cursor() as cur:
                cur.execute(
                    """update sources
                       set ingest_status='processing',
                           processing_started_at = now(),
                           attempt_count = attempt_count + 1
                       where id = %s and ingest_status = 'pending'
                       returning attempt_count""",
                    (sid,),
                )
                claimed = cur.fetchone()
            conn.commit()
            if not claimed:
                continue

            try:
                drafted = _extract(turns, phrase)
            except Exception as exc:
                with conn.cursor() as cur:
                    cur.execute(
                        """update sources set ingest_status='failed',
                                  last_error=%s where id=%s""",
                        (str(exc)[:500], sid),
                    )
                conn.commit()
                report.append(f"### source {sid}\n  FAILED during extraction: {exc}")
                continue

            outcomes = []
            for c in drafted:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """insert into candidates
                               (source_id, title, body, aliases, evidence, section,
                                classification, origin_type, epistemic_status,
                                relevance, specificity, groundedness, score_reasons,
                                sensitive, agent_draft, owner)
                               values (%s,%s,%s,%s,%s::jsonb,%s::kb_section,
                                       'new'::classification,%s::origin_type,
                                       %s::epistemic_status,%s::relevance_tier,
                                       %s::specificity_tier,%s::groundedness_tier,
                                       %s::jsonb,%s,%s::jsonb,%s)
                               returning status, gate_reason""",
                            (sid, c["title"], c["body"], c.get("aliases", []),
                             json.dumps(c.get("evidence", [])), c.get("section",
                             "unclassified"), c.get("origin_type", "human_stated"),
                             c.get("epistemic_status", "reported_practice"),
                             c.get("relevance", "supporting"),
                             c.get("specificity", "medium"),
                             c.get("groundedness", "partial"),
                             json.dumps(c.get("score_reasons", {})),
                             bool(c.get("sensitive", False)), json.dumps(c),
                             c.get("owner")),
                        )
                        status, reason = cur.fetchone()
                    conn.commit()
                    outcomes.append((c["title"], status, reason))
                except Exception as exc:
                    conn.rollback()
                    outcomes.append((c.get("title", "untitled"), "error", str(exc)[:160]))

            with conn.cursor() as cur:
                cur.execute(
                    "update sources set ingest_status='processed', last_error=null "
                    "where id=%s", (sid,),
                )
            conn.commit()

            lines = [f"### source {sid} ({kind}, captured by {by}) — "
                     f"{len(drafted)} extracted"]
            for title, status, reason in outcomes:
                lines.append(f"  [{status}] {title}\n      {reason}")
            report.append("\n".join(lines))

        embedded = 0 if dry_run else _backfill(conn)

    tail = f"\n\nEmbedded {embedded} newly published entries." if embedded else ""
    return "\n\n".join(report) + tail


# ---------------------------------------------------------------- review

@mcp.tool()
def list_review_queue(limit: int = 10) -> str:
    """
    Show candidates awaiting a human decision, with why each was held.
    Ordered lowest-specificity first.
    """
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """select id, title, body, section, relevance, specificity,
                      groundedness, gate_reason, held_as_carve_out,
                      captured_by, captured_at
               from review_queue limit %s""",
            (limit,),
        )
        rows = cur.fetchall()

    if not rows:
        return "Review queue is empty."

    out = []
    for cid, title, body, sec, rel, spec, grnd, reason, carve, by, at in rows:
        tag = " [CARVE-OUT — needs human judgement]" if carve else ""
        out.append(f"### {title}{tag}\n{body}\nid: {cid}\n"
                   f"{sec} · {rel}/{spec}/{grnd} · held: {reason}\n"
                   f"captured by {by} on {at:%Y-%m-%d}")
    return f"{len(rows)} awaiting review.\n\n" + "\n\n".join(out)


@mcp.tool()
def review_candidate(
    candidate_id: str,
    decision: str,
    reviewer: str,
    owner: str = "",
    edited_title: str = "",
    edited_body: str = "",
) -> str:
    """
    Approve or reject a queued candidate.

    decision: 'approve' publishes it as human-reviewed and fully live;
              'reject' records it permanently so it never publishes.

    reviewer is REQUIRED — the database refuses anonymous approvals. owner
    defaults to the reviewer if not given. Optionally correct the text with
    edited_title / edited_body; the original draft is kept in agent_draft.
    """
    if decision not in ("approve", "reject"):
        return "ERROR: decision must be 'approve' or 'reject'."
    if not reviewer.strip():
        return "ERROR: reviewer is required — the database rejects anonymous approvals."

    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select title from candidates where id=%s and status='needs_review'",
                (candidate_id,),
            )
            row = cur.fetchone()
            if not row:
                return f"No candidate {candidate_id} awaiting review."
            title = row[0]

            if decision == "reject":
                cur.execute(
                    """update candidates
                       set status='rejected', reviewer=%s, gate_result=false,
                           gate_reason='rejected by reviewer'
                       where id=%s""",
                    (reviewer, candidate_id),
                )
                conn.commit()
                return f"Rejected '{title}'. Logged permanently; it will not publish."

            sets = ["status='approved'", "reviewer=%s", "owner=%s"]
            params = [reviewer, owner.strip() or reviewer]
            if edited_title:
                sets.append("title=%s"); params.append(edited_title)
            if edited_body:
                sets.append("body=%s"); params.append(edited_body)
            params.append(candidate_id)

            # candidates_publish trigger creates the entry on this update.
            cur.execute(f"update candidates set {', '.join(sets)} where id=%s", params)
        conn.commit()
        embedded = _backfill(conn)

    return (f"Approved '{edited_title or title}' — published as human-reviewed "
            f"and now searchable. Embedded {embedded} "
            f"entr{'y' if embedded == 1 else 'ies'}.")


@mcp.tool()
def backfill_embeddings() -> str:
    """
    Embed any published entry missing a vector. Safe to run any time.

    Needed because entries created by the database trigger — including ones
    approved directly via SQL — have no embedding until this runs. They are
    findable by keyword but not by meaning.
    """
    with _db() as conn:
        n = _backfill(conn)
    return f"Embedded {n} entries." if n else "Nothing to backfill."


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
