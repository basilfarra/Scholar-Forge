# ScholarForge

Multi-agent AI pipeline that automates production of competitive scholarship
application materials (motivation letters, academic CVs, research proposals,
academic emails). Each pipeline stage is a discrete agent with a typed
contract to the next — not a single prompt doing everything.

## Commands

```bash
# Install
pip install anthropic pydantic pytest structlog httpx beautifulsoup4

# Test — fully mocked, no API key needed
python -m pytest tests/ -v

# Windows: use `py`, not `python`, for all of the above
```

## Structure

```
src/
├── agents/                       # one file per agent, all inherit BaseAgent
│   ├── base_agent.py             # retry, telemetry, execute() wrapper
│   ├── profile_ingestion_agent.py
│   └── scholarship_analysis_agent.py
└── pipeline/
    └── schemas.py                # all inter-agent Pydantic contracts
tests/                            # one file per agent, mocked with unittest.mock
```

## Status
- ✅ `ProfileIngestionAgent`, `ScholarshipAnalysisAgent` — built, tested (20 tests passing)
- 🔨 `NarrativeSelectionAgent` — schema finalized (below), agent not implemented yet
- 📋 Agents 4–10 — designed, not started

## Architecture rules
(Things a new session can't get from reading one agent file alone.)

- Every agent inherits `BaseAgent`, implements `run()` only. Callers use
  `execute()`, never `run()` directly — it's what adds retry + telemetry.
- Extraction goes through Anthropic `tool_use`. Never free-text JSON parsing.
- Every schema sets `model_config = {"extra": "forbid"}` — reject unexpected
  LLM output rather than silently accept it.
- **Never design a schema field that asks an LLM for an exact structural
  index** (e.g. `work_experience[0].achievements[1]`) via tool_use. Decided
  against this for `NarrativePlan` — it's fragile in practice. Keep evidence
  fields as self-contained text instead.

## Current task: NarrativeSelectionAgent

Target `NarrativePlan` (replaces the placeholder at the bottom of
`src/pipeline/schemas.py`): `dominant_frame`, `frames_considered`,
`frame_rationale`, `rubric_alignment` (criterion → evidence pairs, must be
non-empty), `rejected_elements` (element + reason), `document_guidance`
(per-`DocumentType`: `lead_with` / `de_emphasize` / `angle`).

**Once this is committed to `schemas.py`, delete this section and replace it
with:** `NarrativePlan schema — see src/pipeline/schemas.py`. Don't let this
file describe code that already exists elsewhere — that's the exact mistake
the README made once already.

## Boundaries
- No FastAPI, no frontend, no `docs/` scaffolding before the 3-agent
  pipeline (Ingestion → Analysis → Narrative Selection) runs end-to-end.
- This file and the README must never describe something the codebase
  doesn't actually do.

## Verification
After any change: run `python -m pytest tests/ -v`. All tests — including
any new ones for the agent you just touched — must pass before the task
is done.

## Conventions
- English only in code, comments, docstrings, commit messages, and technical
  docs. No Arabic-English mixing in technical content.
- Commits: `feat:` / `test:` / `fix:` prefixes, logically separated, push to
  `origin main`.
