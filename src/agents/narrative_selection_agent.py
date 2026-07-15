"""
NarrativeSelectionAgent — Agent 3 of the ScholarForge pipeline.

Takes an ApplicantProfile and a ScholarshipRubric and produces a NarrativePlan:
the narrative frame the applicant's story should take, which profile evidence
aligns with each rubric criterion, which story elements are deliberately
excluded and why, and per-document guidance for the generation agents that
follow.

Input:  ApplicantProfile, ScholarshipRubric
Output: NarrativePlan — validated Pydantic model

Design decisions:
- Uses tool_use for structured extraction (same pattern as the other agents).
- Evidence and rejected-element fields are self-contained text, never
  structural indices into the profile (e.g. NOT work_experience[0].
  achievements[1]) — indices proved fragile in practice; see CLAUDE.md.
- rejected_elements is not an afterthought: the "honest narrative" principle
  requires the agent to log what it excluded and why, not just what it kept.
"""

from __future__ import annotations

from typing import Any, Optional

import anthropic

from ..pipeline.schemas import ApplicantProfile, NarrativePlan, ScholarshipRubric
from .base_agent import BaseAgent


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert narrative strategist for competitive scholarship
applications, with 15 years of experience advising applicants for major international
funding bodies.

Your task is to decide how an applicant's story should be told for one specific
scholarship — not to summarize the applicant, and not to praise them.

Rules:
- Choose exactly one dominant_frame: the single narrative angle that best serves
  this scholarship's rubric (e.g. "research-driven problem solver", "community-rooted
  leader"). List the other frames you considered and rejected in frames_considered.
- frame_rationale must explain, referencing the rubric, why dominant_frame won.
- rubric_alignment must map each evaluation criterion (or eligibility requirement
  worth reinforcing) to a specific, self-contained piece of evidence from the
  profile. Write evidence as standalone text — never as an index or pointer into
  the profile's arrays (never "work_experience[0]" or similar).
- rejected_elements must name real elements from the profile that do NOT serve
  this application, with a concrete reason each was excluded. Do not skip this —
  a narrative that uses everything is not a narrative, it is a list. Every
  application has at least one element worth cutting.
- document_guidance must give lead_with / de_emphasize / angle for each document
  type relevant to this rubric. Vary the angle per document — a motivation letter
  and a research proposal should not repeat the same framing verbatim.
- Do not invent facts. Every rubric_alignment evidence and every rejected_elements
  element must be traceable to something actually present in the profile."""


DOCUMENT_GUIDANCE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["document_type", "angle"],
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "motivation_letter",
                "academic_cv",
                "research_proposal",
                "academic_email",
            ],
        },
        "lead_with":    {"type": "array", "items": {"type": "string"}},
        "de_emphasize": {"type": "array", "items": {"type": "string"}},
        "angle":        {"type": "string"},
    },
}


EXTRACTION_TOOL: dict[str, Any] = {
    "name": "select_narrative",
    "description": (
        "Select the narrative frame, rubric-aligned evidence, and per-document "
        "guidance for a specific applicant applying to a specific scholarship."
    ),
    "input_schema": {
        "type": "object",
        "required": ["dominant_frame", "frame_rationale", "rubric_alignment"],
        "properties": {
            "dominant_frame": {"type": "string"},
            "frames_considered": {
                "type": "array",
                "items": {"type": "string"},
            },
            "frame_rationale": {"type": "string"},
            "rubric_alignment": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["criterion", "evidence"],
                    "properties": {
                        "criterion": {"type": "string"},
                        "evidence":  {"type": "string"},
                    },
                },
            },
            "rejected_elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["element", "reason"],
                    "properties": {
                        "element": {"type": "string"},
                        "reason":  {"type": "string"},
                    },
                },
            },
            "document_guidance": {
                "type": "array",
                "items": DOCUMENT_GUIDANCE_ITEM_SCHEMA,
            },
        },
    },
}


# ─── Agent ────────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Raised when LLM extraction fails or schema validation fails."""


class NarrativeSelectionAgent(BaseAgent):
    """
    Agent 3 of the ScholarForge pipeline.

    Accepts a validated ApplicantProfile and ScholarshipRubric.
    Returns a NarrativePlan: chosen frame, rubric-aligned evidence, rejected
    elements, and per-document guidance for the generation agents.

    Usage:
        agent = NarrativeSelectionAgent(api_key="sk-ant-...")
        plan = agent.execute(profile, rubric)
    """

    AGENT_NAME    = "NarrativeSelectionAgent"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            agent_name=self.AGENT_NAME,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, profile: ApplicantProfile, rubric: ScholarshipRubric) -> NarrativePlan:
        """
        Select a narrative strategy for `profile` against `rubric`.

        Args:
            profile: Validated ApplicantProfile (output of ProfileIngestionAgent).
            rubric:  Validated ScholarshipRubric (output of ScholarshipAnalysisAgent).

        Returns:
            NarrativePlan: Validated structured narrative strategy.
        """
        self.logger.info(
            f"Selecting narrative | applicant={profile.full_name!r} "
            f"| scholarship={rubric.scholarship_name!r} | model={self.model}"
        )

        raw  = self._call_llm(profile, rubric)
        plan = self._build_plan(raw)

        self.logger.info(
            f"Selection complete | frame={plan.dominant_frame!r} "
            f"| alignment={len(plan.rubric_alignment)} "
            f"| rejected={len(plan.rejected_elements)} "
            f"| documents={len(plan.document_guidance)}"
        )

        return plan

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_llm(
        self,
        profile: ApplicantProfile,
        rubric: ScholarshipRubric,
    ) -> dict[str, Any]:
        """Send profile + rubric to the LLM and return the extracted dict."""
        profile_json = profile.model_dump_json(indent=2, exclude_none=True)
        rubric_json  = rubric.model_dump_json(indent=2, exclude_none=True)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "select_narrative"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Select the narrative strategy for this applicant and "
                        "scholarship:\n\n"
                        f"<applicant_profile>\n{profile_json}\n</applicant_profile>\n\n"
                        f"<scholarship_rubric>\n{rubric_json}\n</scholarship_rubric>"
                    ),
                }
            ],
        )

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )

        if tool_block is None:
            raise ExtractionError(
                f"LLM returned no tool_use block. "
                f"stop_reason={response.stop_reason!r}"
            )

        return tool_block.input  # type: ignore[return-value]

    def _build_plan(self, raw: dict[str, Any]) -> NarrativePlan:
        """Parse raw extraction dict into a validated NarrativePlan."""
        try:
            return NarrativePlan(**raw)
        except Exception as exc:
            raise ExtractionError(
                f"NarrativePlan validation failed: {exc}\n"
                f"Extracted keys: {list(raw.keys())}"
            ) from exc
