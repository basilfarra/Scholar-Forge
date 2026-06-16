"""
ProfileIngestionAgent — Agent 1 of the ScholarForge pipeline.

Takes raw CV/resume text and extracts a structured ApplicantProfile
using Anthropic's tool_use API for reliable JSON extraction.

Design rationale:
- tool_use (function calling) is more reliable than JSON mode for nested schemas:
  the model is constrained to the tool's schema, eliminating format errors.
- Fields are Optional where data may legitimately be absent from a CV.
- Input validation runs before any API call to fail fast on bad input.
- Extraction uses a single API call with forced tool_use (no fallback needed
  at this stage — retry logic is handled by BaseAgent.execute()).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import anthropic

from ..pipeline.schemas import ApplicantProfile
from .base_agent import BaseAgent


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise CV and academic profile parser.
Your only task is to extract information that is explicitly stated in the provided text.

Extraction rules:
- Extract ONLY what is present. Never infer, assume, or hallucinate missing fields.
- Dates: use YYYY-MM format where month is available, YYYY where only year is stated.
- GPA: preserve the exact value and note the scale (e.g. 3.8 on 4.0, or 94 on 100).
- Skills proficiency:
    "expert"     → 3+ years of production/lead-level usage
    "proficient" → consistent usage in real projects
    "familiar"   → coursework, study, or minor exposure
- Achievements: extract measurable outcomes exactly as stated. Preserve numbers.
- If a section is absent from the CV, omit the corresponding field entirely."""


# ─── Extraction Tool Schema ────────────────────────────────────────────────────

EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_profile",
    "description": (
        "Extract a structured applicant profile from CV text. "
        "Use only information explicitly present in the text."
    ),
    "input_schema": {
        "type": "object",
        "required": ["full_name"],
        "properties": {
            "full_name":    {"type": "string"},
            "email":        {"type": "string"},
            "phone":        {"type": "string"},
            "location":     {"type": "string"},
            "nationality":  {"type": "string"},
            "linkedin_url": {"type": "string"},
            "github_url":   {"type": "string"},
            "summary":      {"type": "string", "description": "Professional summary if present"},

            "academic_records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["degree", "institution", "field_of_study"],
                    "properties": {
                        "degree":           {"type": "string"},
                        "institution":      {"type": "string"},
                        "field_of_study":   {"type": "string"},
                        "start_date":       {"type": "string"},
                        "end_date":         {"type": "string"},
                        "gpa":              {"type": "number"},
                        "gpa_scale":        {"type": "string"},
                        "grade_description":{"type": "string"},
                        "honors":           {"type": "array", "items": {"type": "string"}},
                        "relevant_courses": {"type": "array", "items": {"type": "string"}},
                        "thesis_title":     {"type": "string"},
                    },
                },
            },

            "work_experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["role", "organization"],
                    "properties": {
                        "role":            {"type": "string"},
                        "organization":    {"type": "string"},
                        "start_date":      {"type": "string"},
                        "end_date":        {"type": "string"},
                        "is_current":      {"type": "boolean"},
                        "location":        {"type": "string"},
                        "employment_type": {
                            "type": "string",
                            "enum": ["full_time", "part_time", "freelance", "internship", "volunteer"],
                        },
                        "achievements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific, measurable achievements. Preserve numbers.",
                        },
                        "technologies": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },

            "technical_skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["category", "skills"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "e.g. 'Programming Languages', 'Frameworks', "
                                "'Databases', 'AI/ML Tools', 'DevOps'"
                            ),
                        },
                        "skills": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "proficiency": {
                                        "type": "string",
                                        "enum": ["expert", "proficient", "familiar"],
                                    },
                                    "years_experience": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },

            "publications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title"],
                    "properties": {
                        "title":      {"type": "string"},
                        "venue":      {"type": "string"},
                        "year":       {"type": "integer"},
                        "doi":        {"type": "string"},
                        "co_authors": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },

            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name":         {"type": "string"},
                        "description":  {"type": "string"},
                        "technologies": {"type": "array", "items": {"type": "string"}},
                        "url":          {"type": "string"},
                        "impact":       {"type": "string"},
                    },
                },
            },

            "certifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name":          {"type": "string"},
                        "issuer":        {"type": "string"},
                        "date":          {"type": "string"},
                        "credential_id": {"type": "string"},
                    },
                },
            },

            "awards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title"],
                    "properties": {
                        "title":       {"type": "string"},
                        "issuer":      {"type": "string"},
                        "year":        {"type": "integer"},
                        "description": {"type": "string"},
                    },
                },
            },

            "languages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["language", "proficiency"],
                    "properties": {
                        "language": {"type": "string"},
                        "proficiency": {
                            "type": "string",
                            "enum": ["native", "fluent", "professional", "intermediate", "basic"],
                        },
                    },
                },
            },

            "extracurricular": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["activity"],
                    "properties": {
                        "activity":     {"type": "string"},
                        "role":         {"type": "string"},
                        "organization": {"type": "string"},
                        "duration":     {"type": "string"},
                        "description":  {"type": "string"},
                    },
                },
            },
        },
    },
}


# ─── Agent ────────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Raised when profile extraction fails due to LLM or validation errors."""


class ProfileIngestionAgent(BaseAgent):
    """
    Agent 1 of the ScholarForge pipeline.

    Parses raw CV/resume text into a validated ApplicantProfile using
    Anthropic's tool_use for structured JSON extraction.

    Input:  str  — raw CV text (plain text or pre-extracted from PDF)
    Output: ApplicantProfile — validated Pydantic model

    Usage:
        agent = ProfileIngestionAgent(api_key="sk-ant-...")
        profile = agent.execute(cv_text)
    """

    AGENT_NAME    = "ProfileIngestionAgent"
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

    def run(self, cv_text: str) -> ApplicantProfile:
        """
        Core extraction logic. Called by BaseAgent.execute() with retry wrapping.

        Args:
            cv_text: Raw CV/resume text content.

        Returns:
            ApplicantProfile: Validated structured profile.

        Raises:
            ValueError:       Input is empty or too short.
            ExtractionError:  LLM failed to return expected structure.
        """
        self._validate_input(cv_text)

        self.logger.info(
            f"Extracting profile | input_chars={len(cv_text)} | model={self.model}"
        )

        raw = self._call_llm(cv_text)
        profile = self._build_profile(raw)

        self.logger.info(
            f"Extraction complete | name={profile.full_name!r} "
            f"| academic={len(profile.academic_records)} "
            f"| work={len(profile.work_experience)} "
            f"| skill_categories={len(profile.technical_skills)}"
        )

        return profile

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_input(self, cv_text: str) -> None:
        if not cv_text or not cv_text.strip():
            raise ValueError("CV text cannot be empty.")
        if len(cv_text.strip()) < 100:
            raise ValueError(
                f"CV text is too short ({len(cv_text)} chars). "
                "A valid CV must be at least 100 characters."
            )

    def _call_llm(self, cv_text: str) -> dict[str, Any]:
        """
        Send CV text to the LLM using tool_use and return the extracted dict.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_profile"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the structured profile from the following CV:\n\n"
                        f"<cv_text>\n{cv_text}\n</cv_text>"
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
                f"stop_reason={response.stop_reason!r}. "
                f"Content types: {[b.type for b in response.content]}"
            )

        return tool_block.input  # type: ignore[return-value]

    def _build_profile(self, raw: dict[str, Any]) -> ApplicantProfile:
        """
        Parse the raw extraction dict into a validated ApplicantProfile.
        """
        try:
            return ApplicantProfile(**raw)
        except Exception as exc:
            raise ExtractionError(
                f"Profile schema validation failed: {exc}\n"
                f"Extracted keys: {list(raw.keys())}"
            ) from exc
