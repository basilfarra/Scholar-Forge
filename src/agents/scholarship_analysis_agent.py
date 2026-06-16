"""
ScholarshipAnalysisAgent — Agent 2 of the ScholarForge pipeline.

Takes a scholarship URL or raw text and extracts a structured ScholarshipRubric:
what the scholarship looks for, who qualifies, how applications are evaluated.

Input:  str  — either a URL ("https://...") or raw scholarship description text
Output: ScholarshipRubric — validated Pydantic model

Design decisions:
- If input starts with "http", fetch the page and extract clean text first.
- Uses tool_use for structured extraction (same pattern as ProfileIngestionAgent).
- Extracts not just stated criteria but inferred themes and committee profile —
  this is the signal that separates ScholarForge from a simple scraper.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import anthropic
import httpx
from bs4 import BeautifulSoup

from ..pipeline.schemas import ScholarshipRubric
from .base_agent import BaseAgent


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert scholarship analyst with 15 years of experience
on evaluation committees at major international funding bodies.

Your task is to read a scholarship description and extract:
1. The explicit criteria (stated requirements, documents, deadlines)
2. The implicit criteria (what values and profile the committee is really looking for)
3. Red flags (anything that would disqualify an applicant immediately)

Rules:
- Extract only what is present or strongly implied in the text.
- For key_themes: identify the 3-5 values this scholarship most rewards
  (e.g. "innovation", "public service", "research potential", "leadership").
  These are NOT always stated explicitly — read between the lines.
- For committee_profile: describe in one sentence who typically evaluates
  this type of scholarship and what they respond to.
- For evaluation_criteria weight: use percentages if stated, otherwise
  use "high / medium / low priority" based on emphasis in the text.
- Do not hallucinate requirements not present or implied in the text."""


EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_scholarship_rubric",
    "description": (
        "Extract a structured scholarship rubric from the description text. "
        "Include both explicit and implicit evaluation criteria."
    ),
    "input_schema": {
        "type": "object",
        "required": ["scholarship_name"],
        "properties": {
            "scholarship_name": {"type": "string"},
            "provider":         {"type": "string"},
            "degree_level": {
                "type": "string",
                "enum": ["bachelor", "master", "phd", "any"],
            },
            "fields_of_study": {
                "type": "array",
                "items": {"type": "string"},
            },
            "deadline":      {"type": "string"},
            "award_details": {
                "type": "string",
                "description": "Full funding, stipend amount, duration, what is covered",
            },
            "eligibility": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["field", "condition"],
                    "properties": {
                        "field":        {"type": "string"},
                        "condition":    {"type": "string"},
                        "is_mandatory": {"type": "boolean"},
                        "notes":        {"type": "string"},
                    },
                },
            },
            "evaluation_criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["criterion"],
                    "properties": {
                        "criterion":   {"type": "string"},
                        "weight":      {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "required_documents": {
                "type": "array",
                "items": {"type": "string"},
            },
            "key_themes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-5 values this scholarship most rewards. "
                    "May be implicit — read between the lines."
                ),
            },
            "committee_profile": {
                "type": "string",
                "description": "One sentence: who evaluates this and what they respond to.",
            },
            "red_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hard disqualifiers explicitly stated in the text.",
            },
        },
    },
}


# ─── Agent ────────────────────────────────────────────────────────────────────

class FetchError(Exception):
    """Raised when a URL cannot be fetched or yields no usable text."""

class ExtractionError(Exception):
    """Raised when LLM extraction fails or schema validation fails."""


class ScholarshipAnalysisAgent(BaseAgent):
    """
    Agent 2 of the ScholarForge pipeline.

    Accepts a scholarship URL or raw text.
    Returns a ScholarshipRubric with explicit + implicit evaluation criteria.

    Usage:
        agent = ScholarshipAnalysisAgent(api_key="sk-ant-...")
        rubric = agent.execute("https://scholarship-url.com/apply")
        # or
        rubric = agent.execute(raw_scholarship_text)
    """

    AGENT_NAME    = "ScholarshipAnalysisAgent"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    MIN_TEXT_LENGTH = 150   # chars — below this, the page likely failed to load

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        max_retries: int = 3,
        fetch_timeout: int = 15,
    ) -> None:
        super().__init__(
            agent_name=self.AGENT_NAME,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self._client        = anthropic.Anthropic(api_key=api_key)
        self._fetch_timeout = fetch_timeout

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, source: str) -> ScholarshipRubric:
        """
        Extract ScholarshipRubric from URL or raw text.

        Args:
            source: A URL ("https://...") or raw scholarship description text.

        Returns:
            ScholarshipRubric: Validated structured rubric.
        """
        if self._is_url(source):
            self.logger.info(f"Fetching URL: {source}")
            text       = self._fetch_url(source)
            source_url = source
        else:
            self.logger.info("Using raw text input")
            text       = source
            source_url = None

        self._validate_text(text)

        self.logger.info(
            f"Analysing scholarship | chars={len(text)} | model={self.model}"
        )

        raw    = self._call_llm(text)
        rubric = self._build_rubric(raw, source_url)

        self.logger.info(
            f"Analysis complete | name={rubric.scholarship_name!r} "
            f"| eligibility={len(rubric.eligibility)} "
            f"| criteria={len(rubric.evaluation_criteria)} "
            f"| themes={rubric.key_themes}"
        )

        return rubric

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_url(source: str) -> bool:
        return source.strip().startswith(("http://", "https://"))

    def _fetch_url(self, url: str) -> str:
        """Fetch a URL and return clean text (HTML tags stripped)."""
        try:
            response = httpx.get(
                url,
                timeout=self._fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (ScholarForge/1.0)"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"Failed to fetch {url}: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navigation, scripts, styles — keep only content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _validate_text(self, text: str) -> None:
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            raise FetchError(
                f"Scholarship text is too short ({len(text)} chars). "
                "The page may have failed to load or be behind a login wall."
            )

    def _call_llm(self, text: str) -> dict[str, Any]:
        """Send scholarship text to the LLM and return extracted dict."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_scholarship_rubric"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyse the following scholarship description and extract "
                        "the structured rubric:\n\n"
                        f"<scholarship_text>\n{text}\n</scholarship_text>"
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

    def _build_rubric(
        self,
        raw: dict[str, Any],
        source_url: Optional[str],
    ) -> ScholarshipRubric:
        """Parse raw extraction dict into a validated ScholarshipRubric."""
        if source_url:
            raw["source_url"] = source_url
        try:
            return ScholarshipRubric(**raw)
        except Exception as exc:
            raise ExtractionError(
                f"ScholarshipRubric validation failed: {exc}\n"
                f"Extracted keys: {list(raw.keys())}"
            ) from exc
