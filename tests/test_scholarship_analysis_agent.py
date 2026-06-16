"""
Tests for ScholarshipAnalysisAgent.
Run: python -m pytest tests/test_scholarship_analysis_agent.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from src.agents.scholarship_analysis_agent import (
    ScholarshipAnalysisAgent, FetchError, ExtractionError
)
from src.pipeline.schemas import ScholarshipRubric

SAMPLE_SCHOLARSHIP_TEXT = """
Fulbright Foreign Student Program

The Fulbright Program offers grants for graduate students, young professionals,
and artists from abroad to study and conduct research in the United States.

Eligibility:
- Must be a citizen of a participating country
- Hold a bachelor's degree or equivalent before the grant begins
- Minimum GPA of 3.0 on a 4.0 scale
- Proficiency in English (TOEFL/IELTS required)
- Must not hold US permanent residency or citizenship

Award Details:
Full funding including tuition, living stipend, health insurance, and travel.
Duration: 1-2 years depending on program.

Evaluation Criteria:
- Academic excellence and achievement (40%)
- Leadership potential and community engagement (30%)
- Research proposal quality and feasibility (20%)
- Cross-cultural communication skills (10%)

Required Documents:
- Statement of Purpose (1000 words)
- Three letters of recommendation
- Academic transcripts (official)
- TOEFL/IELTS score report
- CV/Resume

Deadline: October 15, 2026
"""

MOCK_EXTRACTION = {
    "scholarship_name": "Fulbright Foreign Student Program",
    "provider": "U.S. Department of State",
    "degree_level": "master",
    "fields_of_study": ["All fields"],
    "deadline": "October 15, 2026",
    "award_details": "Full funding: tuition, living stipend, health insurance, travel. 1-2 years.",
    "eligibility": [
        {"field": "citizenship", "condition": "Citizen of participating country", "is_mandatory": True},
        {"field": "education", "condition": "Bachelor's degree before grant begins", "is_mandatory": True},
        {"field": "GPA", "condition": "Minimum 3.0 on 4.0 scale", "is_mandatory": True},
        {"field": "language", "condition": "TOEFL/IELTS required", "is_mandatory": True},
    ],
    "evaluation_criteria": [
        {"criterion": "Academic excellence", "weight": "40%"},
        {"criterion": "Leadership and community engagement", "weight": "30%"},
        {"criterion": "Research proposal quality", "weight": "20%"},
        {"criterion": "Cross-cultural communication", "weight": "10%"},
    ],
    "required_documents": [
        "Statement of Purpose",
        "Three letters of recommendation",
        "Academic transcripts",
        "TOEFL/IELTS score report",
        "CV/Resume",
    ],
    "key_themes": ["academic excellence", "leadership", "cross-cultural exchange", "public service"],
    "committee_profile": "Academic and government professionals who value global leadership and cultural diplomacy.",
    "red_flags": ["US permanent residency or citizenship disqualifies applicants"],
}


class TestInputDetection:
    def setup_method(self):
        self.agent = ScholarshipAnalysisAgent(api_key="fake")

    def test_url_detected(self):
        assert self.agent._is_url("https://fulbright.org/apply") is True

    def test_http_url_detected(self):
        assert self.agent._is_url("http://example.com") is True

    def test_plain_text_not_url(self):
        assert self.agent._is_url(SAMPLE_SCHOLARSHIP_TEXT) is False


class TestTextValidation:
    def setup_method(self):
        self.agent = ScholarshipAnalysisAgent(api_key="fake")

    def test_empty_text_raises(self):
        with pytest.raises(FetchError, match="too short"):
            self.agent._validate_text("")

    def test_short_text_raises(self):
        with pytest.raises(FetchError, match="too short"):
            self.agent._validate_text("Short text.")

    def test_valid_text_passes(self):
        self.agent._validate_text(SAMPLE_SCHOLARSHIP_TEXT)


class TestRubricParsing:
    def setup_method(self):
        self.agent = ScholarshipAnalysisAgent(api_key="fake")

    def test_valid_extraction_builds_rubric(self):
        rubric = self.agent._build_rubric(MOCK_EXTRACTION, source_url=None)
        assert isinstance(rubric, ScholarshipRubric)
        assert rubric.scholarship_name == "Fulbright Foreign Student Program"
        assert len(rubric.eligibility) == 4
        assert len(rubric.evaluation_criteria) == 4
        assert "academic excellence" in rubric.key_themes

    def test_source_url_injected(self):
        rubric = self.agent._build_rubric(MOCK_EXTRACTION, source_url="https://fulbright.org")
        assert rubric.source_url == "https://fulbright.org"

    def test_missing_name_raises(self):
        bad = {k: v for k, v in MOCK_EXTRACTION.items() if k != "scholarship_name"}
        with pytest.raises(ExtractionError):
            self.agent._build_rubric(bad, source_url=None)

    def test_no_criteria_raises(self):
        bad = {**MOCK_EXTRACTION, "eligibility": [], "evaluation_criteria": []}
        with pytest.raises(ExtractionError):
            self.agent._build_rubric(bad, source_url=None)


class TestMockedAPICall:
    def _make_mock_response(self, data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.input = data
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "tool_use"
        return resp

    @patch("src.agents.scholarship_analysis_agent.anthropic.Anthropic")
    def test_run_with_text_returns_rubric(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(MOCK_EXTRACTION)

        agent = ScholarshipAnalysisAgent(api_key="fake")
        rubric = agent.run(SAMPLE_SCHOLARSHIP_TEXT)

        assert rubric.scholarship_name == "Fulbright Foreign Student Program"
        assert len(rubric.evaluation_criteria) == 4

    @patch("src.agents.scholarship_analysis_agent.anthropic.Anthropic")
    def test_no_tool_block_raises(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        text_block = MagicMock()
        text_block.type = "text"
        resp = MagicMock()
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        mock_client.messages.create.return_value = resp

        agent = ScholarshipAnalysisAgent(api_key="fake")
        with pytest.raises(ExtractionError):
            agent.run(SAMPLE_SCHOLARSHIP_TEXT)
