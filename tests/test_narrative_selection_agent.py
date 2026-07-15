"""
Tests for NarrativeSelectionAgent.
Run: python -m pytest tests/test_narrative_selection_agent.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from src.agents.narrative_selection_agent import (
    NarrativeSelectionAgent, ExtractionError
)
from src.agents.profile_ingestion_agent import ProfileIngestionAgent
from src.agents.scholarship_analysis_agent import ScholarshipAnalysisAgent
from src.pipeline.schemas import (
    AcademicRecord, ApplicantProfile, EvaluationCriterion, NarrativePlan,
    ScholarshipRubric,
)

SAMPLE_PROFILE = ApplicantProfile(
    full_name="Layla Haddad",
    summary="Computer science student researching low-resource NLP.",
    academic_records=[
        AcademicRecord(
            degree="BSc Computer Science",
            institution="Islamic University of Gaza",
            field_of_study="Computer Science",
            gpa=3.9,
            gpa_scale="4.0",
        ),
    ],
)

SAMPLE_RUBRIC = ScholarshipRubric(
    scholarship_name="DAAD Research Grant",
    evaluation_criteria=[
        EvaluationCriterion(criterion="Research potential", weight="50%"),
        EvaluationCriterion(criterion="Academic excellence", weight="50%"),
    ],
)

MOCK_EXTRACTION = {
    "dominant_frame": "research-driven problem solver",
    "frames_considered": ["community-rooted leader", "research-driven problem solver"],
    "frame_rationale": (
        "The rubric weights research potential and academic excellence equally; "
        "the applicant's GPA and NLP research background map directly onto both, "
        "while the community-leadership angle has weaker evidentiary support."
    ),
    "rubric_alignment": [
        {
            "criterion": "Research potential",
            "evidence": "Ongoing research in low-resource NLP as a CS undergraduate.",
        },
        {
            "criterion": "Academic excellence",
            "evidence": "3.9 GPA on a 4.0 scale in Computer Science.",
        },
    ],
    "rejected_elements": [
        {
            "element": "Volunteer tutoring unrelated to computer science",
            "reason": "Does not reinforce research potential or academic excellence.",
        },
    ],
    "document_guidance": [
        {
            "document_type": "motivation_letter",
            "lead_with": ["NLP research background"],
            "de_emphasize": ["unrelated volunteer work"],
            "angle": "Frame as a researcher whose academic record proves execution.",
        },
        {
            "document_type": "research_proposal",
            "lead_with": ["low-resource NLP research"],
            "de_emphasize": [],
            "angle": "Lead with the technical research gap, not personal narrative.",
        },
    ],
}


class TestPlanParsing:
    def setup_method(self):
        self.agent = NarrativeSelectionAgent(api_key="fake")

    def test_valid_extraction_builds_plan(self):
        plan = self.agent._build_plan(MOCK_EXTRACTION)
        assert isinstance(plan, NarrativePlan)
        assert plan.dominant_frame == "research-driven problem solver"
        assert len(plan.rubric_alignment) == 2
        assert len(plan.rejected_elements) == 1
        assert len(plan.document_guidance) == 2
        assert plan.document_guidance[0].document_type.value == "motivation_letter"

    def test_missing_dominant_frame_raises(self):
        bad = {k: v for k, v in MOCK_EXTRACTION.items() if k != "dominant_frame"}
        with pytest.raises(ExtractionError):
            self.agent._build_plan(bad)

    def test_empty_rubric_alignment_raises(self):
        bad = {**MOCK_EXTRACTION, "rubric_alignment": []}
        with pytest.raises(ExtractionError):
            self.agent._build_plan(bad)

    def test_invalid_document_type_raises(self):
        bad = {
            **MOCK_EXTRACTION,
            "document_guidance": [
                {"document_type": "cover_letter", "angle": "not a real document type"},
            ],
        }
        with pytest.raises(ExtractionError):
            self.agent._build_plan(bad)


class TestMockedAPICall:
    def _make_mock_response(self, data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.input = data
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "tool_use"
        return resp

    @patch("src.agents.narrative_selection_agent.anthropic.Anthropic")
    def test_run_returns_plan(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(MOCK_EXTRACTION)

        agent = NarrativeSelectionAgent(api_key="fake")
        plan = agent.run(SAMPLE_PROFILE, SAMPLE_RUBRIC)

        assert plan.dominant_frame == "research-driven problem solver"
        assert len(plan.rubric_alignment) == 2

    @patch("src.agents.narrative_selection_agent.anthropic.Anthropic")
    def test_no_tool_block_raises(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        text_block = MagicMock()
        text_block.type = "text"
        resp = MagicMock()
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        mock_client.messages.create.return_value = resp

        agent = NarrativeSelectionAgent(api_key="fake")
        with pytest.raises(ExtractionError):
            agent.run(SAMPLE_PROFILE, SAMPLE_RUBRIC)

    def test_model_matches_other_agents(self):
        assert (
            NarrativeSelectionAgent.DEFAULT_MODEL
            == ProfileIngestionAgent.DEFAULT_MODEL
            == ScholarshipAnalysisAgent.DEFAULT_MODEL
        )
