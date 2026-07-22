"""
Integration test for the ProfileIngestionAgent -> ScholarshipAnalysisAgent ->
NarrativeSelectionAgent pipeline.

Scope: this test validates TYPE AND INTERFACE COMPATIBILITY across the real
agent chain — not narrative quality or semantic coherence. It proves that the
actual Pydantic objects one agent returns are accepted by the next agent's
real method signature, with no TypeErrors, missing-field errors, or silent
attribute mismatches. Only the Anthropic API client boundary is mocked;
everything else (Pydantic construction, agent internals) runs for real.
Quality assessment requires a live API call and is deferred until API
credits are available.

Run: python -m pytest tests/test_pipeline_integration.py -v
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.profile_ingestion_agent import ProfileIngestionAgent, ExtractionError
from src.agents.scholarship_analysis_agent import ScholarshipAnalysisAgent
from src.agents.narrative_selection_agent import NarrativeSelectionAgent
from src.pipeline.schemas import ApplicantProfile, ScholarshipRubric, NarrativePlan

SAMPLE_CV = """
Layla Haddad
layla.haddad@example.com | Gaza, Palestine

EDUCATION
B.Sc. in Computer Science — Islamic University of Gaza, 2023
GPA: 3.9 / 4.0

RESEARCH EXPERIENCE
Research Assistant — NLP Lab, Islamic University of Gaza, 2022-Present
- Built low-resource machine translation models for Arabic dialects
- Co-authored a paper on dialectal Arabic NLP submitted to a regional workshop

SKILLS
Python (Expert), PyTorch (Proficient), Arabic NLP (Expert)

LANGUAGES: Arabic (Native), English (Professional)
"""

SAMPLE_SCHOLARSHIP_TEXT = """
DAAD Research Grant for Doctoral Candidates

The DAAD Research Grant supports outstanding international graduate students
pursuing doctoral research at German universities in any field.

Eligibility:
- Hold a master's degree or equivalent before the grant begins
- Minimum GPA of 3.0 on a 4.0 scale
- Demonstrated research experience in the proposed field

Award Details:
Monthly stipend, travel allowance, and health insurance for up to 3 years.

Evaluation Criteria:
- Research potential and quality of proposal (50%)
- Academic excellence (50%)

Required Documents:
- Research proposal
- Academic transcripts
- Two letters of recommendation

Deadline: November 1, 2026
"""

PROFILE_EXTRACTION = {
    "full_name": "Layla Haddad",
    "email": "layla.haddad@example.com",
    "location": "Gaza, Palestine",
    "academic_records": [
        {
            "degree": "B.Sc.",
            "institution": "Islamic University of Gaza",
            "field_of_study": "Computer Science",
            "end_date": "2023",
            "gpa": 3.9,
            "gpa_scale": "4.0",
        },
    ],
    "work_experience": [
        {
            "role": "Research Assistant",
            "organization": "NLP Lab, Islamic University of Gaza",
            "start_date": "2022",
            "is_current": True,
            "employment_type": "part_time",
            "achievements": [
                "Built low-resource machine translation models for Arabic dialects",
                "Co-authored a paper on dialectal Arabic NLP submitted to a regional workshop",
            ],
            "technologies": ["Python", "PyTorch"],
        },
    ],
    "technical_skills": [
        {
            "category": "Programming",
            "skills": [
                {"name": "Python", "proficiency": "expert"},
                {"name": "PyTorch", "proficiency": "proficient"},
            ],
        },
    ],
    "languages": [
        {"language": "Arabic", "proficiency": "native"},
        {"language": "English", "proficiency": "professional"},
    ],
}

RUBRIC_EXTRACTION = {
    "scholarship_name": "DAAD Research Grant for Doctoral Candidates",
    "provider": "DAAD",
    "degree_level": "phd",
    "fields_of_study": ["All fields"],
    "deadline": "November 1, 2026",
    "award_details": "Monthly stipend, travel allowance, health insurance for up to 3 years.",
    "eligibility": [
        {"field": "education", "condition": "Master's degree before grant begins", "is_mandatory": True},
        {"field": "GPA", "condition": "Minimum 3.0 on 4.0 scale", "is_mandatory": True},
    ],
    "evaluation_criteria": [
        {"criterion": "Research potential and quality of proposal", "weight": "50%"},
        {"criterion": "Academic excellence", "weight": "50%"},
    ],
    "required_documents": ["Research proposal", "Academic transcripts", "Two letters of recommendation"],
    "key_themes": ["research potential", "academic excellence"],
    "committee_profile": "Academic reviewers who prioritize research rigor.",
    "red_flags": [],
}

NARRATIVE_EXTRACTION = {
    "dominant_frame": "research-driven problem solver",
    "frames_considered": ["research-driven problem solver", "community-rooted leader"],
    "frame_rationale": (
        "The rubric weights research potential and academic excellence equally; "
        "the applicant's GPA and NLP research background map directly onto both."
    ),
    "rubric_alignment": [
        {
            "criterion": "Research potential and quality of proposal",
            "evidence": "Built low-resource machine translation models for Arabic dialects "
            "as a research assistant, resulting in a co-authored paper.",
        },
        {
            "criterion": "Academic excellence",
            "evidence": "3.9 GPA on a 4.0 scale in Computer Science.",
        },
    ],
    "rejected_elements": [],
    "document_guidance": [
        {
            "document_type": "research_proposal",
            "lead_with": ["low-resource Arabic NLP research"],
            "de_emphasize": [],
            "angle": "Lead with the technical research gap and prior results.",
        },
    ],
}


def _make_mock_response(data: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "tool_use"
    return resp


class TestPipelineIntegration:
    """
    Only the Anthropic API client is mocked. ProfileIngestionAgent.run() and
    ScholarshipAnalysisAgent.run() produce real ApplicantProfile and
    ScholarshipRubric instances, which are then fed into the real
    NarrativeSelectionAgent.run() — proving the actual inter-agent contract
    holds, not a hand-built stand-in for it.
    """

    def test_full_chain_produces_valid_narrative_plan(self):
        # All three agents do `import anthropic` and share that single module
        # object, so `anthropic.Anthropic` cannot be patched to three
        # different mocks simultaneously — each patch is scoped to just the
        # construction + call of the agent it belongs to, so only one is
        # active at a time.
        with patch("src.agents.profile_ingestion_agent.anthropic.Anthropic") as mock_profile_cls:
            mock_profile_client = MagicMock()
            mock_profile_cls.return_value = mock_profile_client
            mock_profile_client.messages.create.return_value = _make_mock_response(PROFILE_EXTRACTION)

            profile_agent = ProfileIngestionAgent(api_key="fake")
            # Two independent branches — neither depends on the other's output.
            profile = profile_agent.run(SAMPLE_CV)

        with patch("src.agents.scholarship_analysis_agent.anthropic.Anthropic") as mock_rubric_cls:
            mock_rubric_client = MagicMock()
            mock_rubric_cls.return_value = mock_rubric_client
            mock_rubric_client.messages.create.return_value = _make_mock_response(RUBRIC_EXTRACTION)

            rubric_agent = ScholarshipAnalysisAgent(api_key="fake")
            rubric = rubric_agent.run(SAMPLE_SCHOLARSHIP_TEXT)

        assert isinstance(profile, ApplicantProfile)
        assert isinstance(rubric, ScholarshipRubric)

        with patch("src.agents.narrative_selection_agent.anthropic.Anthropic") as mock_narrative_cls:
            mock_narrative_client = MagicMock()
            mock_narrative_cls.return_value = mock_narrative_client
            mock_narrative_client.messages.create.return_value = _make_mock_response(NARRATIVE_EXTRACTION)

            narrative_agent = NarrativeSelectionAgent(api_key="fake")
            # The only step that joins the two branches.
            plan = narrative_agent.run(profile, rubric)

        assert isinstance(plan, NarrativePlan)
        assert plan.dominant_frame == "research-driven problem solver"
        assert len(plan.rubric_alignment) == 2
        assert plan.document_guidance[0].document_type.value == "research_proposal"

    @patch("src.agents.profile_ingestion_agent.anthropic.Anthropic")
    def test_invalid_profile_fails_before_reaching_downstream_agents(self, mock_profile_cls):
        """
        A profile extraction with neither academic_records nor work_experience
        must fail ApplicantProfile's own validator immediately — the chain
        should never propagate bad data to Agent 2 or 3.
        """
        incomplete_extraction = {
            k: v for k, v in PROFILE_EXTRACTION.items()
            if k not in ("academic_records", "work_experience")
        }

        mock_profile_client = MagicMock()
        mock_profile_cls.return_value = mock_profile_client
        mock_profile_client.messages.create.return_value = _make_mock_response(incomplete_extraction)

        profile_agent = ProfileIngestionAgent(api_key="fake")

        with pytest.raises(ExtractionError):
            profile_agent.run(SAMPLE_CV)
