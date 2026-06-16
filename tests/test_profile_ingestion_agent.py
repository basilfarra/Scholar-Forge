"""
Tests for ProfileIngestionAgent.
Run: cd scholarforge && python -m pytest tests/test_profile_ingestion_agent.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from src.agents.profile_ingestion_agent import ProfileIngestionAgent, ExtractionError
from src.pipeline.schemas import ApplicantProfile

SAMPLE_CV = """
Basil Al-Farra
basil@example.com | Gaza, Palestine

EDUCATION
B.Sc. in Applied Information Technology — Islamic University of Gaza, 2020
GPA: 89/100 (Excellent)
MBA in IT & Communication — Al-Quds Open University, 2023
GPA: 91/100

EXPERIENCE
Backend Developer — Freelance, 2020–Present
- Built RESTful APIs using Laravel serving 50,000+ monthly users
- Reduced API response time by 40% through MySQL query optimization

SKILLS
PHP (Expert), Python (Proficient), Laravel (Expert), FastAPI (Proficient)

LANGUAGES: Arabic (Native), English (Professional)
"""

MOCK_EXTRACTION = {
    "full_name": "Basil Al-Farra",
    "email": "basil@example.com",
    "location": "Gaza, Palestine",
    "academic_records": [
        {"degree": "B.Sc.", "institution": "Islamic University of Gaza",
         "field_of_study": "Applied Information Technology", "end_date": "2020",
         "gpa": 89.0, "gpa_scale": "100"},
        {"degree": "MBA", "institution": "Al-Quds Open University",
         "field_of_study": "IT & Communication", "end_date": "2023",
         "gpa": 91.0, "gpa_scale": "100"},
    ],
    "work_experience": [
        {"role": "Backend Developer", "organization": "Freelance",
         "start_date": "2020", "is_current": True, "employment_type": "freelance",
         "achievements": [
             "Built RESTful APIs using Laravel serving 50,000+ monthly users",
             "Reduced API response time by 40% through MySQL query optimization",
         ],
         "technologies": ["Laravel", "MySQL"]},
    ],
    "technical_skills": [
        {"category": "Programming Languages",
         "skills": [{"name": "PHP", "proficiency": "expert"},
                    {"name": "Python", "proficiency": "proficient"}]},
    ],
    "languages": [
        {"language": "Arabic", "proficiency": "native"},
        {"language": "English", "proficiency": "professional"},
    ],
}


class TestInputValidation:
    def setup_method(self):
        self.agent = ProfileIngestionAgent(api_key="fake")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.agent._validate_input("")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            self.agent._validate_input("Short CV.")

    def test_valid_passes(self):
        self.agent._validate_input(SAMPLE_CV)


class TestProfileParsing:
    def setup_method(self):
        self.agent = ProfileIngestionAgent(api_key="fake")

    def test_valid_extraction_builds_profile(self):
        profile = self.agent._build_profile(MOCK_EXTRACTION)
        assert isinstance(profile, ApplicantProfile)
        assert profile.full_name == "Basil Al-Farra"
        assert len(profile.academic_records) == 2
        assert profile.academic_records[0].gpa == 89.0

    def test_missing_name_raises(self):
        bad = {k: v for k, v in MOCK_EXTRACTION.items() if k != "full_name"}
        with pytest.raises(ExtractionError):
            self.agent._build_profile(bad)

    def test_no_academic_or_work_raises(self):
        with pytest.raises(ExtractionError):
            self.agent._build_profile({"full_name": "Test"})


class TestMockedAPICall:
    def _make_mock_response(self, data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.input = data
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "tool_use"
        return resp

    @patch("src.agents.profile_ingestion_agent.anthropic.Anthropic")
    def test_run_returns_profile(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(MOCK_EXTRACTION)

        agent = ProfileIngestionAgent(api_key="fake")
        profile = agent.run(SAMPLE_CV)
        assert profile.full_name == "Basil Al-Farra"

    @patch("src.agents.profile_ingestion_agent.anthropic.Anthropic")
    def test_no_tool_block_raises(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        text_block = MagicMock()
        text_block.type = "text"
        resp = MagicMock()
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        mock_client.messages.create.return_value = resp

        agent = ProfileIngestionAgent(api_key="fake")
        with pytest.raises(ExtractionError):
            agent.run(SAMPLE_CV)
