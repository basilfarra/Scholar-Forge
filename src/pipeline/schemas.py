"""
Inter-agent data contracts for the ScholarForge pipeline.
All schemas are Pydantic models with strict validation.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ─── Enumerations ──────────────────────────────────────────────────────────────

class SkillProficiency(str, Enum):
    EXPERT     = "expert"
    PROFICIENT = "proficient"
    FAMILIAR   = "familiar"

class LanguageProficiency(str, Enum):
    NATIVE       = "native"
    FLUENT       = "fluent"
    PROFESSIONAL = "professional"
    INTERMEDIATE = "intermediate"
    BASIC        = "basic"

class EmploymentType(str, Enum):
    FULL_TIME  = "full_time"
    PART_TIME  = "part_time"
    FREELANCE  = "freelance"
    INTERNSHIP = "internship"
    VOLUNTEER  = "volunteer"

class DegreeLevel(str, Enum):
    BACHELOR = "bachelor"
    MASTER   = "master"
    PHD      = "phd"
    ANY      = "any"


# ─── ApplicantProfile nested models ────────────────────────────────────────────

class AcademicRecord(BaseModel):
    degree: str
    institution: str
    field_of_study: str
    start_date: Optional[str]        = None
    end_date: Optional[str]          = None
    gpa: Optional[float]             = None
    gpa_scale: Optional[str]         = None
    grade_description: Optional[str] = None
    honors: list[str]                = Field(default_factory=list)
    relevant_courses: list[str]      = Field(default_factory=list)
    thesis_title: Optional[str]      = None
    model_config = {"extra": "forbid"}

class Skill(BaseModel):
    name: str
    proficiency: Optional[SkillProficiency] = None
    years_experience: Optional[float]       = None
    model_config = {"extra": "forbid"}

class SkillCategory(BaseModel):
    category: str
    skills: list[Skill] = Field(default_factory=list)
    model_config = {"extra": "forbid"}

class WorkExperience(BaseModel):
    role: str
    organization: str
    start_date: Optional[str]             = None
    end_date: Optional[str]               = None
    is_current: bool                      = False
    location: Optional[str]               = None
    employment_type: Optional[EmploymentType] = None
    achievements: list[str]               = Field(default_factory=list)
    technologies: list[str]               = Field(default_factory=list)
    model_config = {"extra": "forbid"}

class Publication(BaseModel):
    title: str
    venue: Optional[str]  = None
    year: Optional[int]   = None
    doi: Optional[str]    = None
    co_authors: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}

class Project(BaseModel):
    name: str
    description: Optional[str]  = None
    technologies: list[str]     = Field(default_factory=list)
    url: Optional[str]          = None
    impact: Optional[str]       = None
    model_config = {"extra": "forbid"}

class Certification(BaseModel):
    name: str
    issuer: Optional[str]        = None
    date: Optional[str]          = None
    credential_id: Optional[str] = None
    model_config = {"extra": "forbid"}

class Award(BaseModel):
    title: str
    issuer: Optional[str]      = None
    year: Optional[int]        = None
    description: Optional[str] = None
    model_config = {"extra": "forbid"}

class LanguageSkill(BaseModel):
    language: str
    proficiency: LanguageProficiency
    model_config = {"extra": "forbid"}

class Extracurricular(BaseModel):
    activity: str
    role: Optional[str]         = None
    organization: Optional[str] = None
    duration: Optional[str]     = None
    description: Optional[str]  = None
    model_config = {"extra": "forbid"}


# ─── Primary Schemas ───────────────────────────────────────────────────────────

class ApplicantProfile(BaseModel):
    """Output of ProfileIngestionAgent."""
    full_name: str
    email: Optional[str]        = None
    phone: Optional[str]        = None
    location: Optional[str]     = None
    nationality: Optional[str]  = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str]   = None
    summary: Optional[str]      = None

    academic_records: list[AcademicRecord]  = Field(default_factory=list)
    work_experience:  list[WorkExperience]  = Field(default_factory=list)
    technical_skills: list[SkillCategory]   = Field(default_factory=list)
    publications:     list[Publication]     = Field(default_factory=list)
    projects:         list[Project]         = Field(default_factory=list)
    certifications:   list[Certification]   = Field(default_factory=list)
    awards:           list[Award]           = Field(default_factory=list)
    languages:        list[LanguageSkill]   = Field(default_factory=list)
    extracurricular:  list[Extracurricular] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def must_have_academic_or_work(self) -> "ApplicantProfile":
        if not self.academic_records and not self.work_experience:
            raise ValueError(
                "ApplicantProfile must contain at least one academic record "
                "or work experience entry."
            )
        return self


# ─── EligibilityRequirement ────────────────────────────────────────────────────

class EligibilityRequirement(BaseModel):
    """A single eligibility rule extracted from a scholarship page."""
    field: str              # e.g. "nationality", "GPA", "degree_level", "age"
    condition: str          # e.g. "minimum 3.0 on 4.0 scale"
    is_mandatory: bool = True
    notes: Optional[str] = None
    model_config = {"extra": "forbid"}


class EvaluationCriterion(BaseModel):
    """A single weighted evaluation dimension."""
    criterion: str          # e.g. "Academic excellence"
    weight: Optional[str]   = None   # e.g. "40%" or "High priority"
    description: Optional[str] = None
    model_config = {"extra": "forbid"}


class ScholarshipRubric(BaseModel):
    """
    Output of ScholarshipAnalysisAgent.
    Structured representation of what a scholarship looks for.
    """
    scholarship_name: str
    provider: Optional[str]          = None
    source_url: Optional[str]        = None
    degree_level: Optional[DegreeLevel] = None
    fields_of_study: list[str]       = Field(default_factory=list)
    deadline: Optional[str]          = None
    award_details: Optional[str]     = None   # amount, duration, coverage

    eligibility:          list[EligibilityRequirement] = Field(default_factory=list)
    evaluation_criteria:  list[EvaluationCriterion]    = Field(default_factory=list)
    required_documents:   list[str]                    = Field(default_factory=list)

    key_themes: list[str]            = Field(default_factory=list)
    # e.g. ["leadership", "community impact", "research potential"]

    committee_profile: Optional[str] = None
    # Who typically sits on this committee? What do they value?

    red_flags: list[str]             = Field(default_factory=list)
    # Hard disqualifiers explicitly stated

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def must_have_criteria_or_eligibility(self) -> "ScholarshipRubric":
        if not self.eligibility and not self.evaluation_criteria:
            raise ValueError(
                "ScholarshipRubric must contain at least eligibility "
                "requirements or evaluation criteria."
            )
        return self


class NarrativePlan(BaseModel):
    """Output of NarrativeSelectionAgent. Placeholder for Phase 1 completion."""
    primary_narrative: str
    key_evidence_points: list[str]   = Field(default_factory=list)
    documents_to_produce: list[str]  = Field(default_factory=list)
    model_config = {"extra": "forbid"}
