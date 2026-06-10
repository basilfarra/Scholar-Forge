"""
Pydantic schemas for inter-agent communication.

Every piece of data that flows between agents in ScholarForge is validated
against one of these models. There is no free-text passing between agents.

This file is the single source of truth for the pipeline's data contracts.
Any change here must be reflected in the corresponding agent prompts
(src/pipeline/prompts/*.txt) and in the agent implementations.

Schemas are grouped by pipeline stage:
    1. Ingestion stage   — raw input and structured profile
    2. Analysis stage    — scholarship rubric extraction
    3. Selection stage   — narrative selection decisions
    4. Generation stage  — generated documents (letter, CV, proposal, email)
    5. Validation stage  — evidence audit, cliché detection, coherence, critique
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────
# Shared enums and primitives
# ──────────────────────────────────────────────────────────────────────

class Language(str, Enum):
    EN = "en"
    AR = "ar"
    FR = "fr"
    DE = "de"


class Region(str, Enum):
    """Cultural register regions for the Cultural Adapter agent."""
    GERMANY = "germany"
    UK = "uk"
    USA = "usa"
    FRANCE = "france"
    NORDIC = "nordic"
    MIDDLE_EAST = "middle_east"
    EAST_ASIA = "east_asia"
    OTHER = "other"


class Weight(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    CONTEXTUAL = "contextual"


class StrictBaseModel(BaseModel):
    """Base for all schemas — forbids extra fields to catch typos early."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ════════════════════════════════════════════════════════════════════════
# STAGE 1: INGESTION
# ════════════════════════════════════════════════════════════════════════

class RawNarrative(StrictBaseModel):
    """Input to the Profile Ingestion Agent — applicant's story in raw form."""
    text: str = Field(..., min_length=100, description="Full applicant narrative")
    language: Language = Language.EN
    applicant_id: str | None = None


# ── Profile sub-models ────────────────────────────────────────────────

class ThesisOrProject(StrictBaseModel):
    title: str
    description: str
    score: str | None = None
    technologies: list[str] = Field(default_factory=list)
    methodology: str | None = None


class Education(StrictBaseModel):
    degree: str
    field: str
    institution: str
    year: int | None = None
    gpa_or_score: str | None = None
    distinction: str | None = None
    key_courses: list[str] = Field(default_factory=list)
    thesis_or_project: ThesisOrProject | None = None


class Achievement(StrictBaseModel):
    claim: str
    metric: str | None = None
    evidence_type: EvidenceType = EvidenceType.CONTEXTUAL


class Professional(StrictBaseModel):
    role: str
    organization: str
    duration: str
    employment_type: Literal["full-time", "freelance", "contract", "internship", "apprenticeship"]
    location: str | None = None
    modality: Literal["remote", "hybrid", "on-site"] | None = None
    achievements: list[Achievement] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Certification(StrictBaseModel):
    name: str
    provider: str
    category: Literal["AI", "SE", "QA", "UX", "Management", "Data", "Other"] = "Other"
    year: int | None = None


class MentorshipRecord(StrictBaseModel):
    role: str
    scope: str
    beneficiaries_count: int | None = None
    context: str | None = None


class CommunityEngagement(StrictBaseModel):
    activity: str
    impact: str
    context: str | None = None


class Award(StrictBaseModel):
    title: str
    context: str | None = None
    significance: str | None = None
    year: int | None = None


class NarrativeArc(StrictBaseModel):
    """
    A coherent story thread extracted from the raw narrative.
    Multiple facts may belong to the same arc.
    """
    arc_id: str
    summary: str
    supporting_facts: list[str] = Field(
        default_factory=list,
        description="References to other profile elements by path or id",
    )
    emotional_weight: Weight = Weight.MEDIUM
    academic_relevance: Weight = Weight.MEDIUM


class Constraints(StrictBaseModel):
    """Contextual constraints relevant to interpretation."""
    geographic: str | None = None
    infrastructure: str | None = None
    conflict_context: str | None = None


class PersonalInfo(StrictBaseModel):
    name: str
    location: str | None = None
    nationality: str | None = None
    languages: list[str] = Field(default_factory=list)


class Profile(StrictBaseModel):
    """
    Structured applicant profile — the single source of truth.

    Produced by the Profile Ingestion Agent. All downstream agents
    that need to reference applicant facts query this object.
    """
    personal: PersonalInfo
    education: list[Education] = Field(default_factory=list)
    professional: list[Professional] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    mentorship_and_teaching: list[MentorshipRecord] = Field(default_factory=list)
    community_engagement: list[CommunityEngagement] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    narrative_arcs: list[NarrativeArc] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)


# ════════════════════════════════════════════════════════════════════════
# STAGE 2: SCHOLARSHIP ANALYSIS
# ════════════════════════════════════════════════════════════════════════

class ScholarshipPosting(StrictBaseModel):
    """Input to the Scholarship Analysis Agent."""
    posting_text: str = Field(..., min_length=50)
    posting_url: str | None = None
    scholarship_name: str
    institution: str
    country: str
    field: str | None = None


class RegisterStyle(str, Enum):
    """Tone register for academic writing — distinct per cultural context."""
    FORMAL_ACADEMIC = "formal-academic"
    PROFESSIONAL = "professional"
    NARRATIVE_PERSONAL = "narrative-personal"


class StatedCriterion(StrictBaseModel):
    criterion: str
    weight: Weight
    evidence_in_posting: str = Field(
        ..., description="Direct quote or paraphrase from the posting"
    )


class ImpliedCriterion(StrictBaseModel):
    criterion: str
    inference_basis: str
    confidence: Confidence


class FormatRequirements(StrictBaseModel):
    word_limit: int | None = None
    required_sections: list[str] = Field(default_factory=list)
    language: Language = Language.EN
    additional_documents: list[str] = Field(default_factory=list)


class InstitutionalCulture(StrictBaseModel):
    region: Region
    register_style: RegisterStyle
    values_emphasis: list[str] = Field(default_factory=list)
    known_preferences: str | None = None


class Disqualifier(StrictBaseModel):
    condition: str
    source: Literal["stated", "inferred"]


class Rubric(StrictBaseModel):
    """
    Reverse-engineered evaluation rubric for a specific scholarship.

    Produced by the Scholarship Analysis Agent. Used by Narrative
    Selection (to choose what to include), Document Generators
    (to address each criterion), and Skeptical Reviewer (to flag gaps).
    """
    scholarship_name: str
    institution: str
    country: str
    stated_criteria: list[StatedCriterion] = Field(default_factory=list)
    implied_criteria: list[ImpliedCriterion] = Field(default_factory=list)
    format_requirements: FormatRequirements = Field(default_factory=FormatRequirements)
    institutional_culture: InstitutionalCulture
    disqualifiers: list[Disqualifier] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# STAGE 3: NARRATIVE SELECTION
# ════════════════════════════════════════════════════════════════════════

class SelectedElement(StrictBaseModel):
    profile_element: str = Field(
        ..., description="Reference path in profile.json, e.g. 'education[0].thesis'"
    )
    mapped_to_criterion: str
    justification: str
    presentation_angle: str


class RejectedElement(StrictBaseModel):
    profile_element: str
    reason: Literal[
        "irrelevant",
        "redundant",
        "weakens_application",
        "better_alternative_exists",
    ]
    explanation: str | None = None


class Gap(StrictBaseModel):
    rubric_criterion: str
    gap_description: str
    mitigation: str | None = None


class NarrativeSelection(StrictBaseModel):
    """
    Editorial decisions made by the Narrative Selection Agent.

    Crucially includes both `selected` and `rejected` items — making
    explicit what was excluded and why. This is how we avoid the
    "kitchen sink" failure mode common in applicant-written documents.
    """
    selected: list[SelectedElement] = Field(..., min_length=1, max_length=8)
    rejected: list[RejectedElement] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# STAGE 4: DOCUMENT GENERATION
# ════════════════════════════════════════════════════════════════════════

class GenerationContext(StrictBaseModel):
    """Shared input for all document generation agents."""
    profile: Profile
    rubric: Rubric
    selection: NarrativeSelection
    target_region: Region


class MotivationLetter(StrictBaseModel):
    """Output of the Motivation Letter Agent."""
    body: str = Field(..., description="Full letter text, plain markdown")
    word_count: int
    paragraph_count: int
    rubric_criteria_addressed: list[str]
    evidence_pointers: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps each factual claim to profile element paths",
    )


class CVSection(StrictBaseModel):
    heading: str
    entries: list[dict[str, Any]]


class AcademicCV(StrictBaseModel):
    """Output of the Academic CV Agent."""
    sections: list[CVSection]
    target_field: str
    ats_score_estimate: float = Field(..., ge=0.0, le=1.0)
    embedded_keywords: list[str] = Field(default_factory=list)


class ResearchProposal(StrictBaseModel):
    """Output of the Research Proposal Agent."""
    title: str
    research_question: str
    gap_analysis: str
    methodology: str
    timeline: str
    expected_outcomes: str
    impact: str
    field: str
    feasibility_notes: list[str] = Field(default_factory=list)


class EmailType(str, Enum):
    SUPERVISOR_INQUIRY = "supervisor_inquiry"
    ADMISSIONS_CLARIFICATION = "admissions_clarification"
    POST_SUBMISSION_FOLLOWUP = "post_submission_followup"
    REFERENCE_REQUEST = "reference_request"
    REFERENCE_REMINDER = "reference_reminder"
    REFERENCE_THANKYOU = "reference_thankyou"


class AcademicEmail(StrictBaseModel):
    """Output of the Academic Email Agent."""
    email_type: EmailType
    subject: str = Field(..., max_length=100)
    body: str
    word_count: int
    formality_level: Literal["high", "medium", "low"]


# ════════════════════════════════════════════════════════════════════════
# STAGE 5: VALIDATION
# ════════════════════════════════════════════════════════════════════════

class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUBSTANTIATED = "unsubstantiated"
    POTENTIAL_HALLUCINATION = "potential_hallucination"


class ClaimVerification(StrictBaseModel):
    claim: str
    source: str | None = None
    status: ClaimStatus
    explanation: str | None = None


class EvidenceAudit(StrictBaseModel):
    """Output of the Evidence Auditor."""
    total_claims: int
    verified: int
    partially_supported: int
    unsubstantiated: int
    potential_hallucination: int
    claim_map: list[ClaimVerification]

    @property
    def passes(self) -> bool:
        """True if no hallucinations and zero unsupported critical claims."""
        return self.potential_hallucination == 0 and self.unsubstantiated == 0


class ClicheSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    NOTE = "note"


class ClicheMatch(StrictBaseModel):
    phrase: str
    location: str = Field(..., description="Paragraph or character offset")
    severity: ClicheSeverity
    detection_method: Literal["exact_match", "pattern_match", "semantic_match"]
    suggested_alternative: str | None = None


class ClicheReport(StrictBaseModel):
    """Output of the Cliché Detector."""
    matches: list[ClicheMatch] = Field(default_factory=list)
    critical_count: int
    warning_count: int
    note_count: int

    @property
    def passes(self) -> bool:
        """Pass criterion: zero critical clichés, at most three warnings."""
        return self.critical_count == 0 and self.warning_count <= 3


class Contradiction(StrictBaseModel):
    document_a: str
    document_b: str
    issue: str
    severity: Literal["high", "medium", "low"]


class CoherenceReport(StrictBaseModel):
    """Output of the Coherence Validator."""
    contradictions: list[Contradiction] = Field(default_factory=list)
    passes: bool


class ReviewerVerdict(str, Enum):
    SHORTLIST = "shortlist"
    BORDERLINE = "borderline"
    REJECT = "reject"


class WeakestParagraph(StrictBaseModel):
    paragraph_number: int
    reason: str


class SkepticalReview(StrictBaseModel):
    """Output of the Skeptical Reviewer."""
    overall_verdict: ReviewerVerdict
    attention_loss_point: str | None = None
    weakest_paragraph: WeakestParagraph | None = None
    inflated_claims: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(default_factory=list)
    improvement_priorities: list[str] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# PIPELINE-WIDE RESULT
# ════════════════════════════════════════════════════════════════════════

class PipelineResult(StrictBaseModel):
    """
    Final output of a full ScholarForge run.

    Aggregates every artifact produced across the pipeline so the
    applicant sees not only the final documents but also the
    validation reports that vouch for them.
    """
    run_id: str
    started_at: datetime
    completed_at: datetime

    # Generated artifacts (any may be None if not requested)
    motivation_letter: MotivationLetter | None = None
    academic_cv: AcademicCV | None = None
    research_proposal: ResearchProposal | None = None
    academic_emails: list[AcademicEmail] = Field(default_factory=list)

    # Validation reports
    evidence_audit: EvidenceAudit | None = None
    cliche_report: ClicheReport | None = None
    coherence_report: CoherenceReport | None = None
    skeptical_review: SkepticalReview | None = None

    # Pipeline metadata
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
