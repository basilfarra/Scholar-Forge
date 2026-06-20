

<p align="center">
<img width="1774" height="887" alt="ScholarForge" src="https://github.com/user-attachments/assets/7d2c6a51-d525-44a5-bd34-4767f685c9ec" />
</p>

<h1 align="center">ScholarForge</h1>

<p align="center">
  <strong>Multi-agent AI engine for competitive academic applications.</strong><br>
  Not a letter generator. An orchestration system that reads applicants the way committees do.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLMs-Multi--Model-blueviolet" />
  <img src="https://img.shields.io/badge/agents-10-orange" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/status-active%20development-yellow" />
</p>

---

## The Problem

Every year, thousands of students apply to international scholarships using AI-generated motivation letters that read identically: *"Since childhood, I have been passionate about..."*

Committees notice. Applications get rejected not because candidates lack merit, but because their documents fail to communicate it. The existing tools — single-prompt generators that produce generic, template-driven text — make this worse, not better.

**ScholarForge takes a fundamentally different approach.**

Instead of generating text from a template, it operates as a **10-agent orchestration pipeline** that:

- Ingests the applicant's full academic and professional narrative
- Reverse-engineers the scholarship's evaluation rubric
- Selects which parts of the story serve this specific application
- Generates documents against the rubric, not against a template
- Validates every claim against the source profile
- Critiques the output through a simulated hostile reviewer
- Detects and eliminates clichés at the semantic level

The result: documents that are structurally sound, evidence-backed, and difficult to dismiss.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCHOLARFORGE PIPELINE                     │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐      │
│  │ Applicant │───▶│   Profile    │───▶│    Narrative      │      │
│  │   Story   │    │  Ingestion   │    │    Selection      │      │
│  └──────────┘    │   Agent      │    │    Agent          │      │
│                  └──────────────┘    └────────┬──────────┘      │
│                                               │                  │
│  ┌──────────┐    ┌──────────────┐             │                  │
│  │Scholarship│───▶│  Scholarship │─────────────┘                 │
│  │  Posting  │    │   Analysis   │                               │
│  └──────────┘    │   Agent      │                                │
│                  └──────────────┘                                │
│                                                                  │
│                  ┌──────────────────────────────────┐            │
│                  │     GENERATION LAYER              │            │
│                  │  ┌────────────┐ ┌──────────────┐ │            │
│                  │  │ Motivation │ │ Academic CV  │ │            │
│                  │  │  Letter    │ │  (ATS-Ready) │ │            │
│                  │  └────────────┘ └──────────────┘ │            │
│                  │  ┌────────────┐ ┌──────────────┐ │            │
│                  │  │  Research  │ │  Academic    │ │            │
│                  │  │  Proposal  │ │   Emails     │ │            │
│                  │  └────────────┘ └──────────────┘ │            │
│                  └──────────────────────────────────┘            │
│                                                                  │
│                  ┌──────────────────────────────────┐            │
│                  │     VALIDATION LAYER              │            │
│                  │  ┌────────┐ ┌────────┐ ┌───────┐ │           │
│                  │  │Evidence│ │Cliché  │ │Culture│ │           │
│                  │  │Auditor │ │Detector│ │Adapter│ │           │
│                  │  └────────┘ └────────┘ └───────┘ │           │
│                  │  ┌──────────┐ ┌────────────────┐ │           │
│                  │  │Coherence │ │   Skeptical    │ │           │
│                  │  │Validator │ │   Reviewer     │ │           │
│                  │  └──────────┘ └────────────────┘ │           │
│                  └──────────────────────────────────┘            │
│                                                                  │
│                  ┌──────────────────────────────────┐            │
│                  │     INFRASTRUCTURE                │            │
│                  │  Cost Orchestrator · Telemetry    │            │
│                  │  Multi-Model Routing · Logging    │            │
│                  └──────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agents

ScholarForge uses **10 specialized agents**, each with a single responsibility. No agent generates final output alone — every document passes through ingestion, selection, generation, and validation before it reaches the applicant.

| # | Agent | Role | Input | Output |
|---|-------|------|-------|--------|
| 1 | **Profile Ingestion** | Extracts structured facts from raw applicant narrative | Free-text story (any length) | `profile.json` — achievements, dates, metrics, arcs |
| 2 | **Scholarship Analysis** | Reverse-engineers evaluation criteria from posting | Scholarship URL or text | `rubric.json` — stated + implied criteria, weights |
| 3 | **Narrative Selection** | Chooses which story elements serve this application | `profile.json` + `rubric.json` | `selected.json` + `rejected.json` with reasoning |
| 4 | **Document Generators** | Writes against rubric using selected narrative | `selected.json` + `rubric.json` | Draft documents (ML, CV, RP, Emails) |
| 5 | **Evidence Auditor** | Verifies every claim traces to source profile | Generated document + `profile.json` | Flagged unsubstantiated claims |
| 6 | **Cliché Detector** | Identifies template language and overused phrases | Generated document | Cliché report with alternatives |
| 7 | **Cultural Register Adapter** | Adjusts tone for institutional/regional norms | Document + target region | Region-adapted document |
| 8 | **Coherence Validator** | Ensures cross-document consistency | All generated documents | Contradiction report |
| 9 | **Skeptical Reviewer** | Simulates hostile committee member | Final documents | Weakness analysis + rejection risks |
| 10 | **Cost Orchestrator** | Routes tasks to optimal model by complexity | All pipeline tasks | Execution log with cost tracking |

> Full agent specifications: [`docs/AGENTS.md`](docs/AGENTS.md)

---

## What It Generates

### Motivation Letter
- Structured against the scholarship's actual rubric, not a generic template
- Every claim backed by evidence from the applicant's profile
- Hard rules enforced:
  - ❌ No "Since childhood..." / "I have always been passionate..."
  - ❌ No unsubstantiated personality claims
  - ❌ No emotional narratives disconnected from academic merit
  - ❌ No sentences exceeding 30 words
  - ✅ Every paragraph maps to a rubric criterion
  - ✅ Region-appropriate register (German ≠ British ≠ American)
  - ✅ Hook → Trajectory → Fit → Contribution → Ask

### Academic CV (ATS-Ready)
- Optimized for academic ATS systems (distinct from corporate ATS)
- Keyword density calibrated to the target field
- Structure: Education → Research → Publications → Technical → Awards → Service
- No graphics, columns, or icons — pure semantic structure
- Output: LaTeX + Word + PDF from single source
- Includes ATS compatibility score estimate

### Research Proposal
- Field-aware structure (STEM ≠ Humanities ≠ Social Sciences)
- Components: Research Question → Gap Analysis → Methodology → Timeline → Expected Outcomes → Impact
- Methodology validated against program duration
- Detects: overly broad claims, methodology mismatches, impossible timelines

### Academic Emails
- Supervisor inquiry (PhD/Masters)
- Admissions office clarification
- Post-submission follow-up
- Reference request / reminder / thank-you
- Each type: calibrated length, formality level, optimized subject line
- Anti-pattern: prevents over-apologizing (common applicant mistake)

---

## What Makes This Different

Most "AI application tools" are single-prompt wrappers. ScholarForge is architecturally different:

| Capability | Generic Tools | ScholarForge |
|-----------|--------------|--------------|
| Generation method | Single prompt → output | 10-agent pipeline with validation |
| Rubric awareness | None — uses templates | Reverse-engineers criteria from posting |
| Narrative selection | User picks highlights | Agent selects + justifies + rejects |
| Claim verification | None | Evidence Auditor traces every claim |
| Cliché detection | None | 200+ pattern database + semantic matching |
| Cross-document coherence | None | Coherence Validator across all outputs |
| Cultural adaptation | None | Region-specific register rules |
| Quality assurance | None | Skeptical Reviewer simulates hostile committee |
| Cost optimization | Single model, full price | Multi-model routing by task complexity |
| Observability | None | Full telemetry: latency, cost, token usage per agent |

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Runtime** | Python 3.11+ | Industry standard for AI/ML pipelines |
| **API Framework** | FastAPI | Async-native, auto-documented, type-safe |
| **LLM Providers** | Anthropic Claude · OpenAI · Google Gemini | Multi-model orchestration — not locked to one vendor |
| **Vector Store** | ChromaDB | Local-first semantic search for cliché detection |
| **Frontend** | React + Tailwind CSS | Minimal UI for demo and interaction |
| **Database** | SQLite (MVP) → PostgreSQL | Local-first, zero-config for development |
| **Document Output** | LaTeX · python-docx · ReportLab | Multi-format generation from single source |
| **Logging** | Structured JSON | Full pipeline observability |
| **Deployment** | Railway / Render | Demo-ready with public URLs |

---

## Project Structure

```
scholarforge/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry
│   ├── config.py                   # Environment and model configuration
│   │
│   ├── agents/                     # Core agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base class for all agents
│   │   ├── profile_ingestion.py    # Raw narrative → structured profile
│   │   ├── scholarship_analysis.py # Posting → rubric extraction
│   │   ├── narrative_selection.py  # Profile + rubric → selected elements
│   │   ├── motivation_letter.py    # Selected narrative → letter draft
│   │   ├── academic_cv.py          # Profile → ATS-optimized CV
│   │   ├── research_proposal.py    # Profile + field → proposal draft
│   │   ├── academic_email.py       # Context → calibrated email
│   │   ├── evidence_auditor.py     # Document → claim verification
│   │   ├── cliche_detector.py      # Document → cliché report
│   │   ├── cultural_adapter.py     # Document + region → adapted output
│   │   ├── coherence_validator.py  # All docs → consistency check
│   │   ├── skeptical_reviewer.py   # Final docs → weakness analysis
│   │   └── cost_orchestrator.py    # Task → optimal model routing
│   │
│   ├── pipeline/                   # Orchestration logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main pipeline controller
│   │   ├── schemas.py              # Pydantic models for all data flows
│   │   └── prompts/                # System prompts per agent
│   │       ├── profile_ingestion.txt
│   │       ├── scholarship_analysis.txt
│   │       ├── narrative_selection.txt
│   │       ├── motivation_letter.txt
│   │       ├── academic_cv.txt
│   │       ├── research_proposal.txt
│   │       ├── academic_email.txt
│   │       ├── evidence_auditor.txt
│   │       ├── cliche_detector.txt
│   │       ├── cultural_adapter.txt
│   │       ├── coherence_validator.txt
│   │       └── skeptical_reviewer.txt
│   │
│   ├── knowledge/                  # Domain knowledge databases
│   │   ├── cliches.json            # 200+ flagged phrases with categories
│   │   ├── register_rules.json     # Cultural norms per region
│   │   ├── rubric_templates.json   # Common scholarship rubric patterns
│   │   └── ats_keywords.json       # Field-specific keyword databases
│   │
│   ├── providers/                  # LLM provider abstractions
│   │   ├── __init__.py
│   │   ├── base_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   └── google_provider.py
│   │
│   ├── output/                     # Document formatters
│   │   ├── __init__.py
│   │   ├── markdown_formatter.py
│   │   ├── docx_formatter.py
│   │   ├── latex_formatter.py
│   │   └── pdf_formatter.py
│   │
│   └── telemetry/                  # Observability
│       ├── __init__.py
│       ├── logger.py               # Structured event logging
│       ├── cost_tracker.py         # Per-agent cost accounting
│       └── evaluation.py           # Output quality metrics
│
├── frontend/                       # React UI
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── pages/
│   └── public/
│
├── tests/                          # Test suite
│   ├── test_agents/
│   ├── test_pipeline/
│   ├── test_providers/
│   └── fixtures/                   # Sample profiles and scholarships
│
├── docs/                           # Extended documentation
│   ├── AGENTS.md                   # Detailed agent specifications
│   ├── ARCHITECTURE.md             # System design decisions
│   ├── PROMPT_ENGINEERING.md       # Prompt design methodology
│   └── EVALUATION.md              # How output quality is measured
│
├── examples/                       # Usage examples
│   ├── sample_profile.json
│   ├── sample_rubric.json
│   └── sample_output/
│
└── assets/                         # Images and diagrams
    └── scholarforge-banner.png
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/basilfarra/scholarforge.git
cd scholarforge

# Environment
cp .env.example .env
# Add your API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY (optional)

# Install
pip install -e .

# Run
uvicorn src.main:app --reload

# Open
# API docs: http://localhost:8000/docs
# Frontend: http://localhost:3000
```

---

## API Overview

```http
POST /api/v1/profile/ingest
# Input: raw applicant narrative (text)
# Output: structured profile JSON

POST /api/v1/scholarship/analyze
# Input: scholarship posting (text or URL)
# Output: extracted rubric JSON

POST /api/v1/generate/motivation-letter
# Input: profile_id + rubric_id + target_region
# Output: motivation letter + evidence map + cliché report

POST /api/v1/generate/academic-cv
# Input: profile_id + target_field + format (latex|docx|pdf)
# Output: ATS-optimized CV + compatibility score

POST /api/v1/generate/research-proposal
# Input: profile_id + field + program_duration
# Output: structured proposal + feasibility assessment

POST /api/v1/generate/academic-email
# Input: profile_id + email_type + recipient_context
# Output: calibrated email + subject line

POST /api/v1/review/critique
# Input: document_id
# Output: skeptical review + weakness map + improvement suggestions

POST /api/v1/review/coherence
# Input: list of document_ids
# Output: cross-document consistency report
```

---

## Design Principles

**1. Rubric-first, not template-first.**
Every document is generated against the scholarship's extracted evaluation criteria. No templates. No "fill in the blanks."

**2. Evidence-backed claims only.**
If a claim in the output cannot be traced to the applicant's profile, it gets flagged. The system does not hallucinate credentials.

**3. Honest narrative, not inflated narrative.**
The Narrative Selection Agent explicitly rejects story elements that don't serve the application — and logs why. This prevents the common failure mode of "throwing everything at the wall."

**4. Multi-model cost intelligence.**
Drafts use efficient models. Refinement uses capable models. Critique uses the strongest available. Every token is accounted for.

**5. Critique is not optional.**
Every output passes through the Skeptical Reviewer before it reaches the applicant. The reviewer's job is to find reasons to reject. What survives that pass is worth submitting.

**6. Cultural precision over cultural assumption.**
A motivation letter for a German DAAD scholarship and a British Chevening scholarship should read differently — not because of translation, but because of institutional expectations. The Cultural Register Adapter encodes these differences.

---

## Roadmap

- [x] Project architecture and documentation
- [ ] Phase 1: Core agents (Profile Ingestion, Scholarship Analysis, Narrative Selection)
- [ ] Phase 2: Generation agents (Motivation Letter, Academic CV)
- [ ] Phase 3: Validation layer (Evidence Auditor, Cliché Detector, Skeptical Reviewer)
- [ ] Phase 4: Extended generators (Research Proposal, Academic Email)
- [ ] Phase 5: Cultural Register Adapter + Coherence Validator
- [ ] Phase 6: React frontend + API integration
- [ ] Phase 7: Telemetry dashboard + cost analytics
- [ ] Phase 8: Story Bank Manager (cross-application narrative tracking)
- [ ] Phase 9: Application Diff Tool (multi-scholarship consistency)
- [ ] Phase 10: Evaluator Persona Simulation

---

## Motivation

This project exists because I've done this work manually.

Over the past several months, I supported 33+ students with competitive scholarship applications — building research proposals, motivation letters, and academic CVs by orchestrating multiple LLMs (Claude, ChatGPT, Grok) in parallel, comparing outputs, catching hallucinations, and iterating until the result could survive committee scrutiny.

That process taught me two things:
1. The orchestration patterns matter more than the model choice.
2. Most AI writing tools optimize for speed. Committees optimize for substance.

ScholarForge encodes that experience into software.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

This project is in active development. Issues, suggestions, and pull requests are welcome — especially from anyone who has served on scholarship selection committees or has domain expertise in academic admissions.

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/basilfarra">Basel Al-Farra</a> · Khan Younis, Gaza</sub>
</p>
---

<p align="center">
  <sub>Built by <a href="https://github.com/basilfarra">Basel Al-Farra</a> · Khan Younis, Gaza</sub>
</p>
