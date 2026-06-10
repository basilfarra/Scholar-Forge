"""
Base agent abstraction for the ScholarForge pipeline.

Every agent in ScholarForge inherits from BaseAgent. This class encodes the
contracts that make the multi-agent pipeline reliable:

    1. Structured input/output via Pydantic models (no free-text passing).
    2. Provider-agnostic LLM invocation (Anthropic, OpenAI, Google).
    3. Automatic telemetry: latency, token counts, cost, errors.
    4. Retry logic with exponential backoff for transient API failures.
    5. Schema validation on both ends of every call.

Agents implement two methods:
    - `_build_prompt(input_data)` — constructs the system + user prompt
    - `_parse_output(raw_response)` — converts model output to structured form

The orchestrator handles routing, error escalation, and pipeline composition.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Type variables for generic input/output contracts
# ──────────────────────────────────────────────────────────────────────

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


# ──────────────────────────────────────────────────────────────────────
# Model tier routing
# ──────────────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """
    Logical tiers used by the Cost Orchestrator to route tasks.

    The mapping from tier to concrete model lives in config, not here —
    so a tier can be re-routed to a different model without changing agents.
    """
    LOW = "low"        # Mechanical tasks: pattern matching, verification
    MID = "mid"        # Extraction, structured analysis, draft generation
    HIGH = "high"      # Judgment-heavy: narrative selection, critique


# ──────────────────────────────────────────────────────────────────────
# Telemetry record produced for every agent invocation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AgentInvocation:
    """
    Telemetry payload emitted after every agent run.

    Stored as a JSON line in logs/telemetry.jsonl for later analysis.
    Used to track cost, latency, model performance, and error patterns
    across the entire pipeline.
    """
    invocation_id: str
    agent_name: str
    model: str
    tier: ModelTier
    started_at: float
    duration_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    success: bool
    error: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "tier": self.tier.value,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "success": self.success,
            "error": self.error,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────────────────────────────

class AgentError(Exception):
    """Base exception for all agent failures."""


class AgentValidationError(AgentError):
    """
    Raised when agent input or output fails schema validation.

    This is a non-retryable error — schema mismatches indicate a bug,
    not a transient failure.
    """


class AgentProviderError(AgentError):
    """
    Raised when the underlying LLM provider returns an error.

    May be retryable depending on the underlying cause (rate limit,
    transient network error, etc.).
    """


# ──────────────────────────────────────────────────────────────────────
# Provider abstraction
# ──────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Provider-agnostic LLM interface.

    Concrete implementations in src/providers/ wrap Anthropic, OpenAI,
    and Google SDKs. The base agent never imports a specific SDK —
    this keeps agents decoupled from provider choice and enables
    multi-model routing via the Cost Orchestrator.
    """

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> "ProviderResponse":
        ...


@dataclass
class ProviderResponse:
    """Normalized response shape across all providers."""
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    raw: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Base agent
# ──────────────────────────────────────────────────────────────────────

class BaseAgent(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for every agent in the ScholarForge pipeline.

    Subclasses must declare:
        - `input_model`:  Pydantic class for input validation
        - `output_model`: Pydantic class for output validation
        - `tier`:         ModelTier (LOW / MID / HIGH)
        - `name`:         Unique agent identifier for telemetry

    And implement:
        - `_build_prompt(input_data)` → tuple[system_prompt, user_prompt]
        - `_parse_output(raw_text)`   → TOutput instance

    Example:

        class ProfileIngestionAgent(BaseAgent[RawNarrative, Profile]):
            name = "profile_ingestion"
            tier = ModelTier.MID
            input_model = RawNarrative
            output_model = Profile

            def _build_prompt(self, input_data):
                return SYSTEM_PROMPT, f"Narrative:\\n{input_data.text}"

            def _parse_output(self, raw_text):
                return Profile.model_validate_json(raw_text)
    """

    # Subclasses MUST override these
    name: str
    tier: ModelTier
    input_model: type[TInput]
    output_model: type[TOutput]

    # Subclasses MAY override these
    max_tokens: int = 4096
    temperature: float = 0.3
    max_retries: int = 3
    retry_base_delay: float = 1.0

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        telemetry_sink: Any | None = None,
    ) -> None:
        """
        Args:
            provider: Concrete LLM provider (Anthropic, OpenAI, Google).
            model: Provider-specific model identifier.
            telemetry_sink: Optional object with `record(invocation)` method.
                Pipeline orchestrator provides one; tests may omit it.
        """
        self._validate_class_attributes()
        self.provider = provider
        self.model = model
        self.telemetry_sink = telemetry_sink
        self._log = logger.bind(agent=self.name, model=model)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    async def run(self, input_data: TInput | dict[str, Any]) -> TOutput:
        """
        Execute the agent against validated input, return validated output.

        Handles:
          - Input validation (raises AgentValidationError on failure)
          - Prompt construction
          - LLM invocation with retry
          - Output parsing and validation
          - Telemetry emission
        """
        invocation_id = str(uuid.uuid4())
        started_at = time.time()
        start_perf = time.perf_counter()

        # 1. Validate input
        validated_input = self._validate_input(input_data)

        # 2. Build prompts
        system_prompt, user_prompt = self._build_prompt(validated_input)

        # 3. Invoke with retry
        response, retry_count = await self._invoke_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 4. Parse and validate output
        try:
            output = self._parse_output(response.text)
            output = self._validate_output(output)
        except ValidationError as e:
            self._emit_telemetry(
                invocation_id=invocation_id,
                started_at=started_at,
                duration_ms=int((time.perf_counter() - start_perf) * 1000),
                response=response,
                success=False,
                error=f"Output validation failed: {e}",
                retry_count=retry_count,
            )
            raise AgentValidationError(
                f"{self.name} produced output that failed validation: {e}"
            ) from e
        except (json.JSONDecodeError, ValueError) as e:
            self._emit_telemetry(
                invocation_id=invocation_id,
                started_at=started_at,
                duration_ms=int((time.perf_counter() - start_perf) * 1000),
                response=response,
                success=False,
                error=f"Output parsing failed: {e}",
                retry_count=retry_count,
            )
            raise AgentValidationError(
                f"{self.name} produced unparseable output: {e}"
            ) from e

        # 5. Emit success telemetry
        self._emit_telemetry(
            invocation_id=invocation_id,
            started_at=started_at,
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
            response=response,
            success=True,
            retry_count=retry_count,
        )

        return output

    # ──────────────────────────────────────────────────────────
    # Abstract methods — subclasses implement
    # ──────────────────────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, input_data: TInput) -> tuple[str, str]:
        """
        Construct (system_prompt, user_prompt) from validated input.

        System prompts should follow the 7-section structure documented
        in docs/PROMPT_ENGINEERING.md (role, task, input contract, output
        contract, hard constraints, quality criteria, failure modes).

        System prompts are typically loaded from src/pipeline/prompts/*.txt
        rather than inlined here — this keeps prompts versionable and
        editable without touching agent code.
        """

    @abstractmethod
    def _parse_output(self, raw_text: str) -> TOutput:
        """
        Convert raw model output to an instance of self.output_model.

        Most agents will use `self.output_model.model_validate_json(raw_text)`
        directly; agents that produce mixed output (JSON + commentary) may
        need to extract the JSON block first.
        """

    # ──────────────────────────────────────────────────────────
    # Internal: validation, retry, telemetry
    # ──────────────────────────────────────────────────────────

    def _validate_class_attributes(self) -> None:
        """
        Verify subclass declared all required class attributes.

        Called at __init__ to fail loud on incomplete subclasses
        rather than producing confusing errors at run time.
        """
        required = ["name", "tier", "input_model", "output_model"]
        missing = [attr for attr in required if not hasattr(self, attr)]
        if missing:
            raise TypeError(
                f"{self.__class__.__name__} is missing required class "
                f"attributes: {missing}"
            )

    def _validate_input(self, input_data: TInput | dict[str, Any]) -> TInput:
        """Validate input against the declared input_model."""
        if isinstance(input_data, self.input_model):
            return input_data
        if isinstance(input_data, dict):
            try:
                return self.input_model.model_validate(input_data)
            except ValidationError as e:
                raise AgentValidationError(
                    f"{self.name} received invalid input: {e}"
                ) from e
        raise AgentValidationError(
            f"{self.name} expected {self.input_model.__name__} or dict, "
            f"got {type(input_data).__name__}"
        )

    def _validate_output(self, output: TOutput) -> TOutput:
        """Verify output is an instance of the declared output_model."""
        if not isinstance(output, self.output_model):
            raise AgentValidationError(
                f"{self.name} _parse_output returned {type(output).__name__}, "
                f"expected {self.output_model.__name__}"
            )
        return output

    async def _invoke_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[ProviderResponse, int]:
        """
        Call the LLM provider with exponential backoff on transient errors.

        Returns (response, retry_count). Raises AgentProviderError after
        max_retries failures.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.provider.complete(
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response, attempt

            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    self._log.warning(
                        "agent.provider.retry",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay_s=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    self._log.error(
                        "agent.provider.exhausted",
                        attempts=attempt + 1,
                        error=str(e),
                    )

        raise AgentProviderError(
            f"{self.name} exhausted retries after {self.max_retries + 1} "
            f"attempts. Last error: {last_error}"
        ) from last_error

    def _emit_telemetry(
        self,
        *,
        invocation_id: str,
        started_at: float,
        duration_ms: int,
        response: ProviderResponse | None,
        success: bool,
        error: str | None = None,
        retry_count: int = 0,
    ) -> None:
        """Record this invocation for cost tracking and debugging."""
        invocation = AgentInvocation(
            invocation_id=invocation_id,
            agent_name=self.name,
            model=self.model,
            tier=self.tier,
            started_at=started_at,
            duration_ms=duration_ms,
            tokens_in=response.tokens_in if response else 0,
            tokens_out=response.tokens_out if response else 0,
            cost_usd=response.cost_usd if response else 0.0,
            success=success,
            error=error,
            retry_count=retry_count,
        )

        self._log.info(
            "agent.complete" if success else "agent.failed",
            **invocation.to_dict(),
        )

        if self.telemetry_sink is not None:
            self.telemetry_sink.record(invocation)
