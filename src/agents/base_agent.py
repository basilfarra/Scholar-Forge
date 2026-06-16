"""
BaseAgent — Abstract foundation for all ScholarForge agents.

Every agent in the pipeline inherits from this class.
Provides: structured logging, retry logic with exponential backoff,
telemetry timing, and provider-agnostic interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional


def _setup_logger(agent_name: str) -> logging.Logger:
    logger = logging.getLogger(f"scholarforge.{agent_name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class BaseAgent(ABC):
    """
    Abstract base class for all ScholarForge pipeline agents.

    Subclasses must implement `run()` which contains the core logic.
    Call `execute()` externally — it wraps `run()` with telemetry and retry.
    """

    def __init__(
        self,
        agent_name: str,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.agent_name  = agent_name
        self.model       = model
        self.api_key     = api_key
        self.max_tokens  = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger      = _setup_logger(agent_name)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Core agent logic. Implemented by each subclass."""
        ...

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Public entry point. Wraps `run()` with telemetry and retry logic.
        Callers should always use `execute()`, never `run()` directly.
        """
        start_ms = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.run(*args, **kwargs)
                elapsed = (time.monotonic() - start_ms) * 1000
                self.logger.info(
                    "Agent completed successfully",
                    extra={
                        "agent": self.agent_name,
                        "attempt": attempt,
                        "elapsed_ms": round(elapsed, 1),
                    }
                )
                return result

            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {exc}"
                )
                if attempt < self.max_retries:
                    sleep_for = self.retry_delay * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying in {sleep_for:.1f}s...")
                    time.sleep(sleep_for)

        elapsed = (time.monotonic() - start_ms) * 1000
        self.logger.error(
            f"Agent failed after {self.max_retries} attempts "
            f"({elapsed:.0f}ms total)"
        )
        raise last_error
