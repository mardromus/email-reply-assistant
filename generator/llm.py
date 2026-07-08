"""
Provider-agnostic LLM wrapper for the AI Email Response System.

Wraps the OpenAI-compatible Cerebras API with retry logic, token
estimation, cost tracking, and structured JSON generation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_RETRIES: int = 3
_BACKOFF_SCHEDULE: tuple[float, ...] = (0.5, 1.0, 2.0)

# Rough characters-per-token ratio for estimation (GPT-family average).
# Cerebras / Gemma tokenisers differ, but this is a useful approximation
# when the provider doesn't surface token counts reliably.
_CHARS_PER_TOKEN: float = 4.0


# ---------------------------------------------------------------------------
# Token-usage tracking
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    """Accumulated token-usage statistics for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class CumulativeUsage:
    """Running totals across many calls for cost-awareness."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def record(self, usage: TokenUsage) -> None:
        """Add a single call's token usage to the running totals."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base error for LLM operations."""


class LLMConnectionError(LLMError):
    """Could not reach the LLM provider."""


class LLMRateLimitError(LLMError):
    """Rate-limited by the provider."""


class LLMResponseError(LLMError):
    """The provider returned an unusable response."""


class LLMJsonParsingError(LLMError):
    """The response could not be parsed as JSON."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a character heuristic.

    Args:
        text: Arbitrary string (prompt or completion).

    Returns:
        Estimated token count (always ≥ 1 for non-empty strings).
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate prompt tokens from a list of chat messages."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    # Add a small overhead per message for role tokens & separators.
    overhead = len(messages) * 4
    return max(1, int(total_chars / _CHARS_PER_TOKEN) + overhead)


# ---------------------------------------------------------------------------
# CerebrasLLM
# ---------------------------------------------------------------------------
class CerebrasLLM:
    """OpenAI-compatible client pre-configured for the Cerebras API.

    Provides ``generate`` (plain text) and ``generate_json`` (parsed dict)
    convenience methods on top of the chat-completions endpoint.  Includes
    automatic retries with exponential back-off, token estimation, and
    cumulative usage tracking.

    Args:
        api_key:    Override for the API key (defaults to ``settings.cerebras_api_key``).
        model:      Model identifier (defaults to ``settings.llm_model``).
        temperature: Sampling temperature (defaults to ``settings.llm_temperature``).
        max_tokens: Max completion tokens (defaults to ``settings.llm_max_tokens``).
        max_retries: Number of retry attempts on transient failures.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = _DEFAULT_RETRIES,
    ) -> None:
        settings = get_settings()

        self._api_key: str = api_key or settings.cerebras_api_key
        self.model: str = model or settings.llm_model
        self.temperature: float = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens: int = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.max_retries: int = max_retries

        if not self._api_key:
            raise LLMError(
                "No Cerebras API key provided. Set CEREBRAS_API_KEY in your "
                "environment or .env file."
            )

        self._client: OpenAI = OpenAI(
            api_key=self._api_key,
            base_url=settings.cerebras_base_url,
        )

        self.usage: CumulativeUsage = CumulativeUsage()

        logger.info(
            "CerebrasLLM initialised — model=%s, temperature=%.2f, max_tokens=%d",
            self.model,
            self.temperature,
            self.max_tokens,
        )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat-completion request and return the assistant's text.

        Args:
            messages:    List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Per-call temperature override.
            max_tokens:  Per-call max-tokens override.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMConnectionError: Network / DNS issues.
            LLMRateLimitError:  429 from the provider.
            LLMResponseError:   Unexpected response structure.
            LLMError:           Catch-all after retries are exhausted.
        """
        effective_temp = temperature if temperature is not None else self.temperature
        effective_max = max_tokens if max_tokens is not None else self.max_tokens

        logger.debug(
            "LLM request — model=%s, temp=%.2f, max_tokens=%d, messages=%d",
            self.model,
            effective_temp,
            effective_max,
            len(messages),
        )

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=effective_temp,
                    max_tokens=effective_max,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                # ── Extract text ──────────────────────────────────
                choice = response.choices[0] if response.choices else None
                if choice is None or choice.message is None:
                    raise LLMResponseError("Empty choices in LLM response.")

                text: str = choice.message.content or ""

                # ── Token accounting ──────────────────────────────
                usage = self._extract_usage(response, messages, text)
                self.usage.record(usage)

                logger.debug(
                    "LLM response — %d tokens, %.0f ms, text_len=%d",
                    usage.total_tokens,
                    elapsed_ms,
                    len(text),
                )
                return text

            except RateLimitError as exc:
                last_exception = exc
                logger.warning(
                    "Rate-limited (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except APITimeoutError as exc:
                last_exception = exc
                logger.warning(
                    "Timeout (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except APIConnectionError as exc:
                last_exception = exc
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except APIStatusError as exc:
                # Non-retriable server errors (4xx other than 429, 5xx)
                if exc.status_code and 500 <= exc.status_code < 600:
                    last_exception = exc
                    logger.warning(
                        "Server error %d (attempt %d/%d): %s",
                        exc.status_code,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                else:
                    raise LLMError(
                        f"Non-retriable API error ({exc.status_code}): {exc}"
                    ) from exc

            # Back off before next retry.
            if attempt < self.max_retries:
                delay = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
                logger.info("Retrying in %.1f s …", delay)
                time.sleep(delay)

        # All retries exhausted.
        raise LLMError(
            f"LLM request failed after {self.max_retries} attempts."
        ) from last_exception

    # ------------------------------------------------------------------
    # JSON generation
    # ------------------------------------------------------------------
    def generate_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a response and parse it as JSON.

        The method first attempts to parse the raw response.  If that fails
        it tries to extract a JSON block delimited by `````json ... ````.

        Args:
            messages:    Chat messages.
            temperature: Per-call temperature override.
            max_tokens:  Per-call max-tokens override.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            LLMJsonParsingError: If the response cannot be parsed as JSON.
            LLMError:            Propagated from :meth:`generate`.
        """
        raw = self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Try multiple strategies to extract valid JSON from *raw*."""
        # Strategy 1: direct parse
        try:
            return json.loads(raw)  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract ```json ... ``` fenced block
        if "```json" in raw:
            try:
                block = raw.split("```json", 1)[1].split("```", 1)[0]
                return json.loads(block.strip())  # type: ignore[return-value]
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: extract ``` ... ``` fenced block (no language tag)
        if "```" in raw:
            try:
                block = raw.split("```", 1)[1].split("```", 1)[0]
                return json.loads(block.strip())  # type: ignore[return-value]
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 4: find first { ... } substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])  # type: ignore[return-value]
            except json.JSONDecodeError:
                pass

        raise LLMJsonParsingError(
            f"Could not parse LLM output as JSON. Raw response:\n{raw[:500]}"
        )

    @staticmethod
    def _extract_usage(
        response: Any,
        messages: list[dict[str, str]],
        completion_text: str,
    ) -> TokenUsage:
        """Build a :class:`TokenUsage` from the API response or estimation."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            return TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        # Fallback to estimation.
        prompt_est = _estimate_messages_tokens(messages)
        completion_est = estimate_tokens(completion_text)
        return TokenUsage(
            prompt_tokens=prompt_est,
            completion_tokens=completion_est,
            total_tokens=prompt_est + completion_est,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_cumulative_usage(self) -> dict[str, int]:
        """Return a snapshot of cumulative token usage as a plain dict."""
        return {
            "total_prompt_tokens": self.usage.total_prompt_tokens,
            "total_completion_tokens": self.usage.total_completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "call_count": self.usage.call_count,
        }

    def __repr__(self) -> str:
        return (
            f"CerebrasLLM(model={self.model!r}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
