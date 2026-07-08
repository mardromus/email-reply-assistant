"""
Synthetic email-response pair generator for the AI Email Response System.

Generates diverse, high-quality email-response pairs across 14 categories,
9 writing styles, and 6 sender types using the Cerebras API (Gemma 4).
Supports batched generation with checkpointing, retry logic, and validation.

Usage:
    python -m dataset.synthetic_generation --count 2000 --batch-size 5 --output dataset/emails.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `config` is importable when run
# as `python -m dataset.synthetic_generation` or directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Customer Support",
    "HR",
    "Internal Team",
    "Sales",
    "Refund",
    "Technical Issue",
    "Scheduling",
    "Product Questions",
    "Complaint",
    "Thank You",
    "Partnership",
    "Billing",
    "Interview",
    "Logistics",
]

STYLES: list[str] = [
    "formal",
    "informal",
    "angry",
    "happy",
    "ambiguous",
    "missing_info",
    "multi_question",
    "short",
    "long",
]

SENDER_TYPES: list[str] = [
    "customer",
    "employee",
    "partner",
    "manager",
    "vendor",
    "candidate",
]

URGENCY_LEVELS: list[str] = ["low", "medium", "high", "critical"]

SENTIMENTS: list[str] = ["positive", "negative", "neutral", "mixed"]

# ---------------------------------------------------------------------------
# Minimum quality thresholds
# ---------------------------------------------------------------------------
MIN_EMAIL_WORDS = 8
MIN_REPLY_WORDS = 10


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _build_prompt(
    category: str,
    style: str,
    sender_type: str,
    batch_size: int,
) -> str:
    """Build the generation prompt for a single batch.

    The prompt instructs the LLM to produce *batch_size* email-response pairs
    as a JSON array, covering the requested category, style, and sender type
    while randomising urgency and sentiment.

    Args:
        category: Email category (e.g. "Customer Support").
        style: Writing style (e.g. "angry", "formal").
        sender_type: Who is sending the email (e.g. "customer").
        batch_size: Number of pairs to generate in this call.

    Returns:
        The formatted prompt string.
    """
    style_guidance = {
        "formal": "Use professional, polished corporate language with proper salutations and sign-offs.",
        "informal": "Use casual, conversational tone. May include contractions and everyday language.",
        "angry": "The sender is frustrated or upset. Use strong language expressing dissatisfaction.",
        "happy": "The sender is pleased or grateful. Use warm, positive language.",
        "ambiguous": "The email is vague or unclear about what the sender actually needs.",
        "missing_info": "The email is missing key details needed to resolve the request (e.g. order number, dates).",
        "multi_question": "The email asks multiple distinct questions or raises several issues at once.",
        "short": "Keep the email very brief — 1-2 sentences only.",
        "long": "Write a detailed, multi-paragraph email with extensive context and background.",
    }

    guidance = style_guidance.get(style, "Write in a natural, professional tone.")

    return f"""You are a synthetic data generator for an AI email response system.

Generate exactly {batch_size} realistic email-response pairs as a JSON array.

REQUIREMENTS:
- Category: {category}
- Sender type: {sender_type} (the person writing the email)
- Style: {style} — {guidance}
- Each pair must have a DIFFERENT scenario / topic within the category.
- Vary the urgency across: {', '.join(URGENCY_LEVELS)}.
- Vary the sentiment across: {', '.join(SENTIMENTS)}.
- The reply should be a helpful, professional human-like response that addresses every point raised.
- Emails and replies must feel authentic — include realistic names, product references, dates, ticket numbers, etc.

OUTPUT FORMAT — return ONLY a valid JSON array (no markdown, no commentary):
[
  {{
    "email": "<the incoming email text>",
    "reply": "<the ideal human-like reply>",
    "urgency": "<low|medium|high|critical>",
    "sentiment": "<positive|negative|neutral|mixed>"
  }},
  ...
]

Generate exactly {batch_size} objects. No extra text outside the JSON array."""


# ---------------------------------------------------------------------------
# Cerebras client helper
# ---------------------------------------------------------------------------

def _create_client() -> OpenAI:
    """Instantiate an OpenAI-compatible client pointed at Cerebras."""
    settings = get_settings()
    if not settings.cerebras_api_key:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set.  Add it to your .env file or "
            "export it as an environment variable."
        )
    return OpenAI(
        base_url=settings.cerebras_base_url,
        api_key=settings.cerebras_api_key,
    )


# ---------------------------------------------------------------------------
# Single-batch generation with retries
# ---------------------------------------------------------------------------

def _generate_batch(
    client: OpenAI,
    category: str,
    style: str,
    sender_type: str,
    batch_size: int,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Call the LLM to generate one batch of email-response pairs.

    Implements exponential back-off on transient API errors and validates
    the returned JSON before accepting it.

    Args:
        client: OpenAI-compatible client.
        category: Email category.
        style: Writing style.
        sender_type: Sender persona.
        batch_size: Number of pairs requested.
        max_retries: Maximum retry attempts on failure.

    Returns:
        A list of validated pair dictionaries.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    settings = get_settings()
    prompt = _build_prompt(category, style, sender_type, batch_size)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise JSON generator.  "
                            "Output ONLY valid JSON arrays — no markdown fences, "
                            "no commentary, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if the model wraps them anyway
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3].rstrip()
            # Handle ```json prefix
            if raw.startswith("json"):
                raw = raw[4:].lstrip()

            pairs: list[dict[str, Any]] = json.loads(raw)

            if not isinstance(pairs, list):
                raise ValueError("LLM output is not a JSON array.")

            validated = _validate_pairs(
                pairs, category, style, sender_type,
            )

            if not validated:
                raise ValueError("No pairs passed validation.")

            logger.debug(
                "Batch OK — category=%s style=%s sender=%s  validated=%d/%d",
                category, style, sender_type, len(validated), len(pairs),
            )
            return validated

        except json.JSONDecodeError as exc:
            logger.warning(
                "JSON parse error (attempt %d/%d): %s", attempt, max_retries, exc,
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Validation error (attempt %d/%d): %s", attempt, max_retries, exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "API error (attempt %d/%d): %s", attempt, max_retries, exc,
            )

        if attempt < max_retries:
            backoff = 2 ** attempt + random.uniform(0, 1)
            logger.info("Retrying in %.1f seconds …", backoff)
            time.sleep(backoff)

    raise RuntimeError(
        f"Failed to generate batch after {max_retries} retries "
        f"(category={category}, style={style}, sender={sender_type})."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_pairs(
    pairs: list[dict[str, Any]],
    category: str,
    style: str,
    sender_type: str,
) -> list[dict[str, Any]]:
    """Validate and enrich raw LLM output into the canonical schema.

    Each pair is checked for required fields, minimum word counts, and
    correct enum values.  Valid pairs are returned with generated IDs and
    metadata; invalid pairs are silently dropped (logged at DEBUG level).

    Args:
        pairs: Raw list of dicts from the LLM.
        category: Expected category.
        style: Expected style.
        sender_type: Expected sender type.

    Returns:
        List of validated, schema-conforming dicts.
    """
    validated: list[dict[str, Any]] = []

    for idx, pair in enumerate(pairs):
        try:
            email_text: str = pair.get("email", "").strip()
            reply_text: str = pair.get("reply", "").strip()

            if not email_text or not reply_text:
                logger.debug("Pair %d: missing email or reply — skipped.", idx)
                continue

            email_wc = len(email_text.split())
            reply_wc = len(reply_text.split())

            if email_wc < MIN_EMAIL_WORDS:
                logger.debug(
                    "Pair %d: email too short (%d words) — skipped.", idx, email_wc,
                )
                continue
            if reply_wc < MIN_REPLY_WORDS:
                logger.debug(
                    "Pair %d: reply too short (%d words) — skipped.", idx, reply_wc,
                )
                continue

            urgency = pair.get("urgency", "medium").lower().strip()
            if urgency not in URGENCY_LEVELS:
                urgency = "medium"

            sentiment = pair.get("sentiment", "neutral").lower().strip()
            if sentiment not in SENTIMENTS:
                sentiment = "neutral"

            validated.append({
                "id": str(uuid.uuid4()),
                "category": category,
                "sender_type": sender_type,
                "email": email_text,
                "reply": reply_text,
                "urgency": urgency,
                "sentiment": sentiment,
                "style": style,
                "word_count_email": email_wc,
                "word_count_reply": reply_wc,
            })

        except Exception as exc:  # noqa: BLE001
            logger.debug("Pair %d: unexpected error — %s", idx, exc)

    return validated


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write accumulated records to a CSV checkpoint.

    Args:
        records: List of validated pair dicts.
        output_path: Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Checkpoint saved — %d records → %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    count: int = 2000,
    batch_size: int = 5,
    output: str = "dataset/emails.csv",
) -> pd.DataFrame:
    """Generate *count* synthetic email-response pairs and save to CSV.

    The generator cycles through all combinations of category, style, and
    sender type to maximise diversity.  Progress is checkpointed every 50
    pairs so work is not lost on interruption.

    Args:
        count: Total number of pairs to generate.
        batch_size: Pairs per LLM call.
        output: Output CSV path (relative to project root or absolute).

    Returns:
        DataFrame containing all generated pairs.
    """
    settings = get_settings()
    output_path = Path(output) if Path(output).is_absolute() else _PROJECT_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _create_client()

    # Pre-compute a shuffled schedule of (category, style, sender_type) combos
    combos: list[tuple[str, str, str]] = [
        (cat, sty, snd)
        for cat in CATEGORIES
        for sty in STYLES
        for snd in SENDER_TYPES
    ]
    random.shuffle(combos)

    total_batches = (count + batch_size - 1) // batch_size
    records: list[dict[str, Any]] = []
    failed_batches: int = 0
    combo_idx: int = 0

    logger.info(
        "Starting generation — target=%d  batch_size=%d  total_batches≈%d",
        count, batch_size, total_batches,
    )

    pbar = tqdm(total=count, desc="Generating emails", unit="pair")

    while len(records) < count:
        cat, sty, snd = combos[combo_idx % len(combos)]
        combo_idx += 1

        remaining = count - len(records)
        current_batch = min(batch_size, remaining)

        try:
            batch = _generate_batch(
                client, cat, sty, snd, current_batch,
            )
            records.extend(batch)
            pbar.update(len(batch))

        except RuntimeError as exc:
            failed_batches += 1
            logger.error("Batch failed permanently: %s", exc)
            # Continue with the next combo instead of crashing
            if failed_batches > total_batches * 0.25:
                logger.critical(
                    "Too many failed batches (%d) — aborting.", failed_batches,
                )
                break

        # Checkpoint every 50 records
        if len(records) % 50 < batch_size and len(records) > 0:
            _checkpoint(records, output_path)

    pbar.close()

    # Final save
    _checkpoint(records, output_path)

    logger.info(
        "Generation complete — %d pairs generated, %d batches failed.",
        len(records), failed_batches,
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit arg list (for testing).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic email-response pairs for the AI Email Response System.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Total number of email-response pairs to generate (default: 2000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of pairs per LLM API call (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/emails.csv",
        help="Output CSV path, relative to project root (default: dataset/emails.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry-point for CLI execution.

    Args:
        argv: Optional explicit arg list (for testing).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args(argv)
    logger.info(
        "Config — count=%d  batch_size=%d  output=%s",
        args.count, args.batch_size, args.output,
    )

    df = generate_dataset(
        count=args.count,
        batch_size=args.batch_size,
        output=args.output,
    )

    # Quick summary
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"Total pairs generated : {len(df)}")
    if not df.empty:
        print(f"Categories            : {df['category'].nunique()}")
        print(f"Styles                : {df['style'].nunique()}")
        print(f"Sender types          : {df['sender_type'].nunique()}")
        print(f"Avg email word count  : {df['word_count_email'].mean():.0f}")
        print(f"Avg reply word count  : {df['word_count_reply'].mean():.0f}")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
