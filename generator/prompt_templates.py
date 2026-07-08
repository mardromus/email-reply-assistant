"""
Prompt engineering module for the AI Email Response System.

Contains all prompt templates (system, RAG, evaluation), plus helper
functions that transform retrieved examples into the message format
expected by the LLM wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ======================================================================
# 1.  SYSTEM PROMPT
# ======================================================================
SYSTEM_PROMPT: str = """\
You are a professional customer-support email assistant.

### Persona
- You are polite, empathetic, and solution-oriented.
- You write clear, concise, and well-structured replies.
- You maintain a warm yet professional tone at all times.

### Rules
1. Answer every question the customer has asked — do not skip any.
2. Never invent facts, product details, or policies.  If you are unsure, \
say so and offer to escalate.
3. Match the formality level of the incoming email while remaining \
professional.
4. Be concise — avoid filler phrases and unnecessary repetition.
5. If provided with historical examples, ground your response in them \
but adapt to the specifics of the current email.

### Response Format
- Start with an appropriate greeting.
- Provide a structured body (use short paragraphs or bullet points for \
clarity).
- End with a professional sign-off and offer further assistance.
"""

# ======================================================================
# 2.  RAG PROMPT TEMPLATE
# ======================================================================
RAG_PROMPT_TEMPLATE: str = """\
Below are historical email exchanges that are similar to the incoming \
email.  Use them as reference to craft an accurate, grounded reply.  \
Adapt the tone and details to the *current* email — do not copy previous \
replies verbatim.

### Similar Historical Examples
{similar_examples}

### Incoming Customer Email
{incoming_email}

{additional_instructions}\
Generate a professional reply to the incoming email above.  Make sure to \
address every question or concern raised by the customer.\
"""

# ======================================================================
# 3.  HELPER — Format retrieved examples
# ======================================================================

def format_retrieved_examples(examples: list[Any]) -> str:
    """Format a list of retrieved examples into numbered prompt context.

    Each element is expected to have at least these attributes (duck-typed):
        - ``incoming_email``  (str)
        - ``response``        (str)
        - ``score``           (float)

    If an element is a plain ``dict`` the same keys are tried.

    Args:
        examples: Retrieved email objects (e.g. ``RetrievedEmail`` instances)
                  or plain dicts.

    Returns:
        A formatted multi-line string ready for prompt injection.
    """
    if not examples:
        return "(No similar examples found.)"

    parts: list[str] = []
    for idx, ex in enumerate(examples, start=1):
        incoming = _get(ex, "incoming_email", "")
        reply = _get(ex, "response", "")
        score = _get(ex, "score", 0.0)

        block = (
            f"--- Example {idx} (relevance: {score:.2f}) ---\n"
            f"Customer Email:\n{incoming}\n\n"
            f"Agent Reply:\n{reply}\n"
        )
        parts.append(block)

    formatted = "\n".join(parts)
    logger.debug(
        "Formatted %d retrieved examples (%d chars)", len(examples), len(formatted)
    )
    return formatted


# ======================================================================
# 4.  HELPER — Build complete generation prompt
# ======================================================================

def build_generation_prompt(
    email: str,
    examples: list[Any],
    category: str | None = None,
) -> list[dict[str, str]]:
    """Build the full list of chat messages for the LLM.

    Args:
        email:     The incoming customer email to reply to.
        examples:  Retrieved similar examples.
        category:  Optional email category for extra context.

    Returns:
        A list of ``{"role": "...", "content": "..."}`` dicts ready for
        the chat-completions API.
    """
    similar_text = format_retrieved_examples(examples)

    additional: str = ""
    if category:
        additional = (
            f"### Category\nThis email has been classified as: **{category}**.\n\n"
        )

    user_content = RAG_PROMPT_TEMPLATE.format(
        similar_examples=similar_text,
        incoming_email=email,
        additional_instructions=additional,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.debug(
        "Built generation prompt — %d messages, user content length=%d",
        len(messages),
        len(user_content),
    )
    return messages


# ======================================================================
# 5.  EVALUATION PROMPT TEMPLATES
# ======================================================================
EVALUATION_PROMPTS: dict[str, str] = {
    # ------------------------------------------------------------------
    "intent_coverage": """\
You are an expert email analyst.  Given the customer email and the \
agent's reply below, identify all distinct customer intents and check \
whether the reply addresses each one.

### Customer Email
{incoming_email}

### Agent Reply
{response}

Return a JSON object with this schema:
```json
{{
  "intents": [
    {{
      "intent": "<short description>",
      "addressed": true | false,
      "evidence": "<quote or explanation>"
    }}
  ],
  "coverage_score": <float 0.0–1.0>,
  "summary": "<one-sentence summary>"
}}
```
""",
    # ------------------------------------------------------------------
    "completeness": """\
You are a quality-assurance reviewer.  Examine the customer email and \
the agent's reply below.  Determine whether every explicit question has \
been answered.

### Customer Email
{incoming_email}

### Agent Reply
{response}

Return a JSON object:
```json
{{
  "questions": [
    {{
      "question": "<extracted question>",
      "answered": true | false,
      "answer_location": "<quote from reply or null>"
    }}
  ],
  "completeness_score": <float 0.0–1.0>,
  "missing_items": ["<unanswered question>", "..."]
}}
```
""",
    # ------------------------------------------------------------------
    "tone_assessment": """\
You are a communication expert.  Rate the tone of the agent's reply to \
the customer email below.

### Customer Email
{incoming_email}

### Agent Reply
{response}

Return a JSON object:
```json
{{
  "formality": <float 0.0–1.0>,
  "professionalism": <float 0.0–1.0>,
  "empathy": <float 0.0–1.0>,
  "overall_tone_score": <float 0.0–1.0>,
  "tone_description": "<brief characterisation>",
  "suggestions": ["<improvement>", "..."]
}}
```
""",
    # ------------------------------------------------------------------
    "hallucination_check": """\
You are a fact-checking assistant.  Given the customer email, the \
retrieved reference examples, and the agent's reply, identify any \
claims in the reply that are NOT supported by the reference material \
or the customer's email.

### Customer Email
{incoming_email}

### Reference Examples
{similar_examples}

### Agent Reply
{response}

Return a JSON object:
```json
{{
  "claims": [
    {{
      "claim": "<statement from reply>",
      "supported": true | false,
      "source": "<reference or 'not found'>"
    }}
  ],
  "hallucination_score": <float 0.0–1.0 where 1.0 means no hallucinations>,
  "flagged_claims": ["<unsupported claim>", "..."]
}}
```
""",
    # ------------------------------------------------------------------
    "factuality_check": """\
You are an accuracy reviewer.  Check the agent's reply for any internal \
contradictions or factual inconsistencies relative to the customer \
email and retrieved examples.

### Customer Email
{incoming_email}

### Reference Examples
{similar_examples}

### Agent Reply
{response}

Return a JSON object:
```json
{{
  "contradictions": [
    {{
      "statement": "<from reply>",
      "conflicts_with": "<source statement>",
      "severity": "low" | "medium" | "high"
    }}
  ],
  "factuality_score": <float 0.0–1.0>,
  "summary": "<brief assessment>"
}}
```
""",
    # ------------------------------------------------------------------
    "quality_assessment": """\
You are a senior email quality assessor.  Provide a holistic quality \
rating for the agent's reply to the customer email below.

### Customer Email
{incoming_email}

### Agent Reply
{response}

Evaluate on these dimensions: relevance, accuracy, clarity, \
helpfulness, and professionalism.

Return a JSON object:
```json
{{
  "relevance": <float 0.0–1.0>,
  "accuracy": <float 0.0–1.0>,
  "clarity": <float 0.0–1.0>,
  "helpfulness": <float 0.0–1.0>,
  "professionalism": <float 0.0–1.0>,
  "overall_quality_score": <float 0.0–1.0>,
  "strengths": ["<strength>", "..."],
  "weaknesses": ["<weakness>", "..."],
  "summary": "<one-paragraph assessment>"
}}
```
""",
}


# ======================================================================
# Private helpers
# ======================================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve *key* from *obj* whether it is a dict or has attributes."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
