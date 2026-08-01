"""
intake.py
---------
Tool: extract_campaign_info

Purpose
    Parse the manager's free-form chat messages into the structured
    campaign-brief fields the rest of the graph needs (product, offer,
    target_audience, campaign_goal, tone). Used by the `intake` graph node
    on every turn until all required fields are filled.

Input
    message: str                 -- the manager's latest message
    known_fields: dict            -- fields already collected so far

Output
    dict -- only the NEW/updated fields found in this message (merge into
            state with known_fields.update(result))
"""

from __future__ import annotations

import json
import re

from . import llm_client

FIELDS = ["product", "offer", "target_audience", "campaign_goal", "tone"]

SYSTEM_PROMPT = (
    "You extract structured marketing-campaign fields from a manager's chat "
    "message. Fields: product, offer, target_audience, campaign_goal, tone. "
    "Write each value as a clean, standalone phrase -- NOT a copy-pasted "
    "sentence fragment. Drop filler like 'is', 'to', 'that' from the start "
    "of a value (e.g. if they say 'the offer is free trial for 3 months', "
    "extract offer as 'Free trial for 3 months', not 'Is free trial for "
    "3 months'). Preserve acronyms/capitalization the user used (e.g. 'AI', "
    "'SaaS'). Return ONLY a compact JSON object with keys for the fields you "
    "can confidently fill from THIS message (omit fields not mentioned, "
    "don't guess). No prose, no markdown fences, just the JSON object."
)

# Leading filler words that leak into naive extraction ("offer is buy one..."
# -> captures "is buy one...") -- stripped from ANY extracted value,
# LLM-sourced or regex-fallback-sourced, since even a good LLM occasionally
# echoes the sentence structure back.
_FILLER_PREFIXES = ["is ", "the ", "to ", "that ", "a ", "an ", "we ", "our "]


def _clean_field(text: str) -> str:
    text = text.strip().strip('"\'')
    lowered = text.lower()
    changed = True
    while changed:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):].lstrip()
                lowered = text.lower()
                changed = True
    if not text:
        return text
    # Capitalize only the first character -- do NOT lowercase the rest
    # (the old `.capitalize()` call was turning "AI" into "ai", "SaaS" into
    # "saas", etc.)
    return text[0].upper() + text[1:]


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if the model added them anyway
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        return {
            k: _clean_field(str(v))
            for k, v in parsed.items()
            if k in FIELDS and v and str(v).strip()
        }
    except json.JSONDecodeError:
        return {}


def extract_campaign_info(message: str, known_fields: dict | None = None) -> dict:
    known_fields = known_fields or {}
    context = ""
    if known_fields:
        context = f"Already known: {json.dumps(known_fields)}\n"

    raw = llm_client.chat(SYSTEM_PROMPT, f"{context}Message: {message}", temperature=0.0)
    if raw:
        extracted = _extract_json(raw)
        if extracted:
            return extracted
        # LLM responded but we couldn't parse JSON out of it -- fall through
        # to the regex fallback rather than silently returning nothing.

    # --- lightweight keyword fallback if the LLM is unreachable -----------
    # NOTE: matches against the ORIGINAL-case message (not lowercased) so
    # captured text preserves "AI", "SaaS", etc. -- only the *keyword search*
    # is case-insensitive.
    extracted: dict = {}
    if "product" not in known_fields:
        m = re.search(r"product(?:\sis)?[:\-]?\s*([^.,\n]+)", message, re.IGNORECASE)
        if m:
            extracted["product"] = _clean_field(m.group(1))
    if "offer" not in known_fields:
        m = re.search(r"offer[:\-]?\s*([^.,\n]+)", message, re.IGNORECASE) or re.search(
            r"(\d+%\s*off|buy\s*one\s*get\s*one[^.,\n]*)", message, re.IGNORECASE
        )
        if m:
            extracted["offer"] = _clean_field(m.group(1))
    if "target_audience" not in known_fields:
        m = re.search(r"audience(?:\sis)?[:\-]?\s*([^.,\n]+)", message, re.IGNORECASE)
        if m:
            extracted["target_audience"] = _clean_field(m.group(1))
    if "campaign_goal" not in known_fields:
        m = re.search(r"goal(?:\sis)?[:\-]?\s*([^.,\n]+)", message, re.IGNORECASE)
        if m:
            extracted["campaign_goal"] = _clean_field(m.group(1))
    return extracted


def missing_field_prompt(missing: list[str]) -> str:
    """Friendly follow-up question for whatever's still missing."""
    labels = {
        "product": "what product you're advertising",
        "offer": "what the offer or promotion is",
        "target_audience": "who the target audience is",
        "campaign_goal": "what the campaign goal is (e.g. awareness, sales, traffic)",
    }
    asks = [labels[f] for f in missing if f in labels]
    if not asks:
        return "Could you tell me a bit more about the campaign?"
    if len(asks) == 1:
        return f"Could you tell me {asks[0]}?"
    return "Could you tell me " + ", ".join(asks[:-1]) + f", and {asks[-1]}?"
