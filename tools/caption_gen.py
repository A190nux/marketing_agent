"""
caption_gen.py
--------------
Tool: generate_ad_caption

Purpose
    Write an engaging social-media caption for the ad post. Supports an
    optional `feedback` parameter so a manager's "request changes" is
    actually incorporated into the regenerated caption, not ignored.

Input
    product: str, offer: str, target_audience: str, campaign_goal: str,
    tone: str = "friendly and energetic"
    feedback: str | None = None  -- manager's revision request, if any

Output
    caption: str
"""

from __future__ import annotations

from . import llm_client

SYSTEM_PROMPT = (
    "You are a senior social-media copywriter. Write a short, high-converting "
    "ad caption (2-4 sentences, include 2-4 relevant hashtags at the end). "
    "Write real marketing copy -- punchy, benefit-led, in your own words. "
    "Do NOT restate the brief fields verbatim or produce a sentence that "
    "reads like a template merge (e.g. never write '<offer> on <product>, "
    "made for <audience>'). No markdown, no quotation marks around the "
    "caption, just the caption text."
)


def _lower_first(text: str) -> str:
    """Lowercase only the first character, preserving internal
    capitalization/acronyms (e.g. 'AI agents' stays 'AI agents', not
    'ai agents') -- for embedding a field mid-sentence in the fallback
    template."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def generate_ad_caption(
    product: str,
    offer: str,
    target_audience: str,
    campaign_goal: str,
    tone: str = "friendly and energetic",
    feedback: str | None = None,
) -> str:
    user_prompt = (
        f"Product: {product}\n"
        f"Offer: {offer}\n"
        f"Target audience: {target_audience}\n"
        f"Campaign goal: {campaign_goal}\n"
        f"Tone: {tone}\n"
    )
    if feedback:
        user_prompt += (
            f"\nThe manager reviewed a previous draft and asked for this "
            f"change: \"{feedback}\"\nWrite a NEW caption that addresses "
            f"that feedback.\n"
        )
    user_prompt += "\nWrite the caption now."

    caption = llm_client.chat(SYSTEM_PROMPT, user_prompt, temperature=0.8)
    if caption:
        return caption

    # --- deterministic template fallback (Ollama unreachable) -------------
    hashtag = "#" + "".join(w.capitalize() for w in product.split())[:24]
    base = (
        f"{offer} on {product}, made for {_lower_first(target_audience)}! "
        f"Don't miss this chance to {_lower_first(campaign_goal)}. "
        f"Shop now while it lasts. {hashtag} #LimitedOffer #ShopNow"
    )
    if feedback:
        base += f" (Note: manager requested \"{feedback}\" -- LLM unreachable, showing template caption; please retry once Ollama is up.)"
    return base
