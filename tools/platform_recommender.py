"""
platform_recommender.py
------------------------
Tool: recommend_platform_and_time

Purpose
    Suggest the best social-media platform and publish time for the
    campaign, based on target audience and goal.

    Platform choice is asked of the LLM first (it can reason about
    audiences the keyword table can't anticipate -- e.g. "software
    companies that use AI agents" -> LinkedIn), with the small heuristic
    table used only as a fallback if Ollama isn't reachable or returns
    something we can't parse. Previously this was heuristic-only and
    defaulted to Facebook for any audience that didn't hit a keyword, which
    is why it looked "hardcoded" -- see README changelog.

    Only one platform is simulated end-to-end for publishing (per the
    assignment's "one platform is sufficient" guidance), but the
    recommendation itself genuinely considers several.

Input
    target_audience: str
    campaign_goal: str

Output
    dict matching state.PlatformRecommendation:
        {platform, publish_datetime, rationale}
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from . import llm_client

CANDIDATE_PLATFORMS = ["facebook", "instagram", "tiktok", "linkedin"]

SYSTEM_PROMPT = (
    "You are a social-media strategist. Given a target audience and campaign "
    f"goal, choose the single best platform from exactly this list: "
    f"{CANDIDATE_PLATFORMS}. Also pick the best hour of day to post (24h, "
    "e.g. 19 for 7pm) for that goal. Then write ONE concise, specific sentence "
    "explaining the choice -- natural prose, not a template. "
    'Return ONLY compact JSON: {"platform": "...", "hour": <int 0-23>, '
    '"rationale": "..."}. No markdown fences, no other text.'
)

# Fallback only -- used if Ollama is unreachable or returns unparsable output.
_AUDIENCE_KEYWORDS = {
    "linkedin": ["b2b", "professional", "business", "enterprise", "corporate",
                 "saas", "software", "startup", "company", "companies", "developer", "engineer"],
    "facebook": ["parent", "family", "35", "40", "50", "adult", "local", "community"],
    "instagram": ["young", "gen z", "18", "24", "fashion", "lifestyle", "beauty", "visual"],
    "tiktok": ["teen", "trend", "short-form", "video"],
}

_GOAL_BEST_HOUR = {
    "awareness": 18,     # 6pm - people scrolling after work
    "engagement": 12,    # lunchtime
    "sales": 19,         # evening, higher purchase-intent browsing
    "traffic": 9,        # morning commute
}


def _fallback_platform(target_audience: str) -> str:
    audience_lower = target_audience.lower()
    scores = {p: 0 for p in CANDIDATE_PLATFORMS}
    for platform, keywords in _AUDIENCE_KEYWORDS.items():
        for kw in keywords:
            if kw in audience_lower:
                scores[platform] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "facebook"


def _fallback_hour(campaign_goal: str) -> int:
    goal_lower = campaign_goal.lower()
    for goal_key, hour in _GOAL_BEST_HOUR.items():
        if goal_key in goal_lower:
            return hour
    return _GOAL_BEST_HOUR["awareness"]


def _next_occurrence(hour: int, minute: int = 0) -> datetime:
    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < now + timedelta(hours=2):
        candidate += timedelta(days=1)
    return candidate


def recommend_platform_and_time(target_audience: str, campaign_goal: str) -> dict:
    raw = llm_client.chat(
        SYSTEM_PROMPT,
        f"Audience: {target_audience}\nGoal: {campaign_goal}",
        temperature=0.4,
    )

    platform = None
    hour = None
    rationale = None

    if raw:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
            candidate_platform = str(parsed.get("platform", "")).lower().strip()
            if candidate_platform in CANDIDATE_PLATFORMS:
                platform = candidate_platform
            candidate_hour = int(parsed.get("hour"))
            if 0 <= candidate_hour <= 23:
                hour = candidate_hour
            rationale = str(parsed.get("rationale", "")).strip() or None
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass  # fall through to heuristic below

    if platform is None:
        platform = _fallback_platform(target_audience)
    if hour is None:
        hour = _fallback_hour(campaign_goal)

    candidate = _next_occurrence(hour)

    if not rationale:
        rationale = (
            f"{platform.capitalize()} best matches this audience, and "
            f"{candidate.strftime('%A %H:%M')} lines up with peak activity "
            f"for a goal of {campaign_goal[0].lower() + campaign_goal[1:] if campaign_goal else campaign_goal}."
        )

    return {
        "platform": platform,
        "publish_datetime": candidate.isoformat(timespec="minutes"),
        "rationale": rationale,
    }
