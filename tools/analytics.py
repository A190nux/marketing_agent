"""
analytics.py
------------
Tool: analyze_performance   *** SIMULATED ***

Purpose
    Stand in for pulling real post-performance metrics from a platform
    Insights API. Generates plausible-but-fake engagement numbers seeded
    deterministically from the post_id (so re-running the demo on the same
    post gives consistent numbers), then asks the LLM to turn those numbers
    into 2-3 concrete improvement recommendations. Falls back to a templated
    recommendation if the LLM isn't reachable.

Input
    post_id: str

Output
    dict matching state.AnalyticsResult:
        {impressions, reach, likes, comments, shares, ctr,
         engagement_rate, recommendations, simulated}
"""

from __future__ import annotations

import hashlib
import json
import os
import random

from . import llm_client

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
POSTS_FILE = os.path.join(DATA_DIR, "posts.json")


def _lookup_post(post_id: str) -> dict | None:
    if not os.path.exists(POSTS_FILE):
        return None
    with open(POSTS_FILE) as f:
        posts = json.load(f)
    for p in posts:
        if p["post_id"] == post_id:
            return p
    return None


def _seeded_rng(post_id: str) -> random.Random:
    seed = int(hashlib.sha256(post_id.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def analyze_performance(post_id: str) -> dict:
    rng = _seeded_rng(post_id)

    impressions = rng.randint(2000, 20000)
    reach = int(impressions * rng.uniform(0.55, 0.85))
    likes = int(impressions * rng.uniform(0.01, 0.06))
    comments = int(likes * rng.uniform(0.03, 0.12))
    shares = int(likes * rng.uniform(0.02, 0.10))
    clicks = int(impressions * rng.uniform(0.005, 0.035))

    ctr = round(clicks / impressions, 4)
    engagement_rate = round((likes + comments + shares) / max(reach, 1), 4)

    post = _lookup_post(post_id)
    caption = post.get("caption") if post else None
    platform = post.get("platform") if post else "the platform"

    metrics_summary = (
        f"Platform: {platform}\nImpressions: {impressions}\nReach: {reach}\n"
        f"Likes: {likes}\nComments: {comments}\nShares: {shares}\n"
        f"CTR: {ctr:.2%}\nEngagement rate: {engagement_rate:.2%}\n"
        f"Caption: {caption or 'n/a'}"
    )

    recommendations = llm_client.chat(
        "You are a social-media performance analyst. Given these post "
        "metrics, give exactly 3 short, concrete, numbered recommendations "
        "to improve the next post's performance. No preamble.",
        metrics_summary,
        temperature=0.5,
    )
    if not recommendations:
        tips = []
        if ctr < 0.01:
            tips.append("1. CTR is low -- try a stronger call-to-action in the first line of the caption.")
        else:
            tips.append("1. CTR is solid -- test a second creative variant to see if it can go higher.")
        if engagement_rate < 0.03:
            tips.append("2. Engagement is below average -- ask a direct question in the caption to prompt comments.")
        else:
            tips.append("2. Engagement is healthy -- consider boosting spend behind this post while it's performing.")
        tips.append("3. Re-test the same offer at a different time slot next week to compare performance.")
        recommendations = "\n".join(tips)

    return {
        "impressions": impressions,
        "reach": reach,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "ctr": ctr,
        "engagement_rate": engagement_rate,
        "recommendations": recommendations,
        "simulated": True,
    }
