"""
publish.py
----------
Tool: publish_post   *** SIMULATED ***

Purpose
    Stand in for an actual social-media publish/schedule API call (e.g. the
    Facebook Graph API). Clearly labeled as simulated per the assignment's
    instructions ("this part can be simulated if needed" / "one platform is
    sufficient"). Writes a persisted JSON record so the rest of the app
    (analytics, UI history) has something real to read back.

    This tool is only ever reachable AFTER the human_approval graph node has
    recorded approval_status == "approved" -- enforced in graph.py, not here,
    but documented here too since it's a hard requirement of the assignment.

Input
    post_content: dict {caption, ad_image_path}
    platform: str
    publish_datetime: str (ISO 8601) -- if in the future, status="scheduled",
                                         else status="published"

Output
    dict matching state.PublishResult:
        {status, post_id, platform, publish_datetime, simulated}
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
POSTS_FILE = os.path.join(DATA_DIR, "posts.json")


def _load_posts() -> list[dict]:
    if not os.path.exists(POSTS_FILE):
        return []
    with open(POSTS_FILE) as f:
        return json.load(f)


def _save_posts(posts: list[dict]) -> None:
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)


def publish_post(post_content: dict, platform: str, publish_datetime: str) -> dict:
    now = datetime.now()
    try:
        target = datetime.fromisoformat(publish_datetime)
    except ValueError:
        target = now

    status = "scheduled" if target > now else "published"
    post_id = f"sim-{uuid.uuid4().hex[:10]}"

    record = {
        "post_id": post_id,
        "platform": platform,
        "publish_datetime": publish_datetime,
        "status": status,
        "caption": post_content.get("caption"),
        "ad_image_path": post_content.get("ad_image_path"),
        "created_at": now.isoformat(timespec="seconds"),
        "simulated": True,
    }

    posts = _load_posts()
    posts.append(record)
    _save_posts(posts)

    return {
        "status": status,
        "post_id": post_id,
        "platform": platform,
        "publish_datetime": publish_datetime,
        "simulated": True,
    }
