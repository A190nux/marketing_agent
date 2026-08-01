"""
state.py
--------
Central state schema for the Marketing Assistant LangGraph app.

Everything the graph nodes read/write lives in this single TypedDict so that
LangGraph's checkpointer can persist/resume the whole campaign (including
mid-conversation interrupts for manager approval).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from operator import add


class ImageQuality(TypedDict, total=False):
    width: int
    height: int
    megapixels: float
    blur_score: float          # variance of Laplacian-ish edge response; lower = blurrier
    needs_enhancement: bool


class PlatformRecommendation(TypedDict, total=False):
    platform: str
    publish_datetime: str      # ISO 8601
    rationale: str


class PublishResult(TypedDict, total=False):
    status: Literal["published", "scheduled", "failed"]
    post_id: str
    platform: str
    publish_datetime: str
    simulated: bool


class AnalyticsResult(TypedDict, total=False):
    impressions: int
    reach: int
    likes: int
    comments: int
    shares: int
    ctr: float
    engagement_rate: float
    recommendations: str
    simulated: bool


class CampaignState(TypedDict, total=False):
    # --- chat transcript -----------------------------------------------
    # list of {"role": "user"|"assistant", "content": str}
    messages: Annotated[list[dict], add]

    # --- campaign brief (filled during `intake`) ------------------------
    product: str
    offer: str
    target_audience: str
    campaign_goal: str
    tone: str

    # --- images ----------------------------------------------------------
    raw_image_path: str            # as uploaded by the manager
    image_quality: ImageQuality
    sr_backend: Literal["real_esrgan", "mri_model", "none"]
    sr_used_fallback: bool         # True if the SR backend silently fell back to Lanczos
    enhanced_image_path: str
    ad_style: Literal["composite", "img2img"]
    ad_image_path: str
    caption: str

    # --- recommendation / approval ---------------------------------------
    platform_recommendation: PlatformRecommendation
    approval_status: Literal["pending", "approved", "changes_requested"]
    approval_feedback: str

    # --- publishing / analytics -------------------------------------------
    publish_result: PublishResult
    analytics_result: AnalyticsResult

    # --- control flow -------------------------------------------------------
    next_step: str                 # scratch field routers can use
    missing_fields: list[str]


REQUIRED_BRIEF_FIELDS = ["product", "offer", "target_audience", "campaign_goal"]


def missing_brief_fields(state: CampaignState) -> list[str]:
    """Which required campaign-brief fields are still unset/empty."""
    return [f for f in REQUIRED_BRIEF_FIELDS if not state.get(f)]


def initial_state() -> CampaignState:
    return {
        "messages": [],
        "approval_status": "pending",
        "sr_backend": "real_esrgan",
    }
