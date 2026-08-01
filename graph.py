"""
graph.py
--------
The LangGraph state machine for the Marketing Assistant.

    intake (loops on itself until brief is complete)
       -> image_check
            -> super_res (only if enhancement needed)
            -> ad_creation   (caption + ad image)
       -> platform_time_recommender
       -> human_approval   [INTERRUPT -- hard gate, see note below]
            - approved            -> publish -> analytics -> END
            - changes_requested   -> ad_creation (loop back)

Human-in-the-loop / approval gate
    `human_approval` calls LangGraph's `interrupt()`, which pauses execution
    and returns control to the caller (the Streamlit app) with whatever
    payload we pass it. The graph will NOT proceed past this node -- and
    therefore `publish_post` can NEVER be called -- until the app resumes
    the graph with a `Command(resume=...)` carrying the manager's decision.
    This is what satisfies "the assistant must not schedule/publish before
    approval": it isn't just a prompt instruction, it's structurally
    impossible for the graph to reach `publish` any other way.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from state import CampaignState, initial_state, missing_brief_fields
from tools import intake, image_quality, super_resolution, caption_gen, ad_image
from tools import platform_recommender, publish, analytics


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def intake_node(state: CampaignState) -> dict:
    messages = state.get("messages", [])
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    known = {
        k: state[k]
        for k in ("product", "offer", "target_audience", "campaign_goal", "tone")
        if state.get(k)
    }
    was_complete = not missing_brief_fields(state)
    updates = intake.extract_campaign_info(last_user_msg, known) if last_user_msg else {}

    new_state = {**state, **updates}
    missing = missing_brief_fields(new_state)

    result: dict = {**updates, "missing_fields": missing}

    if missing:
        question = intake.missing_field_prompt(missing)
        result["messages"] = [{"role": "assistant", "content": question}]
    elif not was_complete:
        # brief just became complete this turn -- announce it once
        result["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Got it — here's the brief I have:\n"
                    f"- Product: {new_state.get('product')}\n"
                    f"- Offer: {new_state.get('offer')}\n"
                    f"- Audience: {new_state.get('target_audience')}\n"
                    f"- Goal: {new_state.get('campaign_goal')}\n\n"
                    "Please upload a product image and I'll get started."
                ),
            }
        ]
    # else: brief was already complete before this turn (e.g. manager just
    # uploaded an image) -- proceed silently, no need to repeat ourselves.
    return result


def route_after_intake(state: CampaignState) -> str:
    if state.get("missing_fields"):
        return "wait_for_input"          # loop: pause, let manager reply
    if not state.get("raw_image_path"):
        return "wait_for_image"          # brief complete but no image yet
    return "image_check"


def image_check_node(state: CampaignState) -> dict:
    quality = image_quality.check_image_quality(state["raw_image_path"])
    return {"image_quality": quality}


def route_after_image_check(state: CampaignState) -> str:
    return "super_res" if state["image_quality"]["needs_enhancement"] else "ad_creation"


def super_res_node(state: CampaignState) -> dict:
    backend = state.get("sr_backend", "real_esrgan")
    enhanced_path = super_resolution.super_resolve_image(state["raw_image_path"], backend)
    status = super_resolution.get_backend_status(backend)

    if status["used_fallback"]:
        content = (
            f"⚠️ Couldn't load the '{backend}' model ({status['fallback_reason']}) -- "
            f"used a plain Lanczos upscale instead. Check that the weights file exists "
            f"at {status['weights_path']} and required packages are installed."
        )
    else:
        content = f"Enhanced the product photo using the '{backend}' super-resolution model."

    return {
        "enhanced_image_path": enhanced_path,
        "sr_used_fallback": status["used_fallback"],
        "messages": [{"role": "assistant", "content": content}],
    }


def ad_creation_node(state: CampaignState) -> dict:
    image_for_ad = state.get("enhanced_image_path") or state["raw_image_path"]
    feedback = state.get("approval_feedback") or None

    caption = caption_gen.generate_ad_caption(
        product=state["product"],
        offer=state["offer"],
        target_audience=state["target_audience"],
        campaign_goal=state["campaign_goal"],
        tone=state.get("tone", "friendly and energetic"),
        feedback=feedback,
    )
    ad_image_path = ad_image.generate_ad_image(
        enhanced_image_path=image_for_ad,
        offer_text=state["offer"],
        product=state["product"],
        style=state.get("ad_style", "composite"),
    )
    msg = "Here's the draft ad image and caption for your review."
    if feedback:
        msg = f"Updated based on your feedback (\"{feedback}\") — here's the revised ad image and caption."
    return {
        "caption": caption,
        "ad_image_path": ad_image_path,
        "approval_status": "pending",
        "approval_feedback": "",  # consumed -- don't let it leak into a future unrelated regeneration
        "messages": [{"role": "assistant", "content": msg}],
    }


def platform_time_node(state: CampaignState) -> dict:
    rec = platform_recommender.recommend_platform_and_time(
        target_audience=state["target_audience"],
        campaign_goal=state["campaign_goal"],
    )
    return {
        "platform_recommendation": rec,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Recommended platform: {rec['platform'].capitalize()} at "
                    f"{rec['publish_datetime']}. {rec['rationale']}"
                ),
            }
        ],
    }


def human_approval_node(state: CampaignState) -> dict:
    """Hard gate: pauses the graph until the Streamlit app resumes it with
    the manager's decision. See module docstring."""
    decision = interrupt(
        {
            "type": "approval_request",
            "ad_image_path": state.get("ad_image_path"),
            "caption": state.get("caption"),
            "platform_recommendation": state.get("platform_recommendation"),
        }
    )
    # decision: {"approved": bool, "feedback": str}
    if decision.get("approved"):
        return {"approval_status": "approved", "approval_feedback": ""}
    return {
        "approval_status": "changes_requested",
        "approval_feedback": decision.get("feedback", ""),
        "messages": [
            {
                "role": "user",
                "content": f"[Requested changes] {decision.get('feedback', '')}",
            }
        ],
    }


def route_after_approval(state: CampaignState) -> str:
    return "publish" if state["approval_status"] == "approved" else "ad_creation"


def publish_node(state: CampaignState) -> dict:
    rec = state["platform_recommendation"]
    result = publish.publish_post(
        post_content={"caption": state["caption"], "ad_image_path": state["ad_image_path"]},
        platform=rec["platform"],
        publish_datetime=rec["publish_datetime"],
    )
    return {
        "publish_result": result,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"[SIMULATED] Post {result['status']} on {result['platform']} "
                    f"(post_id={result['post_id']})."
                ),
            }
        ],
    }


def analytics_node(state: CampaignState) -> dict:
    result = analytics.analyze_performance(state["publish_result"]["post_id"])
    return {
        "analytics_result": result,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "[SIMULATED] Performance snapshot — "
                    f"Impressions: {result['impressions']}, "
                    f"Engagement rate: {result['engagement_rate']:.2%}.\n"
                    f"Recommendations:\n{result['recommendations']}"
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(CampaignState)

    graph.add_node("intake", intake_node)
    graph.add_node("image_check", image_check_node)
    graph.add_node("super_res", super_res_node)
    graph.add_node("ad_creation", ad_creation_node)
    graph.add_node("platform_time_recommender", platform_time_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("publish", publish_node)
    graph.add_node("analytics", analytics_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "wait_for_input": END,     # pause turn, wait for manager's next chat message
            "wait_for_image": END,     # pause turn, wait for image upload
            "image_check": "image_check",
        },
    )
    graph.add_conditional_edges(
        "image_check",
        route_after_image_check,
        {"super_res": "super_res", "ad_creation": "ad_creation"},
    )
    graph.add_edge("super_res", "ad_creation")
    graph.add_edge("ad_creation", "platform_time_recommender")
    graph.add_edge("platform_time_recommender", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"publish": "publish", "ad_creation": "ad_creation"},
    )
    graph.add_edge("publish", "analytics")
    graph.add_edge("analytics", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "initial_state", "CampaignState", "Command"]
