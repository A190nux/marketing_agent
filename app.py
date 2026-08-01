"""
app.py
------
Streamlit interface for the Marketing Assistant.

Run with:  streamlit run app.py

Provides:
    - chat with the assistant (intake conversation)
    - product image upload
    - SR backend picker (real_esrgan / mri_model) + before/after comparison
    - ad creative style picker (composite / img2img)
    - generated ad image + caption preview
    - platform/time recommendation display
    - Approve / Request changes buttons  (the human-approval gate)
    - publishing status + performance analysis (both clearly marked SIMULATED)
    - a diagnostics panel showing whether Ollama / SR backends are actually
      reachable, so a silent fallback doesn't look like "the model is dumb"
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from PIL import Image

from graph import build_graph, Command
from state import initial_state
from tools import llm_client, super_resolution

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="AI Marketing Assistant", page_icon="📣", layout="wide")


# ---------------------------------------------------------------------------
# Session / graph bootstrap
# ---------------------------------------------------------------------------

@st.cache_resource
def get_graph():
    return build_graph()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "latest_state" not in st.session_state:
    st.session_state.latest_state = {}
if "image_uploaded_name" not in st.session_state:
    st.session_state.image_uploaded_name = None

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


def run_graph(payload: dict | Command):
    """Invoke the graph and refresh local state.

    Note: `messages` uses LangGraph's `add` reducer, so `result["messages"]`
    returned by `invoke()` is already the FULL accumulated conversation for
    this thread (not just the messages produced by this call) -- we simply
    replace our local view with it rather than appending, to avoid
    duplicating history on every turn.
    """
    result = graph.invoke(payload, config=config)
    st.session_state.latest_state = result
    interrupts = result.get("__interrupt__")
    st.session_state.pending_interrupt = interrupts[0].value if interrupts else None


# ---------------------------------------------------------------------------
# Sidebar: campaign status, model pickers, diagnostics
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📣 Marketing Assistant")
    st.caption("LangGraph agent · Qwen3.5:9b via Ollama · human-in-the-loop approval gate")

    st.subheader("Campaign brief")
    state = st.session_state.latest_state
    for label, key in [
        ("Product", "product"),
        ("Offer", "offer"),
        ("Audience", "target_audience"),
        ("Goal", "campaign_goal"),
    ]:
        val = state.get(key)
        st.write(f"**{label}:** {val if val else '_not yet provided_'}")

    st.divider()
    st.subheader("Super-resolution backend")
    sr_backend = st.radio(
        "Model to use when the product photo needs enhancing:",
        options=["real_esrgan", "mri_model"],
        format_func=lambda x: {
            "real_esrgan": "Real-ESRGAN (general photos, recommended)",
            "mri_model": "MRI SR model (Y-channel adapter — domain mismatch, for comparison)",
        }[x],
        key="sr_backend_choice",
    )
    st.caption(
        "Both models always upscale exactly 4x, whatever the input size. The MRI "
        "model was only *trained* on 64x64→256x256 crops though, so quality drops "
        "off the larger/more detailed the input gets relative to that -- it's not "
        "'useless' above 256px, just increasingly out-of-domain. Real-ESRGAN was "
        "trained on full-size natural photos and is the one built to handle that."
    )

    st.subheader("Ad creative style")
    ad_style = st.radio(
        "How to generate the ad image:",
        options=["composite", "img2img"],
        format_func=lambda x: {
            "composite": "Photo + banner overlay (default, always available)",
            "img2img": "AI-stylized (SD1.5 img2img — needs GPU + diffusers)",
        }[x],
        key="ad_style_choice",
    )

    st.divider()
    with st.expander("🩺 Diagnostics", expanded=False):
        ollama_ok = llm_client.is_available()
        st.write(f"**Ollama ({llm_client.MODEL_NAME}):** {'🟢 reachable' if ollama_ok else '🔴 unreachable — using template fallbacks'}")
        if not ollama_ok:
            st.caption(f"Checked {llm_client.OLLAMA_HOST}. Run `ollama serve` and `ollama pull {llm_client.MODEL_NAME}`.")
        for backend_name in ("real_esrgan", "mri_model"):
            status = super_resolution.get_backend_status(backend_name)
            if status["used_fallback"] is False and status["fallback_reason"] is None:
                # hasn't been run yet this session
                st.write(f"**{backend_name}:** ⚪ not used yet")
            elif status["used_fallback"]:
                st.write(f"**{backend_name}:** 🔴 fell back to Lanczos — {status['fallback_reason']}")
            else:
                st.write(f"**{backend_name}:** 🟢 real inference ran")
        st.caption("Full error details also print to the terminal running `streamlit run app.py`.")

    st.divider()
    if st.button("🔄 Start a new campaign"):
        st.session_state.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout: chat (left) / workflow output (right)
# ---------------------------------------------------------------------------

col_chat, col_workflow = st.columns([1, 1.3])

with col_chat:
    st.subheader("Chat with your Marketing Assistant")

    for m in st.session_state.latest_state.get("messages", []):
        role = "assistant" if m["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.write(m["content"])

    uploaded = st.file_uploader("Upload product photo", type=["png", "jpg", "jpeg", "webp"])
    if uploaded and uploaded.name != st.session_state.image_uploaded_name:
        save_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.image_uploaded_name = uploaded.name
        with st.spinner("Processing image..."):
            run_graph({
                "raw_image_path": save_path,
                "sr_backend": sr_backend,
                "ad_style": ad_style,
            })
        st.rerun()

    user_msg = st.chat_input(
        "Describe the product, offer, audience, goal...",
        disabled=bool(st.session_state.pending_interrupt),
    )
    if st.session_state.pending_interrupt:
        st.caption(
            "⏸️ A post is waiting for your approval below — use Approve / "
            "Request changes there. (Chatting here while a post is pending "
            "would otherwise silently restart the whole pipeline from "
            "scratch, since re-invoking the graph re-enters from intake.)"
        )
    if user_msg:
        with st.spinner("Thinking..."):
            run_graph({"messages": [{"role": "user", "content": user_msg}]})
        st.rerun()


with col_workflow:
    st.subheader("Campaign workflow")
    state = st.session_state.latest_state

    quality = state.get("image_quality")
    if quality:
        with st.expander("📷 Image quality check", expanded=False):
            st.json(quality)

    # --- before / after SR comparison ------------------------------------
    raw_path = state.get("raw_image_path")
    enhanced = state.get("enhanced_image_path")
    if raw_path and os.path.exists(raw_path):
        st.markdown("### Super-resolution: before / after")
        if state.get("sr_used_fallback"):
            st.warning(
                "The selected SR model didn't load — this is a Lanczos upscale, "
                "not the real model. Check the Diagnostics panel in the sidebar."
            )
        if enhanced and os.path.exists(enhanced):
            before_col, after_col = st.columns(2)
            with before_col:
                st.image(Image.open(raw_path), caption="Before (uploaded)", use_container_width=True)
            with after_col:
                st.image(Image.open(enhanced), caption=f"After ({state.get('sr_backend', 'unknown')})", use_container_width=True)
        else:
            st.image(Image.open(raw_path), caption="Uploaded photo (no enhancement needed)", width=320)

    ad_image_path = state.get("ad_image_path")
    caption = state.get("caption")
    if ad_image_path and os.path.exists(ad_image_path):
        st.markdown("### Draft advertising post")
        st.image(Image.open(ad_image_path), caption="Generated ad image", width=420)
        st.text_area("Generated caption", value=caption or "", height=100, disabled=True)

    rec = state.get("platform_recommendation")
    if rec:
        st.markdown("### Recommended platform & time")
        st.write(f"**Platform:** {rec['platform'].capitalize()}")
        st.write(f"**Publish time:** {rec['publish_datetime']}")
        st.caption(rec["rationale"])

    # --- human approval gate --------------------------------------------
    pending = st.session_state.pending_interrupt
    if pending and pending.get("type") == "approval_request":
        st.markdown("### ✅ Manager approval required")
        st.info("The post will NOT be published or scheduled until you approve it here.")
        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button("✅ Approve & publish", type="primary", use_container_width=True):
                with st.spinner("Publishing..."):
                    run_graph(Command(resume={"approved": True}))
                st.rerun()
        with reject_col:
            # st.form guarantees the text_area value and the submit click are
            # captured together in a single, atomic rerun -- the previous
            # popover+button combo could require an extra interaction to
            # "commit" the click depending on Streamlit version.
            with st.form("feedback_form", clear_on_submit=True):
                feedback = st.text_area("What should change?", key="feedback_box")
                submitted = st.form_submit_button("✏️ Request changes", use_container_width=True)
                if submitted:
                    if not feedback.strip():
                        st.error("Please describe what should change before submitting.")
                    else:
                        with st.spinner("Revising based on your feedback..."):
                            run_graph(Command(resume={"approved": False, "feedback": feedback}))
                        st.rerun()

    publish_result = state.get("publish_result")
    if publish_result:
        st.markdown("### 📤 Publishing status")
        badge = "🟢" if publish_result["status"] == "published" else "🟡"
        st.write(
            f"{badge} **{publish_result['status'].upper()}** on "
            f"{publish_result['platform'].capitalize()} — "
            f"post id `{publish_result['post_id']}`"
        )
        if publish_result.get("simulated"):
            st.caption("⚠️ Simulated publish — no real social-media API call was made.")

    analytics_result = state.get("analytics_result")
    if analytics_result:
        st.markdown("### 📊 Performance analysis")
        if analytics_result.get("simulated"):
            st.caption("⚠️ Simulated metrics, randomly generated per post — not hardcoded, but not real either.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Impressions", f"{analytics_result['impressions']:,}")
        metric_cols[1].metric("Reach", f"{analytics_result['reach']:,}")
        metric_cols[2].metric("CTR", f"{analytics_result['ctr']:.2%}")
        metric_cols[3].metric("Engagement", f"{analytics_result['engagement_rate']:.2%}")
        st.text_area(
            "Recommendations for next time",
            value=analytics_result["recommendations"],
            height=120,
            disabled=True,
        )

    if not state:
        st.info("Start chatting on the left to describe your campaign.")
