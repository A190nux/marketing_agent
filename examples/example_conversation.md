# Example Conversation — Complete Workflow Demo

This transcript was captured from an actual run of the *current* codebase
(`python examples/run_demo.py`) — see `examples/demo_transcript.json` for
the raw messages, and `examples/demo_product_raw.jpg` /
`examples/demo_ad_output.png` for the before/after images. Reproduce it
with:

```bash
python examples/run_demo.py
```

**This particular run was captured in offline/fallback mode** — Ollama
wasn't running and no Real-ESRGAN weights were present in this environment
— specifically so it doubles as a demonstration that the app degrades
gracefully end-to-end rather than breaking. The script prints a "Run mode
summary" at the end telling you plainly which parts used the real
model(s) vs. a fallback; check that section for your own run before
assuming any given output reflects Qwen3.5:9b's actual writing quality.
With Ollama + real weights in place, captions/rationales read
noticeably better than what's shown below, and the SR step does real
image enhancement instead of a Lanczos upscale.

---

**Manager:**
> Hi! I need help advertising our new ceramic pour-over coffee maker. We're
> running a launch promo. Product: Ceramic pour-over coffee maker. Offer:
> Buy one get a free bag of beans. Target audience: home coffee enthusiasts
> aged 25-40. Goal: drive sales for the launch week.

**Assistant** *(`intake` node — brief now complete)*:
> Got it — here's the brief I have:
> - Product: Ceramic pour-over coffee maker
> - Offer: Buy one get a free bag of beans
> - Audience: Home coffee enthusiasts aged 25-40
> - Goal: Drive sales for the launch week
>
> Please upload a product image and I'll get started.

**Manager:** *(uploads `demo_product_raw.jpg`, 320×220px, low resolution)*

**Assistant** *(`image_check` flags it as low-res → `super_res` runs; weights
weren't present in this environment, so the fallback fired — and now says
so explicitly instead of silently degrading)*:
> ⚠️ Couldn't load the 'real_esrgan' model (model failed to load, see
> stderr above) -- used a plain Lanczos upscale instead. Check that the
> weights file exists at `checkpoints/RealESRGAN_x4plus.pth` and required
> packages are installed.

**Assistant** *(`ad_creation` runs — caption + composited ad image)*:
> Here's the draft ad image and caption for your review.

*(UI shows the generated ad image — see `demo_ad_output.png` — and caption:*
*"Buy one get a free bag of beans on Ceramic pour-over coffee maker, made*
*for home coffee enthusiasts aged 25-40! Don't miss this chance to drive*
*sales for the launch week. Shop now while it lasts.*
*#CeramicPour-overCoffeeMa #LimitedOffer #ShopNow" — this is the*
*deterministic fallback template since Ollama wasn't reachable; with the*
*LLM up this reads as real copy, not a field merge.)*

**Assistant** *(`platform_time_recommender` runs)*:
> Recommended platform: Facebook at 2026-08-01T19:00. Facebook best matches
> this audience, and Saturday 19:00 lines up with peak activity for a goal
> of drive sales for the launch week.

**Assistant** *(`human_approval` — graph pauses here via `interrupt()`)*:
> ✅ Manager approval required. The post will NOT be published or scheduled
> until you approve it.

**Manager:** *(clicks "Request changes")*
> [Requested changes] Make the caption punchier and mention free shipping.

**Assistant** *(graph resumes at `human_approval` → routes back to
`ad_creation`, regenerates caption + ad image WITH the feedback this time
— this used to be silently ignored, now it's actually threaded through)*:
> Updated based on your feedback ("Make the caption punchier and mention
> free shipping.") — here's the revised ad image and caption.
>
> Recommended platform: Facebook at 2026-08-01T19:00. ...

**Manager:** *(clicks "Approve & publish")*

**Assistant** *(`publish` node — SIMULATED)*:
> [SIMULATED] Post scheduled on facebook (post_id=sim-304be69f93).

**Assistant** *(`analytics` node — SIMULATED)*:
> [SIMULATED] Performance snapshot — Impressions: 17,895, Engagement rate: 5.65%.
>
> Recommendations:
> 1. CTR is solid — test a second creative variant to see if it can go higher.
> 2. Engagement is healthy — consider boosting spend behind this post while
>    it's performing.
> 3. Re-test the same offer at a different time slot next week to compare
>    performance.

---

## What actually ran under the hood

| Step | Node | Tool(s) called |
|---|---|---|
| 1 | `intake` | `extract_campaign_info` |
| 2 | `image_check` | `check_image_quality` |
| 3 | `super_res` | `super_resolve_image` (backend: `real_esrgan`, `ad_style`: `composite`) |
| 4 | `ad_creation` (×2, before and after feedback) | `generate_ad_caption` (now feedback-aware), `generate_ad_image` |
| 5 | `platform_time_recommender` | `recommend_platform_and_time` (LLM-driven platform choice, heuristic fallback) |
| 6 | `human_approval` | — (interrupt, no tool call) |
| 7 | `publish` | `publish_post` *(simulated)* |
| 8 | `analytics` | `analyze_performance` *(simulated)* |

The graph never called `publish_post` until `approval_status == "approved"`
was written by `human_approval` — structurally enforced by the `interrupt()`
gate in `graph.py`, not just a prompt instruction.

## Checking whether YOUR run used the real models

`run_demo.py` ends with a summary like this:

```
=== Run mode summary (was this transcript produced by the real models, or fallbacks?) ===
Ollama (qwen3.5:9b) reachable: False
  real_esrgan: FALLBACK -- model failed to load, see stderr above
  mri_model: not selected this run (only 'real_esrgan' was used) -- see compare_sr_backends.py to test both directly
```

If you see `FALLBACK` where you expected real inference, the reason string
tells you exactly why (missing weights file, Ollama not running, import
error, etc.) — the same information is shown live in the Streamlit app's
sidebar **Diagnostics** panel. See `examples/compare_sr_backends.py` for a
way to test the two SR backends directly, outside the agent, with no
ambiguity about which one produced which output.
