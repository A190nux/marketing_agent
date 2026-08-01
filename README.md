# Agentic Marketing Assistant

An AI Marketing Assistant that helps a marketing manager go from a rough
campaign idea to an approved, scheduled social-media post — understanding
the brief, enhancing the product photo, generating an ad image + caption,
recommending platform/time, gating on manager approval, "publishing", and
reporting performance.

## Framework used

**LangGraph.** The workflow is modeled as an explicit state machine
(`graph.py`) with a `MemorySaver` checkpointer and a hard `interrupt()` gate
before publishing — this is what LangGraph is built for (stateful,
multi-tool, human-in-the-loop agents), as opposed to e.g. DSPy (prompt/
pipeline optimization) which isn't the right tool for this kind of
orchestration.

**Model:** Qwen3.5:9b served locally via [Ollama](https://ollama.com)
(vision + tool-calling capable, chosen to fit on an 8GB GPU). All LLM calls
go through `tools/llm_client.py`, which talks to `http://localhost:11434`.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but recommended) pull and run the LLM
ollama pull qwen3.5:9b
ollama serve            # if not already running

# 3. Launch the app
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

**The app also runs without Ollama.** Every LLM-backed tool has a
deterministic template fallback (see each tool's docstring), so you can
demo/grade the full workflow — intake, image enhancement, ad creation,
approval gate, publish, analytics — with zero external services running.
Quality of captions/rationales is noticeably better with the real model.

### Headless demo (no UI)

```bash
python examples/run_demo.py
```

Reproduces the exact scenario in `examples/example_conversation.md` and
writes the transcript to `examples/demo_transcript.json`. Ends with a "run
mode summary" telling you whether Ollama and each SR backend were actually
reached or fell back — useful for confirming your local setup before
demoing live.

### Comparing the two SR backends directly

```bash
python examples/compare_sr_backends.py path/to/product_photo.jpg
```

Runs both backends outside the agent and fails loudly (rather than
silently falling back) if either can't load its real weights — useful for
judging the MRI model's domain-mismatch behavior on an actual product
photo without any ambiguity about which output came from which model. See
`--help` for the optional `--hr` ground-truth-comparison flag.

## Interface

The Streamlit app (left pane: chat + upload, right pane: workflow output)
covers every interaction point from the assignment brief:

- **Chat** with the assistant to describe the campaign; it asks for
  whatever's still missing.
- **Upload** a product photo via the file uploader.
- **Sidebar controls:** SR backend picker (`real_esrgan` / `mri_model`,
  with a note on what the resolution behavior actually means), ad-creative
  style picker (`composite` / `img2img`), and a **Diagnostics** panel
  showing live whether Ollama and each SR backend are reachable or
  silently falling back.
- **Before/after** super-resolution comparison, shown before the ad image
  is generated.
- **Generated ad image + caption** preview.
- **Recommended platform + time**, with a genuinely LLM-reasoned rationale
  (heuristic fallback only if Ollama is unreachable).
- **Approve / Request changes** — the human-approval gate. Chat is disabled
  while a post is pending approval (sending a message there would otherwise
  re-run the whole pipeline from intake instead of being treated as
  feedback — use the Request changes box for that).
- **Publishing status + performance analysis**, both clearly marked
  simulated.

## Project layout

```
marketing_assistant/
├── app.py                     # Streamlit UI
├── graph.py                   # LangGraph state machine (the agent)
├── state.py                   # shared CampaignState schema
├── tools/
│   ├── llm_client.py          # Ollama/Qwen3.5:9b wrapper (+ offline fallback)
│   ├── intake.py              # extract_campaign_info
│   ├── image_quality.py       # check_image_quality
│   ├── super_resolution.py    # super_resolve_image  ⭐ required SR tool
│   ├── mri_sr_model.py        # RRDBNet arch. for your trained MRI SR model
│   ├── ad_image.py            # generate_ad_image
│   ├── caption_gen.py         # generate_ad_caption
│   ├── platform_recommender.py# recommend_platform_and_time
│   ├── publish.py             # publish_post          (SIMULATED)
│   └── analytics.py           # analyze_performance    (SIMULATED)
├── checkpoints/                # drop best_sr_model.pt here (see checkpoints/README.md)
├── diagrams/workflow.md       # Mermaid workflow diagram
├── examples/
│   ├── run_demo.py            # reproducible headless demo
│   ├── compare_sr_backends.py # standalone real_esrgan vs mri_model comparison
│   ├── example_conversation.md
│   ├── demo_product_raw.jpg   # sample low-res input photo
│   └── demo_ad_output.png     # sample generated ad image
├── static/{uploads,generated}/  # runtime image storage
└── data/posts.json            # simulated "published" posts
```

## Tools

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `extract_campaign_info` | Parse the manager's chat into structured brief fields | `message: str`, `known_fields: dict` | new/updated fields (`product`, `offer`, `target_audience`, `campaign_goal`, `tone`) |
| `check_image_quality` | Decide if the uploaded photo needs enhancing | `image_path: str` | `{width, height, megapixels, blur_score, needs_enhancement}` |
| **`super_resolve_image`** ⭐ | **Required SR tool.** Enhance a low-quality product photo | `image_path: str`, `backend: "real_esrgan" \| "mri_model"` | `enhanced_image_path: str` |
| `generate_ad_caption` | Write the ad's social copy | `product, offer, target_audience, campaign_goal, tone` | `caption: str` |
| `generate_ad_image` | Compose the enhanced photo into a finished ad creative | `enhanced_image_path, offer_text, product` | `ad_image_path: str` |
| `recommend_platform_and_time` | Suggest platform + best publish time | `target_audience, campaign_goal` | `{platform, publish_datetime, rationale}` |
| `publish_post` *(simulated)* | Publish/schedule the approved post | `post_content, platform, publish_datetime` | `{status, post_id, platform, publish_datetime, simulated}` |
| `analyze_performance` *(simulated)* | Report post performance + recommendations | `post_id` | `{impressions, reach, likes, comments, shares, ctr, engagement_rate, recommendations, simulated}` |

## The required image super-resolution tool

`super_resolve_image` (`tools/super_resolution.py`) exposes one `SRModel`
interface with **two swappable backends**, selectable from the sidebar:

- **`real_esrgan`** — general-purpose photo super-resolution. Recommended
  default for product photography. Lazy-imports `realesrgan`/`basicsr`;
  falls back to Lanczos upscaling if weights/libraries aren't present.
- **`mri_model`** — the team's own trained model: an RRDBNet (`tools/mri_sr_model.py`,
  architecture copied verbatim from the training notebook so the checkpoint
  loads exactly as trained), originally built for single-channel T1
  brain-MRI slices (64×64 → 256×256, scale=4). Reused here on RGB product
  photos via a documented domain adapter: convert to YCbCr, run the model
  on the Y (luminance) channel only, upsample Cb/Cr with bicubic, merge
  back to RGB. **This backend runs real inference** against
  `checkpoints/best_sr_model.pt` when present (see `checkpoints/README.md`
  for the expected checkpoint format) — it is not a stand-in. Since the
  architecture is fully convolutional, it runs on any input size at
  inference despite being trained on fixed 64×64 crops. **Not recommended**
  for product photography due to domain mismatch (trained on medical
  imagery; may over-smooth or introduce MRI-like texture on real photos) —
  included for comparison, per the team's earlier scheduling mix-up (see
  "Known limitations").

Weights paths are hardcoded (not env vars, per earlier preference) as
class-level constants — edit them directly if you move the files:

| Backend | Constant | File | Default |
|---|---|---|---|
| `real_esrgan` | `RealESRGANBackend.WEIGHTS_PATH` | `tools/super_resolution.py` | `checkpoints/RealESRGAN_x4plus.pth` |
| `mri_model` | `MRIModelBackend.WEIGHTS_PATH` / `.DEVICE` | `tools/super_resolution.py` | `checkpoints/best_sr_model.pt` / `cuda` (auto-falls back to `cpu`) |

Both backends fall back to a deterministic Lanczos upscale if their real
dependencies/weights aren't available — so the tool interface, routing
logic, and UI are all fully exercised even without GPU weights present, but
with `checkpoints/best_sr_model.pt` in place, `mri_model` does the real
thing. Every fallback now also prints its exact reason to stderr and shows
up in the Streamlit sidebar's **Diagnostics** panel — if `real_esrgan` and
`mri_model` ever look identical in the app, check there first; it almost
always means both silently fell back to the same Lanczos path.

## Human approval gate

`human_approval` (`graph.py`) calls LangGraph's `interrupt()`, which pauses
the graph entirely and hands control back to the Streamlit app. **The graph
has no other path to the `publish` node** — it can only resume via
`Command(resume={"approved": True})` sent from the "Approve & publish"
button. "Request changes" resumes with `{"approved": False, "feedback": ...}`
and routes back to `ad_creation`, regenerating the caption/ad image before
presenting for approval again. This is a structural guarantee, not just a
prompt instruction.

## Workflow diagram

See [`diagrams/workflow.md`](diagrams/workflow.md) (Mermaid, generated
directly from the compiled graph so it can't drift from the code).

## Example conversation

See [`examples/example_conversation.md`](examples/example_conversation.md)
for a full transcript (intake → image upload → super-resolution → ad
creation → platform/time recommendation → approval gate → change request →
re-approval → simulated publish → simulated analytics), reproducible via
`python examples/run_demo.py`.

## Simulated components — clearly identified

Per the assignment ("this part can be simulated if needed", "one platform
is sufficient"), the following are simulated rather than hitting a real
external API:

- **`publish_post`** — writes a local JSON record (`data/posts.json`)
  instead of calling the Facebook Graph API. Response shape mirrors what a
  real integration would return (`status`, `post_id`, etc.) and is tagged
  `"simulated": true`.
- **`analyze_performance`** — generates plausible, seeded-random engagement
  metrics instead of pulling from a platform Insights API. Also tagged
  `"simulated": true`. Recommendations are genuinely LLM-generated (or
  template-generated offline) from those simulated metrics.
- **`mri_model` SR backend** — **not simulated**: it runs real inference
  with your actual trained RRDBNet weights (`checkpoints/best_sr_model.pt`).
  What's disclosed here is a genuine *domain* mismatch, not a fake tool: the
  model was trained for T1 brain MRI slices, not product photography. Kept
  as a secondary, clearly-labeled option (see "The required image
  super-resolution tool" above) rather than hidden or presented as
  equivalent to `real_esrgan`.
- **Ad-image generation** defaults to deterministic Pillow compositing
  (photo + banner + offer text), not text/image-to-image diffusion — chosen
  deliberately for reliability under the project deadline. An optional SD1.5
  img2img "stylize" pass exists in `tools/ad_image.py` (`style="img2img"`)
  and is used opportunistically if `torch` + `diffusers` + a CUDA GPU are
  available, otherwise silently skipped.
- **Every LLM-backed tool** (`extract_campaign_info`, `generate_ad_caption`,
  `recommend_platform_and_time`'s rationale, `analyze_performance`'s
  recommendations) falls back to a deterministic, non-LLM template if
  Ollama isn't reachable, so the app degrades gracefully rather than
  breaking a live demo.

## Known limitations

- The `intake` keyword-regex fallback (used only when Ollama is
  unreachable) is noticeably less robust than the LLM-based extraction —
  e.g. it can only pull one field per matched keyword and doesn't handle
  free-form phrasing well. This is expected and documented; it exists purely
  so the app doesn't hard-fail without a running LLM.
- `MemorySaver` checkpointing is in-process/in-memory only — campaign state
  does not persist across an app restart. Swapping in `SqliteSaver` or
  `PostgresSaver` (both drop-in LangGraph checkpointers) would add
  persistence with no changes to `graph.py`'s node logic.
- `.nii` (NIfTI) file support for the MRI backend's native input format is a
  data-loading detail, not core architecture, and was deliberately
  deprioritized for this deadline — the backend currently accepts standard
  RGB images.
- Re-invoking the graph with a plain message (not `Command(resume=...)`)
  re-enters from `START` — so a chat message sent while a post is pending
  approval would silently discard that interrupt and re-run the whole
  pipeline from `intake`. The approval gate itself still holds (there's no
  path from that re-run to `publish` without a fresh approval), but it's
  wasteful and confusing. The UI works around this by disabling chat input
  while approval is pending; a future revision could instead treat such a
  message as implicit feedback in `human_approval`'s resume payload.
