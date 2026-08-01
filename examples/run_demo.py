"""
run_demo.py
-----------
Headless, reproducible run of the exact scenario described in
examples/example_conversation.md. Useful for grading/demo without needing
to click through the Streamlit UI.

Run from the project root:
    python examples/run_demo.py

Requires nothing external to run (falls back to deterministic templates if
Ollama isn't reachable and to a Lanczos upscale if the SR checkpoints
aren't in checkpoints/) -- but every LLM/SR call logs to stderr whether it
used the real model or a fallback, so you can tell which mode produced this
particular transcript. See the printed "run mode" summary at the end.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph import build_graph, Command  # noqa: E402
from tools import llm_client, super_resolution  # noqa: E402

HERE = os.path.dirname(__file__)


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo"}}

    print("=== Turn 1: manager describes the campaign ===")
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Hi! I need help advertising our new ceramic pour-over coffee maker. "
                        "We're running a launch promo. Product: Ceramic pour-over coffee maker. "
                        "Offer: Buy one get a free bag of beans. Target audience: home coffee "
                        "enthusiasts aged 25-40. Goal: drive sales for the launch week."
                    ),
                }
            ]
        },
        config=config,
    )
    print(result["messages"][-1]["content"])

    print("\n=== Turn 2: manager uploads the product photo ===")
    raw_image = os.path.join(HERE, "demo_product_raw.jpg")
    result = graph.invoke(
        {"raw_image_path": raw_image, "sr_backend": "real_esrgan", "ad_style": "composite"},
        config=config,
    )
    for m in result["messages"][-3:]:
        print(f"[{m['role']}] {m['content']}")
    print("Ad image saved to:", result["ad_image_path"])
    print("Platform recommendation:", result["platform_recommendation"])
    assert result.get("__interrupt__"), "expected to be paused for manager approval"
    caption_before_feedback = result["caption"]

    print("\n=== Turn 3: manager requests changes ===")
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "Make the caption punchier and mention free shipping."}),
        config=config,
    )
    print(result["messages"][-1]["content"])
    print(f"\nCaption before feedback: {caption_before_feedback}")
    print(f"Caption after feedback:  {result['caption']}")
    print(f"Feedback actually changed the caption: {result['caption'] != caption_before_feedback}")
    assert result.get("__interrupt__"), "expected to be paused for approval again"

    print("\n=== Turn 4: manager approves ===")
    result = graph.invoke(Command(resume={"approved": True}), config=config)
    for m in result["messages"][-2:]:
        print(f"[{m['role']}] {m['content']}")

    assert not result.get("__interrupt__"), "workflow should be complete"
    assert result["publish_result"]["simulated"] is True
    assert result["analytics_result"]["simulated"] is True

    out_path = os.path.join(HERE, "demo_transcript.json")
    with open(out_path, "w") as f:
        json.dump(result["messages"], f, indent=2)
    print(f"\nFull transcript written to {out_path}")

    print("\n=== Run mode summary (was this transcript produced by the real models, or fallbacks?) ===")
    print(f"Ollama ({llm_client.MODEL_NAME}) reachable: {llm_client.is_available()}")
    used_backend = result.get("sr_backend", "real_esrgan")
    for backend_name in ("real_esrgan", "mri_model"):
        if backend_name != used_backend:
            print(f"  {backend_name}: not selected this run (only '{used_backend}' was used) -- see compare_sr_backends.py to test both directly")
            continue
        status = super_resolution.get_backend_status(backend_name)
        if status["used_fallback"]:
            print(f"  {backend_name}: FALLBACK -- {status['fallback_reason']}")
        else:
            print(f"  {backend_name}: real inference")


if __name__ == "__main__":
    main()
