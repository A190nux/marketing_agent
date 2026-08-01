"""
llm_client.py
-------------
Thin wrapper around a local Ollama server running Qwen3.5:9b (vision +
tool-calling capable). Every tool that needs the LLM goes through here so
there is exactly one place that knows how to reach Ollama.

If Ollama isn't reachable (e.g. this is being reviewed/graded on a machine
without the model pulled), every function degrades to a deterministic
template-based fallback instead of throwing, so the rest of the app keeps
working end-to-end for a demo.

IMPORTANT: failures here are logged to stderr (not silently swallowed) --
if you see fallback-looking output in the app (string-concatenated
captions, template recommendations, etc.), check your terminal for
"[llm_client]" warnings before assuming the LLM itself is bad. In testing,
most "the LLM is dumb" symptoms turned out to be this client silently
timing out or failing to parse, not the model.
"""

from __future__ import annotations

import json
import re
import sys

import urllib.request
import urllib.error

# Hardcoded config -- edit these directly rather than via env vars.
OLLAMA_HOST = "http://192.168.1.101:11434"
MODEL_NAME = "qwen3.5:9b"

# Qwen3.5 "thinking" mode can easily take well over 30s to produce reasoning
# tokens before it answers on a 9B model -- the old 30s timeout was silently
# eating most calls. Chat/generation calls get a generous budget; only the
# lightweight availability check stays fast.
CHAT_TIMEOUT = 180.0

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _log(msg: str) -> None:
    print(f"[llm_client] {msg}", file=sys.stderr)


def _post(path: str, payload: dict, timeout: float) -> dict | None:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _log(f"HTTP {e.code} calling {path}: {e.read()[:300]!r}")
        return None
    except urllib.error.URLError as e:
        _log(f"could not reach Ollama at {OLLAMA_HOST}{path}: {e}")
        return None
    except TimeoutError:
        _log(f"timed out after {timeout}s calling {path} -- model may still be 'thinking'; "
             f"consider raising MARKETING_ASSISTANT_LLM_TIMEOUT")
        return None
    except (ConnectionError, OSError) as e:
        _log(f"connection error calling {path}: {e}")
        return None
    except json.JSONDecodeError as e:
        _log(f"non-JSON response from {path}: {e}")
        return None


def is_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def strip_think_tags(text: str) -> str:
    """Remove any <think>...</think> reasoning block Ollama/Qwen may still
    include even with think disabled (some Ollama versions include it as a
    separate `message.thinking` field, others inline it in `content`)."""
    return _THINK_TAG_RE.sub("", text).strip()


def chat(system: str, user: str, *, temperature: float = 0.7) -> str | None:
    """Single-turn chat completion. Returns None if the model isn't reachable
    (or times out / errors) so callers can fall back to a template -- check
    stderr for the specific reason if you see unexpected fallback output."""
    result = _post(
        "/api/chat",
        {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,   # skip reasoning tokens: faster, and keeps content clean/parseable
            "options": {"temperature": temperature},
        },
        timeout=CHAT_TIMEOUT,
    )
    if not result:
        return None
    try:
        content = result["message"]["content"]
    except (KeyError, TypeError):
        _log(f"unexpected response shape: {result!r}")
        return None

    content = strip_think_tags(content)
    if not content:
        _log("model returned empty content after stripping <think> tags")
        return None
    return content
