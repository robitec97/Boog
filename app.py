"""Boog Chat – enhanced backend with resilient AI search

Changelog (2025‑08‑06)
----------------------
* **Resilient web search** – `search_web()` now tries a prioritized list of
  endpoints:
  1. `SEARCH_API_ENDPOINT` (if provided in env)
  2. Community DuckDuckGo search proxy (vercel) – kept for backward‑compat.
  3. Official DuckDuckGo Instant‑Answer API (`https://api.duckduckgo.com/`).

  The function stops at the first endpoint that returns at least one result.
* Graceful degrade: if an endpoint returns HTTP 404/500 or empty results, we
  transparently try the next one before giving up.
* Added internal `_parse_ddg_instant_answer()` helper to convert DuckDuckGo’s
  `RelatedTopics` JSON format to our unified schema.

No other changes to public behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependency handling: try to import `requests`; fall back to urllib
# ---------------------------------------------------------------------------
try:
    import requests  # type: ignore

    _HAVE_REQUESTS = True
except ModuleNotFoundError:  # Minimal slug with no extra wheels
    import urllib.request
    import urllib.parse

    _HAVE_REQUESTS = False

# ---------------------------------------------------------------------------
# OpenAI SDK (required for AI mode)
# ---------------------------------------------------------------------------
try:
    import openai  # type: ignore
except ImportError as exc:  # Fail fast if the dependency is missing
    raise RuntimeError(
        "openai python package is not installed –\n"
        "pip install openai>=1.14.0"
    ) from exc

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# ---------- Constants -------------------------------------------------------
BOOG_QUOTES: Dict[str, List[str]] = {
    "wisdom": [
        "Sleep is the answer to most problems.",
        "If it fits, sit. If it doesn’t fit, sit anyway.",
        "The humans rush, the cat observes.",
        "Sometimes doing nothing is doing everything.",
        "Patience is a purr-tue.",
    ],
    "roast": [
        "That's your plan? I've seen mice with better strategy.",
        "You're typing? Cute. Still won't fix your code.",
        "Bold of you to assume anyone cares.",
        "Wow. Even the litter box smells better than that idea.",
        "You again? I was hoping for someone interesting.",
    ],
}

# File system on Heroku is ephemeral; store in /tmp
LOG_FILE = os.path.join("/tmp", "boog_log.json")

# ---------- Helper functions ------------------------------------------------

def log_chat(user_msg: str, boog_msg: str) -> None:
    """Append the chat pair to a local JSON log (best-effort)."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_msg,
        "boog": boog_msg,
    }
    try:
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    except FileNotFoundError:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([entry], f, indent=2)


# ---------------------- Resilient web search layer --------------------------

def _http_get_json(url: str, params: Optional[Dict[str, str]] = None) -> Any:
    """Lightweight helper to GET JSON via requests or urllib."""
    if _HAVE_REQUESTS:
        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # pylint: disable=broad-except
            app.logger.debug("requests GET failed: %s", exc)
            raise  # Reraise so caller can handle
    else:
        import urllib.request
        import urllib.parse

        full_url = url
        if params:
            full_url += "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(full_url, timeout=8) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # pylint: disable=broad-except
            app.logger.debug("urllib GET failed: %s", exc)
            raise


def _parse_ddg_instant_answer(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert DuckDuckGo Instant‑Answer JSON to unified schema."""
    results: List[Dict[str, str]] = []

    # The structure is a bit nested; flatten RelatedTopics
    for topic in payload.get("RelatedTopics", []):
        # Two shapes: either contains "Text" & "FirstURL" or is a category with "Topics"
        if "Text" in topic and "FirstURL" in topic:
            results.append({
                "title": topic["Text"].split(" – ")[0][:80],
                "snippet": topic["Text"],
                "url": topic["FirstURL"],
            })
        elif "Topics" in topic:
            for sub in topic["Topics"]:
                if "Text" in sub and "FirstURL" in sub:
                    results.append({
                        "title": sub["Text"].split(" – ")[0][:80],
                        "snippet": sub["Text"],
                        "url": sub["FirstURL"],
                    })
    return results


def search_web(query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
    """Return search results via a chain of fallback endpoints."""
    endpoints = []

    # 1) User‑specified micro‑service (highest priority)
    if os.getenv("SEARCH_API_ENDPOINT"):
        endpoints.append((os.getenv("SEARCH_API_ENDPOINT"), "generic"))

    # 2) Community proxy (same schema we used before)
    endpoints.append((
        "https://ddg-webapp-search.vercel.app/api/search",
        "community_proxy",
    ))

    # 3) Official DuckDuckGo Instant‑Answer API
    endpoints.append(("https://api.duckduckgo.com/", "instant_answer"))

    for url, kind in endpoints:
        try:
            if kind == "generic" or kind == "community_proxy":
                payload = _http_get_json(url, params={"q": query, "max_results": str(max_results)})
                results = payload.get("results", [])
            elif kind == "instant_answer":
                payload = _http_get_json(url, params={"q": query, "format": "json"})
                results = _parse_ddg_instant_answer(payload)
            else:
                results = []

            if results:  # success!
                return results[:max_results]
        except Exception as exc:  # pylint: disable=broad-except
            app.logger.warning("Search error (%s): %s", url, exc)
            continue  # Try next endpoint

    # All endpoints failed
    return []


# --------------------------- AI generation ----------------------------------

def generate_ai_response(question: str) -> str:
    """Combine web search context with OpenAI chat completion."""
    # 1. Run web search
    results = search_web(question, max_results=8)
    context_lines = [
        f"- {res.get('title')}: {res.get('snippet')} (source: {res.get('url')})"
        for res in results
    ]
    context = "\n".join(context_lines) if context_lines else "(No relevant results found.)"

    # 2. Build system & user messages
    system_prompt = (
        "You are Boog, a sassy but helpful research cat. "
        "Provide concise, accurate answers using the provided web context. "
        "Always cite facts inline as (source). If context is empty, answer politely "
        "that you couldn't find current info."
    )

    user_prompt = (
        f"Web context:\n{context}\n\nUser question: {question}\n\n"
        "Answer like a research agent cat."
    )

    # 3. Call OpenAI ChatCompletion
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai.api_key:
        return (
            "OPENAI_API_KEY is not set on the server – AI mode is temporarily unavailable."
        )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Small, fast model; change if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    # Ensure we have some text to send back
    return answer or "(No response from AI)"

# ---------- Flask Routes ----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input: str = payload.get("message", "").strip()
    mode: str = payload.get("mode", "wisdom").lower()

    if mode == "ai":
        boog_response = generate_ai_response(user_input)
    else:
        # Fallback to canned quotes if mode not recognised
        if mode not in BOOG_QUOTES:
            mode = "roast"
        boog_response = random.choice(BOOG_QUOTES[mode])

    log_chat(user_input, boog_response)
    return jsonify(response=boog_response)


# ---------- Entrypoint ------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Setting threaded=True plays nicer with OpenAI and HTTP requests concurrency
    app.run(host="0.0.0.0", port=port, threaded=True)
