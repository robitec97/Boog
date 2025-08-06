from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any

import requests
from flask import Flask, render_template, request, jsonify

try:
    import openai  # type: ignore
except ImportError as exc:  # Fail fast if the dependency is missing
    raise RuntimeError("openai python package is not installed –\n"
                       "pip install openai>=1.14.0") from exc

app = Flask(__name__)

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


def search_web(query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
    """Return a list of search results using a simple JSON search API.

    The default provider is an unofficial read-only endpoint around DuckDuckGo
    results (no API key required).  You can point the function to a corporate
    micro-service or SerpAPI by setting the `SEARCH_API_ENDPOINT` env var.
    """
    endpoint = os.getenv(
        "SEARCH_API_ENDPOINT",
        "https://ddg-webapp-search.vercel.app/api/search",  # Fallback
    )
    try:
        resp = requests.get(endpoint, params={"q": query, "max_results": max_results}, timeout=8)
        resp.raise_for_status()
    except requests.RequestException as exc:
        # If the search fails we still want the conversation to continue
        app.logger.warning("Search error: %s", exc)
        return []

    # Expected schema: { "results": [ {"title": "..", "snippet": "..", "url": ".."}, ... ] }
    payload: Dict[str, Any] = resp.json()
    return payload.get("results", [])[:max_results]


def generate_ai_response(question: str) -> str:
    """Combine web search context with OpenAI chat completion."""
    # 1. Run web search
    results = search_web(question, max_results=8)
    context_lines = [f"- {res.get('title')}: {res.get('snippet')} (source: {res.get('url')})" for res in results]
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
    openai.api_key = os.environ["OPENAI_API_KEY"]
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
        try:
            boog_response = generate_ai_response(user_input)
        except KeyError:
            boog_response = (
                "OPENAI_API_KEY is not set on the server – AI mode is temporarily unavailable."
            )
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

