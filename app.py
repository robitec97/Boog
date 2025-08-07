"""BoogGPT – simplified backend powered by Groq GPT‑OSS 120B

Changelog (2025‑08‑07)
----------------------
* **Removed web search layer** – the assistant now behaves like a straight‑up
  conversational chatbot. No external searches are performed.
* **Switched to Groq API** – `generate_ai_response()` now calls Groq’s
  `openai/gpt-oss-120b` model via the official `groq` Python client. Supply your
  API key through the `GROQ_API_KEY` Heroku config var.
* **Renamed AI personality** – the cat persona is retired; meet *BoogGPT*,
  a friendly, helpful assistant.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List

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
# Groq SDK (required for AI mode)
# ---------------------------------------------------------------------------
try:
    from groq import Groq  # type: ignore
except ImportError as exc:  # Fail fast if the dependency is missing
    raise RuntimeError(
        "groq python package is not installed –\n"
        "pip install groq>=0.5.0"
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
    """Append the chat pair to a local JSON log (best‑effort)."""
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


# --------------------------- AI generation ----------------------------------
def generate_ai_response(prompt: str) -> str:
    """Generate a chat completion using Groq GPT‑OSS 120B."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return (
            "GROQ_API_KEY is not set on the server – AI mode is temporarily unavailable."
        )

    # Lazily create client to avoid import overhead in non‑AI requests
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are BoogGPT – a friendly, helpful AI assistant. "
                        "Keep answers concise and clear."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        answer = response.choices[0].message.content.strip()
        return answer or "(No response from BoogGPT 🙀)"
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.error("Groq API error: %s", exc)
        return "Sorry, BoogGPT ran into an error contacting the language model."

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
    # Setting threaded=True plays nicer with Groq and HTTP requests concurrency
    app.run(host="0.0.0.0", port=port, threaded=True)
