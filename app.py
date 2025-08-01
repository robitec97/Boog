import os
import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

boog_quotes = {
    "wisdom": [
        "Sleep is the answer to most problems.",
        "If it fits, sit. If it doesn’t fit, sit anyway.",
        "The humans rush, the cat observes.",
        "Sometimes doing nothing is doing everything.",
        "Patience is a purr-tue."
    ],
    "roast": [
        "That's your plan? I've seen mice with better strategy.",
        "You're typing? Cute. Still won't fix your code.",
        "Bold of you to assume anyone cares.",
        "Wow. Even the litter box smells better than that idea.",
        "You again? I was hoping for someone interesting."
    ]
}

LOG_FILE = os.path.join("/tmp", "boog_log.json")  # Heroku’s dyno FS is ephemeral

def log_chat(user_msg, boog_msg):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_msg,
        "boog": boog_msg
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "").strip()
    mode = payload.get("mode", "wisdom")
    if mode not in boog_quotes:
        mode = "roast"

    boog_response = random.choice(boog_quotes[mode])
    log_chat(user_input, boog_response)
    return jsonify(response=boog_response)

# Heroku injects $PORT; default to 5000 for local dev
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
