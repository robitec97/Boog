from flask import Flask, render_template, request, jsonify
import random
import json
from datetime import datetime

app = Flask(__name__)

boog_quotes = {
    'wisdom': [
        "Sleep is the answer to most problems.",
        "If it fits, sit. If it doesn’t fit, sit anyway.",
        "The humans rush, the cat observes.",
        "Sometimes doing nothing is doing everything.",
        "Patience is a purr-tue."
    ],
    'roast': [
        "That's your plan? I've seen mice with better strategy.",
        "You're typing? Cute. Still won't fix your code.",
        "Bold of you to assume anyone cares.",
        "Wow. Even the litter box smells better than that idea.",
        "You again? I was hoping for someone interesting."
    ]
}

def log_chat(user_msg, boog_msg):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg,
        "boog": boog_msg
    }
    try:
        with open("boog_log.json", "r+") as file:
            data = json.load(file)
            data.append(entry)
            file.seek(0)
            json.dump(data, file, indent=2)
    except FileNotFoundError:
        with open("boog_log.json", "w") as file:
            json.dump([entry], file, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    mode = request.json.get("mode", "wisdom")
    if mode not in boog_quotes:
        mode = "roast"

    boog_response = random.choice(boog_quotes[mode])
    log_chat(user_input, boog_response)
    return jsonify({"response": boog_response})

if __name__ == "__main__":
    app.run(debug=True)
