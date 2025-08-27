import os, json, logging
from typing import List, TypedDict, Optional

try:
    import requests
    _HAVE_REQUESTS = True
except ModuleNotFoundError:
    import urllib.request, urllib.error
    _HAVE_REQUESTS = False

from groq import Groq
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# ---------- Models ----------
class SearchItem(TypedDict):
    title: str
    url: str
    snippet: str

# ---------- HTTP helper ----------
def _post_json(url: str, headers: dict, payload: dict) -> dict:
    if _HAVE_REQUESTS:
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        r.raise_for_status()
        return r.json()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=12) as resp:  # nosec
        return json.loads(resp.read().decode("utf-8"))

# ---------- Tavily search ----------
def tavily_search(query: str, k: int = 5, depth: str = "basic") -> List[SearchItem]:
    """
    depth: 'basic' (1 credit) or 'advanced' (2 credits).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    # Respect ~400 char query limit (hard trim as a guard).
    q = (query or "").strip()
    if len(q) > 400:
        q = q[:400]

    payload = {
        "query": q,
        "search_depth": depth,          # 'basic' or 'advanced'
        "include_answer": False,        # we let Groq synthesize
        "include_raw_content": False,
        "max_results": max(1, min(k, 8)),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = _post_json("https://api.tavily.com/search", headers, payload)

    items: List[SearchItem] = []
    for r in data.get("results", [])[:k]:
        items.append({
            "title": (r.get("title") or "").strip(),
            "url": r.get("url") or "",
            "snippet": (r.get("content") or "").strip()[:500],
        })
    return items

# ---------- Groq LLM ----------
def _groq() -> Optional[Groq]:
    key = os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None

def generate_ai_response(prompt: str) -> str:  # unchanged pure-LLM path
    client = _groq()
    if not client:
        return "GROQ_API_KEY is not set on the server – AI mode is unavailable."
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are Boog – concise, helpful."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip() or "(No response)"

def answer_with_web_search(query: str, k: int = 5, depth: str = "basic") -> str:
    try:
        results = tavily_search(query, k=k, depth=depth)
    except Exception as exc:
        app.logger.error("Tavily error: %s", exc)
        return "Web search is temporarily unavailable."

    if not results:
        return "No results found."

    # Build grounded prompt
    sources_txt = []
    for i, r in enumerate(results, 1):
        title = r["title"] or r["url"]
        sources_txt.append(
            f"[{i}] {title}\nURL: {r['url']}\nSnippet: {r['snippet']}"
        )
    prompt = (
        "Use the numbered sources to answer. Cite like [1], [2]. "
        "Only include supported claims; if unclear, say so.\n\n"
        f"USER QUESTION:\n{query}\n\nSOURCES:\n" + "\n\n".join(sources_txt)
    )

    client = _groq()
    if not client:
        return "GROQ_API_KEY is not set on the server – AI mode is unavailable."

    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Ground answers in the sources and cite."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    answer = (r.choices[0].message.content or "").strip()

    links = "\n".join(
        f"- [{i}] {it['title'] or it['url']} — {it['url']}"
        for i, it in enumerate(results, 1)
    )
    return f"{answer}\n\n---\n**Sources (links):**\n{links}"


# ---------- Flask Routes ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input: str = (payload.get("message") or "").strip()
    mode: str = (payload.get("mode") or "ai").lower()  # "ai" | "web"

    if not user_input:
        return jsonify(response="Please provide a message.")

    if mode in ("web", "web-search", "search"):
        # You can flip to depth="advanced" for tougher queries (costs 2 credits).
        resp = answer_with_web_search(user_input, k=5, depth="basic")
    else:
        resp = generate_ai_response(user_input)

    return jsonify(response=resp)


# ---------- Entrypoint ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Setting threaded=True plays nicer with Groq and HTTP requests concurrency
    app.run(host="0.0.0.0", port=port, threaded=True)
