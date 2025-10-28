import os
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal

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

BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_CHARS = 4000

# ---------- Models ----------
class SearchItem(TypedDict):
    title: str
    url: str
    snippet: str


class ActionItem(TypedDict, total=False):
    """Represents a single tool decision."""

    type: Literal["search", "read_file"]
    reason: str
    query: str
    depth: Literal["basic", "advanced"]
    path: str


class ActionPlan(TypedDict):
    """Structure returned by the reasoning controller."""

    reasoning: List[str]
    actions: List[ActionItem]
    respond_directly: bool


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM text."""

    if not text:
        raise ValueError("Empty response")

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:120]}...")


def decide_agent_actions(client: Groq, user_message: str) -> ActionPlan:
    """Ask the model for a structured plan of action."""

    decision_system_prompt = (
        "You control Boog's tools. Decide whether to search the web or read files. "
        "Minimize unnecessary calls while ensuring accuracy. "
        "Respond with strict JSON using keys reasoning (list of short strings), "
        "actions (list) and respond_directly (boolean)."
    )

    tool_description = (
        "Tools available:\n"
        "1. search(query, depth) -> Use when the user needs fresh or missing knowledge. "
        "Return at most two search actions. Depth is 'basic' by default; choose 'advanced' "
        "only if essential.\n"
        "2. read_file(path) -> Use for repository files when explicitly needed. Paths must be "
        "relative to the project root, no parent-directory traversals, max three files."
    )

    decision_prompt = (
        f"User message:\n{user_message}\n\n"
        f"{tool_description}\n\n"
        "Return JSON shaped as:\n"
        "{\n"
        "  \"reasoning\": [\"step 1\", \"step 2\"],\n"
        "  \"actions\": [\n"
        "    {\"type\": \"search\", \"query\": \"...\", \"reason\": \"...\", \"depth\": \"basic\"},\n"
        "    {\"type\": \"read_file\", \"path\": \"README.md\", \"reason\": \"...\"}\n"
        "  ],\n"
        "  \"respond_directly\": true\n"
        "}\n\n"
        "If no tools are required, return an empty actions list and set respond_directly true."
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        messages=[
            {"role": "system", "content": decision_system_prompt},
            {"role": "user", "content": decision_prompt},
        ],
    )

    data = _extract_json_object(response.choices[0].message.content or "")

    reasoning = [str(item).strip() for item in data.get("reasoning", []) if str(item).strip()]
    actions: List[ActionItem] = []
    for raw in data.get("actions", []):
        if not isinstance(raw, dict):
            continue
        action_type = str(raw.get("type", "")).strip().lower()
        reason = str(raw.get("reason", "")).strip()
        if action_type == "search":
            query = str(raw.get("query", "")).strip()
            if not query:
                continue
            depth = str(raw.get("depth", "basic")).strip().lower()
            depth = "advanced" if depth == "advanced" else "basic"
            actions.append(
                ActionItem(
                    type="search",
                    reason=reason or "Investigate via web search.",
                    query=query,
                    depth=depth,
                )
            )
        elif action_type == "read_file":
            path = str(raw.get("path", "")).strip()
            if not path:
                continue
            actions.append(
                ActionItem(
                    type="read_file",
                    reason=reason or "Inspect repository file for details.",
                    path=path,
                )
            )

    # Enforce tool limits to keep costs predictable.
    search_actions = [a for a in actions if a["type"] == "search"][:2]
    file_actions = [a for a in actions if a["type"] == "read_file"][:3]
    actions = search_actions + file_actions

    respond_directly = bool(data.get("respond_directly", not actions))

    if not reasoning:
        reasoning = [
            "Analyzed the user's intent to decide whether external information is needed.",
        ]

    return ActionPlan(
        reasoning=reasoning,
        actions=actions,
        respond_directly=respond_directly,
    )


def safe_read_file(path: str, *, max_chars: int = MAX_FILE_CHARS) -> str:
    """Safely read a repository file and return a trimmed excerpt."""

    normalized = path.strip()
    if not normalized:
        raise ValueError("Empty file path")

    if ".." in Path(normalized).parts:
        raise ValueError("Parent directory references are not allowed")

    target = (BASE_DIR / normalized).resolve()
    if not target.is_file():
        raise ValueError("File not found")

    if BASE_DIR not in target.parents and target != BASE_DIR:
        raise ValueError("Path outside project scope")

    content = target.read_text(encoding="utf-8", errors="ignore")

    if len(content) > max_chars:
        return content[:max_chars] + "\n… [truncated]"
    return content


def execute_actions(plan: ActionPlan) -> Dict[str, Any]:
    """Run tool actions decided by the controller."""

    action_logs: List[str] = []
    search_outputs: List[Dict[str, Any]] = []
    file_outputs: List[Dict[str, str]] = []

    for action in plan["actions"]:
        if action["type"] == "search":
            query = action.get("query", "")
            depth = action.get("depth", "basic")
            try:
                results = tavily_search(query, k=4, depth=depth)
                search_outputs.append({
                    "query": query,
                    "depth": depth,
                    "reason": action.get("reason", ""),
                    "results": results,
                })
                action_logs.append(
                    f"Web search executed for '{query}' (depth: {depth})."
                )
            except Exception as exc:  # pragma: no cover - network errors
                app.logger.error("Tavily search failed: %s", exc)
                action_logs.append(
                    f"Web search for '{query}' failed: {exc}."
                )
        elif action["type"] == "read_file":
            path = action.get("path", "")
            try:
                content = safe_read_file(path)
                snippet = content[:1600]
                if len(content) > 1600:
                    snippet += "\n… [truncated]"
                file_outputs.append({
                    "path": path,
                    "reason": action.get("reason", ""),
                    "content": snippet,
                })
                action_logs.append(f"Read file '{path}'.")
            except Exception as exc:
                app.logger.error("File read failed: %s", exc)
                action_logs.append(
                    f"Could not read file '{path}': {exc}."
                )

    return {
        "search_outputs": search_outputs,
        "file_outputs": file_outputs,
        "action_logs": action_logs,
    }


def _format_context(search_outputs: List[Dict[str, Any]], file_outputs: List[Dict[str, str]]) -> str:
    """Prepare context notes for the response synthesizer."""

    blocks: List[str] = []

    for idx, block in enumerate(search_outputs, 1):
        label = f"search-{idx}"
        lines = [
            f"{label}: query='{block['query']}' depth='{block['depth']}' reason='{block.get('reason', '')}'"
        ]
        for jdx, result in enumerate(block["results"], 1):
            snippet = textwrap.shorten(result["snippet"], width=360, placeholder="…")
            lines.append(
                f"{label}.{jdx}: title='{result['title']}' url='{result['url']}' snippet='{snippet}'"
            )
        blocks.append("\n".join(lines))

    for idx, block in enumerate(file_outputs, 1):
        label = f"file-{idx}"
        excerpt = textwrap.shorten(block["content"], width=360, placeholder="…")
        blocks.append(
            f"{label}: path='{block['path']}' reason='{block.get('reason', '')}'\n{excerpt}"
        )

    return "\n\n".join(blocks) if blocks else "(no external context used)"


def synthesize_final_answer(
    client: Groq,
    user_message: str,
    plan: ActionPlan,
    search_outputs: List[Dict[str, Any]],
    file_outputs: List[Dict[str, str]],
) -> str:
    """Ask the LLM to write the final response using gathered context."""

    reasoning_summary = "\n".join(
        f"Step {idx + 1}: {text}" for idx, text in enumerate(plan["reasoning"])
    )
    context_blob = _format_context(search_outputs, file_outputs)

    synthesis_prompt = (
        "You are Boog, an aligned assistant. Use the provided context if it exists. "
        "Cite web search evidence using (search-1), (search-1.2) etc. Cite file excerpts as (file-1). "
        "If context is missing, respond from prior knowledge but note any uncertainty."
    )

    user_payload = (
        f"User message:\n{user_message}\n\n"
        f"Internal reasoning summary:\n{reasoning_summary or 'n/a'}\n\n"
        f"Context:\n{context_blob}\n\n"
        "Draft a clear, structured answer."
    )

    result = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.4,
        messages=[
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": user_payload},
        ],
    )

    return (result.choices[0].message.content or "").strip() or "(No response generated.)"


def build_transparent_response(
    plan: ActionPlan,
    action_logs: List[str],
    final_answer: str,
    search_outputs: List[Dict[str, Any]],
) -> str:
    """Compose the final markdown sent to the UI with reasoning transparency."""

    sections: List[str] = []

    if plan["reasoning"]:
        reasoning_lines = "\n".join(
            f"{idx + 1}. {line}" for idx, line in enumerate(plan["reasoning"])
        )
        sections.append(f"### Reasoning Workflow\n{reasoning_lines}")

    if action_logs:
        log_lines = "\n".join(f"- {log}" for log in action_logs)
        sections.append(f"### Tool Actions\n{log_lines}")
    else:
        sections.append(
            "### Tool Actions\n- No external tools were executed for this turn."
        )

    sections.append(f"### Final Answer\n{final_answer}")

    if search_outputs:
        link_lines: List[str] = []
        for idx, block in enumerate(search_outputs, 1):
            label = f"search-{idx}"
            for jdx, result in enumerate(block["results"], 1):
                title = result["title"] or result["url"] or "(untitled)"
                link_lines.append(
                    f"- ({label}.{jdx}) [{title}]({result['url']})"
                )
        if link_lines:
            sections.append("### Sources\n" + "\n".join(link_lines))

    return "\n\n".join(sections)


def run_agent(user_message: str) -> str:
    client = _groq()
    if not client:
        return "GROQ_API_KEY is not set on the server – AI mode is unavailable."

    plan = decide_agent_actions(client, user_message)
    execution = execute_actions(plan)

    final_answer = synthesize_final_answer(
        client,
        user_message,
        plan,
        execution["search_outputs"],
        execution["file_outputs"],
    )

    return build_transparent_response(
        plan,
        execution["action_logs"],
        final_answer,
        execution["search_outputs"],
    )

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

    if not user_input:
        return jsonify(response="Please provide a message.")

    try:
        resp = run_agent(user_input)
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("Agent pipeline error: %s", exc)
        resp = "An internal error occurred while generating a response. Please try again."

    return jsonify(response=resp)


# ---------- Entrypoint ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Setting threaded=True plays nicer with Groq and HTTP requests concurrency
    app.run(host="0.0.0.0", port=port, threaded=True)
