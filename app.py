import os
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, TypedDict

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

DEFAULT_MODEL = "openai/gpt-oss-120b"
SESSION_TTL_SECONDS = 45 * 60  # prune idle conversations after 45 minutes

# ---------- Models ----------
class SearchItem(TypedDict):
    title: str
    url: str
    snippet: str


@dataclass
class ConversationTurn:
    role: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationMemory:
    turns: Deque[ConversationTurn] = field(
        default_factory=lambda: deque(maxlen=18)
    )
    summary: str = ""
    last_active: float = field(default_factory=time.time)

    def append(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.turns.append(ConversationTurn(role=role, content=content, meta=meta or {}))
        self.last_active = time.time()

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def as_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]


@dataclass
class SearchAnswer:
    answer: str
    sources: List[SearchItem] = field(default_factory=list)


@dataclass
class AgentDecision:
    action: str
    reason: str
    mode: str = "autonomous"
    forced: bool = False
    search_query: str = ""
    search_depth: str = "basic"
    used_search: bool = False
    search_results: List[SearchItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["search_results"] = self.search_results
        return data


@dataclass
class AgentResult:
    conversation_id: str
    text: str
    decision: AgentDecision
    memory: ConversationMemory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "response": self.text,
            "meta": {
                "decision": {
                    key: value
                    for key, value in self.decision.to_dict().items()
                    if key != "search_results"
                },
                "sources": self.decision.search_results,
                "memory": {
                    "summary": self.memory.summary,
                    "turns": self.memory.turn_count,
                },
            },
        }

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
_GROQ_CLIENT: Optional[Groq] = None


def _groq() -> Optional[Groq]:
    """Return a cached Groq client if credentials are available."""

    global _GROQ_CLIENT
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=key)
    return _GROQ_CLIENT


def _call_groq(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.6,
    response_format: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Utility wrapper that safely executes a Groq chat completion."""

    client = _groq()
    if not client:
        return None

    try:
        params: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            params["response_format"] = response_format
        response = client.chat.completions.create(**params)
    except Exception as exc:  # pragma: no cover - network heavy
        app.logger.error("Groq error: %s", exc)
        return None

    message = response.choices[0].message.content if response.choices else ""
    if not message:
        return None
    return message.strip() or None


def generate_ai_response(prompt: str) -> str:
    """Fallback helper that mirrors the legacy behaviour."""

    content = _call_groq(
        [
            {"role": "system", "content": "You are Boog – concise, helpful."},
            {"role": "user", "content": prompt},
        ]
    )
    if content is None:
        return "GROQ_API_KEY is not set on the server – AI mode is unavailable."
    return content or "(No response)"


def answer_with_web_search(
    query: str,
    k: int = 5,
    depth: str = "basic",
    context: Optional[str] = None,
) -> SearchAnswer:
    try:
        results = tavily_search(query, k=k, depth=depth)
    except Exception as exc:
        app.logger.error("Tavily error: %s", exc)
        return SearchAnswer(
            answer="Web search is temporarily unavailable.",
            sources=[],
        )

    if not results:
        return SearchAnswer(answer="No results found.", sources=[])

    sources_txt = []
    for i, r in enumerate(results, 1):
        title = r["title"] or r["url"]
        sources_txt.append(
            f"[{i}] {title}\nURL: {r['url']}\nSnippet: {r['snippet']}"
        )
    prompt = (
        "Use the numbered sources to answer the user. Cite the sources inline as [1], [2]. "
        "If the information is insufficient, state the limitations clearly.\n\n"
        f"USER QUESTION:\n{query}\n\nSOURCES:\n" + "\n\n".join(sources_txt)
    )
    if context:
        prompt += f"\n\nCONVERSATION CONTEXT:\n{context}"

    answer = _call_groq(
        [
            {"role": "system", "content": "Ground answers in the supplied sources and cite them."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    if answer is None:
        answer = "Unable to access the language model right now."

    return SearchAnswer(answer=answer, sources=results)


def _format_recent_dialogue(memory: ConversationMemory, limit: int = 6) -> str:
    recent: List[ConversationTurn] = list(memory.turns)[-limit:]
    formatted = []
    for turn in recent:
        if turn.role not in {"user", "assistant"}:
            continue
        speaker = "User" if turn.role == "user" else "Boog"
        formatted.append(f"{speaker}: {turn.content}")
    return "\n".join(formatted)


class BoogAgent:
    """Stateful agent that manages conversation memory and tool use."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _get_memory(self, session_id: str) -> ConversationMemory:
        key = session_id or "default"
        with self._lock:
            memory = self._sessions.get(key)
            if memory is None:
                memory = ConversationMemory()
                self._sessions[key] = memory
        return memory

    def _prune_sessions(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                session_id
                for session_id, memory in self._sessions.items()
                if now - memory.last_active > SESSION_TTL_SECONDS
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    def _heuristic_decision(self, user_input: str, default_depth: str) -> AgentDecision:
        lowered = user_input.lower()
        factual_keywords = (
            "latest",
            "recent",
            "update",
            "news",
            "current",
            "today",
            "official",
            "evidence",
            "statistics",
            "data",
            "http://",
            "https://",
        )
        question_words = ("who", "what", "when", "where", "why", "how")
        if any(keyword in lowered for keyword in factual_keywords) or (
            "?" in user_input and lowered.split()[:1] and lowered.split()[0] in question_words
        ):
            return AgentDecision(
                action="search_and_respond",
                reason="Heuristic: the query looks factual or time-sensitive.",
                search_query=user_input,
                search_depth=default_depth,
                used_search=True,
            )
        return AgentDecision(
            action="respond",
            reason="Heuristic: the request appears conversational or creative.",
            search_depth=default_depth,
        )

    def _autonomous_decision(
        self,
        memory: ConversationMemory,
        user_input: str,
        default_depth: str,
    ) -> AgentDecision:
        payload = {
            "summary": memory.summary,
            "recent_messages": [
                {"role": turn.role, "content": turn.content}
                for turn in list(memory.turns)[-6:]
            ],
            "user_input": user_input,
        }

        response_text = _call_groq(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the reasoning module for Boog, an autonomous assistant. "
                        "Decide whether to respond directly or to perform web search first. "
                        "Return strictly valid JSON with keys 'action', 'reason', 'search_query', 'search_depth'. "
                        "Valid actions: 'respond', 'search', 'search_and_respond'. "
                        "Choose a search action when the user needs external facts, citations, or recent information."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        if response_text:
            try:
                data = json.loads(response_text)
                action = str(data.get("action", "respond")).strip().lower()
                reason = str(data.get("reason", "")).strip() or "Autonomous decision."
                search_query = str(data.get("search_query", "")).strip() or user_input
                depth = str(data.get("search_depth", default_depth)).strip() or default_depth
                if action not in {"respond", "search", "search_and_respond"}:
                    action = "respond"
                decision = AgentDecision(
                    action=action if action != "search" else "search_and_respond",
                    reason=reason,
                    mode="autonomous",
                    search_query=search_query,
                    search_depth=depth if depth in {"basic", "advanced"} else default_depth,
                    used_search=action in {"search", "search_and_respond"},
                )
                return decision
            except json.JSONDecodeError:
                app.logger.debug("Failed to parse autonomous decision JSON: %s", response_text)

        return self._heuristic_decision(user_input, default_depth)

    def _build_context(self, memory: ConversationMemory) -> str:
        parts: List[str] = []
        if memory.summary:
            parts.append(f"Summary: {memory.summary}")
        recent = _format_recent_dialogue(memory)
        if recent:
            parts.append("Recent dialogue:\n" + recent)
        return "\n\n".join(parts)

    def _respond_with_memory(self, memory: ConversationMemory) -> str:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Boog, an autonomous, thoughtful AI cat assistant. "
                    "Answer using the conversation context. Be proactive, concise yet thorough, and cite prior search results if referenced."
                ),
            }
        ]
        if memory.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Conversation summary so far: {memory.summary}",
                }
            )
        messages.extend(memory.as_messages())

        response = _call_groq(messages, temperature=0.55)
        if response is not None:
            return response

        latest_user = next(
            (turn.content for turn in reversed(memory.turns) if turn.role == "user"),
            "",
        )
        return generate_ai_response(latest_user) if latest_user else "I am unable to respond right now."

    def _maybe_summarize(self, memory: ConversationMemory) -> None:
        if memory.turn_count < memory.turns.maxlen:
            return

        transcript = "\n".join(
            f"{turn.role.upper()}: {turn.content}"
            for turn in memory.turns
        )
        summary = _call_groq(
            [
                {
                    "role": "system",
                    "content": (
                        "Summarise the following chat transcript in 3 concise bullet points. "
                        "Focus on decisions, user preferences, and pending tasks."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            temperature=0.2,
        )
        if summary:
            memory.summary = summary
            while memory.turn_count > memory.turns.maxlen // 2:
                memory.turns.popleft()

    # ------------------------------------------------------------------
    def handle_message(
        self,
        session_id: str,
        user_input: str,
        *,
        forced_mode: Optional[str] = None,
        search_depth: str = "basic",
    ) -> AgentResult:
        memory = self._get_memory(session_id)
        memory.append("user", user_input)

        normalized_mode = (forced_mode or "").strip().lower()
        if normalized_mode in {"web", "web-search", "search"}:
            decision = AgentDecision(
                action="search_and_respond",
                reason="User enabled Web Search mode.",
                mode="user_web",
                forced=True,
                search_query=user_input,
                search_depth=search_depth,
                used_search=True,
            )
        else:
            decision = self._autonomous_decision(memory, user_input, search_depth)

        context = self._build_context(memory)
        if decision.used_search:
            search_query = decision.search_query or user_input
            answer = answer_with_web_search(
                search_query,
                k=5,
                depth=decision.search_depth,
                context=context,
            )
            decision.search_results = answer.sources
            response_text = answer.answer
            if not decision.search_results:
                decision.reason += " (Search returned no results.)"
        else:
            response_text = self._respond_with_memory(memory)

        memory.append(
            "assistant",
            response_text,
            meta={
                "decision": decision.action,
                "used_search": decision.used_search,
                "forced": decision.forced,
            },
        )

        self._maybe_summarize(memory)
        self._prune_sessions()

        return AgentResult(
            conversation_id=session_id or "default",
            text=response_text,
            decision=decision,
            memory=memory,
        )


AGENT = BoogAgent()


# ---------- Flask Routes ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input: str = (payload.get("message") or "").strip()
    mode: str = (payload.get("mode") or "ai").lower()
    session_id: str = (
        payload.get("session_id")
        or request.cookies.get("boog-session-id", "")
        or request.remote_addr
        or "default"
    )

    if not user_input:
        return jsonify({
            "conversation_id": session_id,
            "response": "Please provide a message.",
            "meta": {"error": True},
        })

    search_depth = str(payload.get("search_depth") or "basic").lower()
    if search_depth not in {"basic", "advanced"}:
        search_depth = "basic"

    result = AGENT.handle_message(
        session_id,
        user_input,
        forced_mode=mode,
        search_depth=search_depth,
    )

    return jsonify(result.to_dict())


# ---------- Entrypoint ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Setting threaded=True plays nicer with Groq and HTTP requests concurrency
    app.run(host="0.0.0.0", port=port, threaded=True)
