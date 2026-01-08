import os
import logging
from datetime import datetime
from functools import wraps

from groq import Groq
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from apscheduler.schedulers.background import BackgroundScheduler

# Import our new modules
from tools import create_default_registry
from session import SessionManager
from agent import generate_agent_response, stream_agent_response


# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize global components
tool_registry = create_default_registry()
session_manager = SessionManager()

# Get configuration
SESSION_TIMEOUT = int(os.getenv("BOOG_SESSION_TIMEOUT", 60))  # minutes

# ---------- Rate Limiting ----------
rate_limit_storage = {}

def rate_limit(max_requests=30, window_seconds=60):
    """Rate limiting decorator to prevent abuse."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            payload = request.get_json(silent=True) or {}
            session_id = payload.get('session_id', 'anonymous')
            now = datetime.now()

            if session_id not in rate_limit_storage:
                rate_limit_storage[session_id] = []

            # Clean old requests
            from datetime import timedelta
            rate_limit_storage[session_id] = [
                ts for ts in rate_limit_storage[session_id]
                if now - ts < timedelta(seconds=window_seconds)
            ]

            if len(rate_limit_storage[session_id]) >= max_requests:
                return jsonify({
                    "error": "Rate limit exceeded. Please wait a moment.",
                    "retry_after": window_seconds
                }), 429

            rate_limit_storage[session_id].append(now)
            return f(*args, **kwargs)

        return wrapped
    return decorator


# ---------- Groq Client ----------
def _groq():
    """Get Groq client instance."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key)


# ---------- Session Cleanup ----------
def cleanup_sessions():
    """Periodic task to clean up expired sessions."""
    count = session_manager.cleanup_expired_sessions(timeout_minutes=SESSION_TIMEOUT)
    if count > 0:
        app.logger.info(f"Cleaned up {count} expired sessions")

# Setup background scheduler for session cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_sessions, trigger="interval", minutes=15)
scheduler.start()


# ---------- Flask Routes ----------
@app.route("/")
def index():
    """Serve main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@rate_limit(max_requests=30, window_seconds=60)
def chat():
    """
    Main chat endpoint with agent capabilities.

    Supports both streaming (query param stream=true) and non-streaming modes.
    """
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id")
    conversation_id = payload.get("conversation_id")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Check if streaming is requested via query param
    stream_mode = request.args.get('stream', 'false').lower() == 'true'

    # Get Groq client
    groq_client = _groq()
    if not groq_client:
        return jsonify({
            "error": "GROQ_API_KEY is not configured on the server"
        }), 503

    # Get or create session and conversation
    session = session_manager.get_or_create_session(session_id)
    conversation = session.get_conversation(conversation_id)

    # Add user message to conversation
    conversation.add_message("user", user_message)

    app.logger.info(f"Processing message in session {session.id}, conversation {conversation.id}")

    # Handle streaming mode
    if stream_mode:
        def generate():
            try:
                for event in stream_agent_response(conversation, tool_registry, groq_client):
                    yield event
            except Exception as e:
                app.logger.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: {{\"type\": \"error\", \"message\": \"Streaming failed\"}}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    # Handle non-streaming mode
    try:
        response_data = generate_agent_response(conversation, tool_registry, groq_client)
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Agent error: {e}", exc_info=True)
        return jsonify({
            "error": f"Failed to generate response: {str(e)}",
            "session_id": session.id,
            "conversation_id": conversation.id
        }), 500


@app.route("/session/clear", methods=["POST"])
def clear_session():
    """Clear a conversation (creates new conversation ID)."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    conversation_id = payload.get("conversation_id")

    if session_id and conversation_id:
        session = session_manager.get_or_create_session(session_id)
        session.delete_conversation(conversation_id)
        app.logger.info(f"Cleared conversation {conversation_id}")
        return jsonify({"success": True})

    return jsonify({"error": "Missing session_id or conversation_id"}), 400


@app.route("/stats", methods=["GET"])
def stats():
    """Get system statistics (for debugging)."""
    return jsonify(session_manager.get_stats())


# ---------- Shutdown Handler ----------
@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    """Shutdown scheduler on app teardown."""
    if scheduler.running:
        scheduler.shutdown()


# ---------- Entrypoint ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"Starting Boog on port {port}")
    app.logger.info(f"Workspace directory: {os.getenv('BOOG_WORKSPACE_DIR', '/tmp/boog_workspace')}")
    app.logger.info(f"Session timeout: {SESSION_TIMEOUT} minutes")
    app.run(host="0.0.0.0", port=port, threaded=True)
