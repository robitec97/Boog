"""
Session and conversation management for Boog agent.

Provides in-memory storage for conversation history with automatic pruning
and session expiration.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)


class Message:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str, **kwargs):
        """
        Create a message.

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            **kwargs: Additional metadata (tool_call_id, tool_calls, etc.)
        """
        self.role = role
        self.content = content
        self.metadata = kwargs
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Groq API."""
        msg = {
            "role": self.role,
            "content": self.content,
        }
        # Add metadata fields
        msg.update(self.metadata)
        return msg

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content[:50]}...)"


class Conversation:
    """Manages a single conversation thread."""

    def __init__(self, session_id: str, conversation_id: str = None):
        """
        Create a conversation.

        Args:
            session_id: Parent session ID
            conversation_id: Unique conversation ID (auto-generated if None)
        """
        self.session_id = session_id
        self.id = conversation_id or str(uuid.uuid4())
        self.messages: List[Message] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()

    def add_message(self, role: str, content: str, **kwargs):
        """
        Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            **kwargs: Additional metadata
        """
        self.messages.append(Message(role, content, **kwargs))
        self.last_active = datetime.now()
        logger.debug(f"Added {role} message to conversation {self.id}")

    def get_messages(self, max_messages: int = 20) -> List[Dict[str, Any]]:
        """
        Get messages as dicts for API calls.

        Implements simple pruning: keeps last max_messages.

        Args:
            max_messages: Maximum number of messages to return

        Returns:
            List of message dicts
        """
        # Keep last N messages
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [m.to_dict() for m in recent]

    def count_approximate_tokens(self) -> int:
        """
        Estimate token count for conversation.

        Uses rough heuristic: 4 characters = 1 token.

        Returns:
            Estimated token count
        """
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4

    def clear(self):
        """Clear all messages from conversation."""
        self.messages = []
        self.last_active = datetime.now()
        logger.info(f"Cleared conversation {self.id}")

    def __repr__(self):
        return f"Conversation(id={self.id}, messages={len(self.messages)})"


class Session:
    """Manages multiple conversation threads for a user session."""

    def __init__(self, session_id: str = None):
        """
        Create a session.

        Args:
            session_id: Unique session ID (auto-generated if None)
        """
        self.id = session_id or str(uuid.uuid4())
        self.conversations: Dict[str, Conversation] = {}
        self.created_at = datetime.now()
        self.last_active = datetime.now()

    def get_conversation(self, conversation_id: str = None) -> Conversation:
        """
        Get or create a conversation.

        Args:
            conversation_id: Conversation ID (creates new if None or not found)

        Returns:
            Conversation instance
        """
        if conversation_id is None:
            # Create new conversation
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(self.id, conversation_id)
            logger.info(f"Created new conversation {conversation_id} in session {self.id}")

        self.last_active = datetime.now()
        return self.conversations[conversation_id]

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation from session."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation {conversation_id} from session {self.id}")

    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """
        Check if session has expired.

        Args:
            timeout_minutes: Inactivity timeout in minutes

        Returns:
            True if session is expired
        """
        return datetime.now() - self.last_active > timedelta(minutes=timeout_minutes)

    def __repr__(self):
        return f"Session(id={self.id}, conversations={len(self.conversations)})"


class SessionManager:
    """Global manager for all sessions."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        logger.info("SessionManager initialized")

    def get_or_create_session(self, session_id: str = None) -> Session:
        """
        Get existing session or create new one.

        Args:
            session_id: Session ID (creates new if None or not found)

        Returns:
            Session instance
        """
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            logger.debug(f"Retrieved existing session {session_id}")
            return session

        # Create new session
        session = Session(session_id)
        self.sessions[session.id] = session
        logger.info(f"Created new session {session.id}")
        return session

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")

    def cleanup_expired_sessions(self, timeout_minutes: int = 60) -> int:
        """
        Remove expired sessions.

        Args:
            timeout_minutes: Inactivity timeout in minutes

        Returns:
            Number of sessions cleaned up
        """
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(timeout_minutes)
        ]

        for sid in expired:
            del self.sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about sessions.

        Returns:
            Dict with session count, conversation count
        """
        total_conversations = sum(
            len(session.conversations)
            for session in self.sessions.values()
        )
        total_messages = sum(
            len(conv.messages)
            for session in self.sessions.values()
            for conv in session.conversations.values()
        )

        return {
            "sessions": len(self.sessions),
            "conversations": total_conversations,
            "messages": total_messages
        }

    def __repr__(self):
        stats = self.get_stats()
        return f"SessionManager(sessions={stats['sessions']}, conversations={stats['conversations']})"
