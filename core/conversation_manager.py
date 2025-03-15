import json
from datetime import datetime
import os


class ConversationManager:
    def __init__(self, memory_system, persistence_path="./data/conversations/"):
        self.memory_system = memory_system
        self.persistence_path = persistence_path
        self.conversation_history = []
        self.current_session_id = self._generate_session_id()

        # Create persistence directory if it doesn't exist
        os.makedirs(persistence_path, exist_ok=True)

    def _generate_session_id(self):
        """Generate a unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_message(self, sender, content, metadata=None):
        """Add a message to the conversation history"""
        message = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender,
            "content": content
        }

        if metadata:
            message["metadata"] = metadata

        self.conversation_history.append(message)
        return message

    def get_session_history(self, session_id=None):
        """Get conversation history for a specific session"""
        if session_id is None:
            session_id = self.current_session_id

        return [msg for msg in self.conversation_history if msg["session_id"] == session_id]

    def get_recent_history(self, n=5):
        """Get the n most recent messages"""
        return self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history

    def save_conversation(self, filename=None):
        """Save conversation history to disk"""
        if filename is None:
            filename = f"conversation_{self.current_session_id}.json"

        file_path = os.path.join(self.persistence_path, filename)

        with open(file_path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

        return file_path

    def load_conversation(self, file_path):
        """Load conversation history from disk"""
        with open(file_path, 'r') as f:
            self.conversation_history = json.load(f)

        # Set current session to the last session in the history
        if self.conversation_history:
            self.current_session_id = self.conversation_history[-1]["session_id"]

        return self.conversation_history

    def start_new_session(self):
        """Start a new conversation session"""
        self.current_session_id = self._generate_session_id()
        return self.current_session_id

    def summarize_conversation(self, session_id=None, max_length=200):
        """Create a summary of the conversation"""
        history = self.get_session_history(session_id)

        if not history:
            return "No conversation history found."

        summary = f"Conversation with {len(history)} messages. "

        # Extract key information
        user_queries = [msg["content"] for msg in history if msg["sender"] == "user"]

        if user_queries:
            first_query = user_queries[0]
            last_query = user_queries[-1]
            summary += f"Started with: '{first_query[:50]}{'...' if len(first_query) > 50 else ''}'. "
            summary += f"Latest query: '{last_query[:50]}{'...' if len(last_query) > 50 else ''}'."

        return summary[:max_length] + "..." if len(summary) > max_length else summary