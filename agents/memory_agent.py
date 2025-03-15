class MemoryAgent:
    def __init__(self, memory_system, conversation_manager):
        self.memory = memory_system
        self.conversation = conversation_manager

    def process_query(self, query):
        """Process a memory-related query"""
        query_lower = query.lower()

        # Summarize current context
        if "context" in query_lower or "summary" in query_lower or "status" in query_lower or "what we've done" in query_lower or "so far" in query_lower:
            return self._summarize_context()

        # List operations history
        if "history" in query_lower or "operations" in query_lower:
            return self._list_operations()

        # List visualizations
        if "visualizations" in query_lower or "plots" in query_lower:
            return self._list_visualizations()

        # List conversation history
        if "conversation" in query_lower:
            return self._summarize_conversation()

        # Save checkpoint
        if "save" in query_lower or "checkpoint" in query_lower:
            return self._save_checkpoint()

        # Default response
        return {
            "success": False,
            "message": "I couldn't understand your memory query. Try asking for context summary, operation history, or visualization list.",
            "result": None
        }

    def _summarize_context(self):
        """Summarize the current system context"""
        context = self.memory.summarize_current_context()

        # Add information about current conversation
        context["conversation"] = {
            "session_id": self.conversation.current_session_id,
            "message_count": len(self.conversation.get_session_history()),
            "summary": self.conversation.summarize_conversation()
        }

        return {
            "success": True,
            "message": "Current system context summary",
            "result": context
        }

    def _list_operations(self, limit=10):
        """List recent operations"""
        operations = self.memory.memory["operation_history"]
        recent_ops = operations[-limit:] if len(operations) > limit else operations

        return {
            "success": True,
            "message": f"Recent operations (showing {len(recent_ops)} of {len(operations)} total)",
            "result": recent_ops
        }

    def _list_visualizations(self):
        """List all visualizations"""
        visualizations = self.memory.memory["visualizations"]

        return {
            "success": True,
            "message": f"All visualizations ({len(visualizations)} total)",
            "result": visualizations
        }

    def _summarize_conversation(self):
        """Summarize the conversation history"""
        history = self.conversation.get_session_history()
        recent_messages = self.conversation.get_recent_history(5)

        summary = {
            "session_id": self.conversation.current_session_id,
            "total_messages": len(history),
            "summary": self.conversation.summarize_conversation(),
            "recent_messages": recent_messages
        }

        return {
            "success": True,
            "message": "Conversation summary",
            "result": summary
        }

    def _save_checkpoint(self):
        """Save the current system state"""
        memory_path = self.memory.save_checkpoint()
        conversation_path = self.conversation.save_conversation()

        return {
            "success": True,
            "message": "Successfully saved system checkpoint",
            "result": {
                "memory_path": memory_path,
                "conversation_path": conversation_path
            }
        }

    def update_after_operation(self, operation_type, operation_details, result):
        """Update memory after an operation is performed by another agent"""
        # Here, you would implement logic to condense and summarize operations
        # This is a simplified version

        if operation_type == "data_processing":
            # Update key variables or add notes about data state changes
            pass
        elif operation_type == "visualization":
            # Add notes about generated visualizations
            pass

        # Log the operation in conversation history
        self.conversation.add_message(
            "system",
            f"Performed {operation_type}: {operation_details}",
            {"type": "operation_log", "result": result}
        )

        return {
            "success": True,
            "message": f"Memory updated after {operation_type} operation",
            "result": self.memory.summarize_current_context(max_ops=3)
        }