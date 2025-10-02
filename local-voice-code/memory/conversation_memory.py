"""
Conversation Memory Management

This module provides flexible memory management for chatbot conversations,
including support for:
- Fixed window memory (keep last N messages)
- Token-based memory (trim based on token count)
- Summary memory (compress old messages into summaries)
- Buffer memory (simple append-only)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class Message:
    """Represents a single conversation message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class BaseMemory(ABC):
    """Abstract base class for memory implementations"""
    
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.messages: List[Message] = []
        
    @abstractmethod
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new message to memory"""
        raise NotImplementedError
    
    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Get messages formatted for the LLM"""
        raise NotImplementedError
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add user message"""
        self.add_message("user", content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add assistant message"""
        self.add_message("assistant", content, metadata)
    
    def get_conversation_length(self) -> int:
        """Get the number of messages in conversation"""
        return len(self.messages)
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary"""
        return {
            "type": self.__class__.__name__,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMemory':
        """Deserialize memory from dictionary"""
        memory = cls(system_prompt=data.get("system_prompt", ""))
        memory.messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        return memory


class BufferMemory(BaseMemory):
    """Simple buffer memory that keeps all messages"""
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to buffer"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
    
    def get_messages(self) -> List[Message]:
        """Return all messages"""
        return self.messages.copy()


class WindowMemory(BaseMemory):
    """Fixed window memory that keeps last N messages"""
    
    def __init__(self, system_prompt: str = "", window_size: int = 10):
        super().__init__(system_prompt)
        self.window_size = window_size
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message and maintain window size"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        
        # Keep only the last window_size messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    
    def get_messages(self) -> List[Message]:
        """Return messages within window"""
        return self.messages.copy()


class TokenBasedMemory(BaseMemory):
    """Memory that trims messages based on estimated token count"""
    
    def __init__(self, system_prompt: str = "", max_tokens: int = 2000, tokens_per_message: int = 50):
        super().__init__(system_prompt)
        self.max_tokens = max_tokens
        self.tokens_per_message = tokens_per_message  # Rough estimate
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message and trim if necessary"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        
        # Estimate tokens and trim if needed
        self._trim_to_token_limit()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _trim_to_token_limit(self) -> None:
        """Trim messages to stay within token limit"""
        total_tokens = 0
        
        # Count tokens from the end (keep most recent)
        for i in range(len(self.messages) - 1, -1, -1):
            message_tokens = self._estimate_tokens(self.messages[i].content)
            
            if total_tokens + message_tokens > self.max_tokens:
                # Remove older messages
                self.messages = self.messages[i + 1:]
                break
                
            total_tokens += message_tokens
    
    def get_messages(self) -> List[Message]:
        """Return messages within token limit"""
        return self.messages.copy()


class SummaryMemory(BaseMemory):
    """Memory that summarizes old messages when context gets too long"""
    
    def __init__(self, system_prompt: str = "", max_messages: int = 8, summarize_threshold: int = 12):
        super().__init__(system_prompt)
        self.max_messages = max_messages
        self.summarize_threshold = summarize_threshold
        self.summary: Optional[str] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message and summarize if needed"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        
        # Check if we need to summarize
        if len(self.messages) >= self.summarize_threshold:
            self._create_summary()
    
    def _create_summary(self) -> None:
        """Create summary of old messages (placeholder for now)"""
        # In a real implementation, this would use an LLM to create a summary
        # For now, we'll create a simple summary
        
        messages_to_summarize = self.messages[:-self.max_messages]
        if not messages_to_summarize:
            return
        
        # Create simple summary
        user_messages = [msg.content for msg in messages_to_summarize if msg.role == "user"]
        assistant_messages = [msg.content for msg in messages_to_summarize if msg.role == "assistant"]
        
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User discussed: {', '.join(user_messages[:3])}")
        if assistant_messages:
            summary_parts.append(f"Assistant provided information about: {', '.join(assistant_messages[:3])}")
        
        if self.summary:
            self.summary += f" {'; '.join(summary_parts)}"
        else:
            self.summary = "; ".join(summary_parts)
        
        # Keep only recent messages
        self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Message]:
        """Return messages with summary if available"""
        messages = []
        
        # Add summary as a system message if available
        if self.summary:
            summary_msg = Message(
                role="system",
                content=f"Previous conversation summary: {self.summary}",
                metadata={"is_summary": True}
            )
            messages.append(summary_msg)
        
        # Add current messages
        messages.extend(self.messages)
        
        return messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize with summary"""
        data = super().to_dict()
        data["summary"] = self.summary
        data["max_messages"] = self.max_messages
        data["summarize_threshold"] = self.summarize_threshold
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummaryMemory':
        """Deserialize with summary"""
        memory = cls(
            system_prompt=data.get("system_prompt", ""),
            max_messages=data.get("max_messages", 8),
            summarize_threshold=data.get("summarize_threshold", 12)
        )
        memory.messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        memory.summary = data.get("summary")
        return memory


class ConversationMemory:
    """Main conversation memory manager with multiple strategies"""
    
    def __init__(self, 
                 memory_type: str = "window",
                 system_prompt: str = "",
                 **kwargs):
        """
        Initialize conversation memory
        
        Args:
            memory_type: Type of memory ('buffer', 'window', 'token', 'summary')
            system_prompt: System prompt for the conversation
            **kwargs: Additional arguments for specific memory types
        """
        self.memory_type = memory_type
        
        # Create appropriate memory instance
        if memory_type == "buffer":
            self.memory = BufferMemory(system_prompt)
        elif memory_type == "window":
            window_size = kwargs.get("window_size", 10)
            self.memory = WindowMemory(system_prompt, window_size)
        elif memory_type == "token":
            max_tokens = kwargs.get("max_tokens", 2000)
            tokens_per_message = kwargs.get("tokens_per_message", 50)
            self.memory = TokenBasedMemory(system_prompt, max_tokens, tokens_per_message)
        elif memory_type == "summary":
            max_messages = kwargs.get("max_messages", 8)
            summarize_threshold = kwargs.get("summarize_threshold", 12)
            self.memory = SummaryMemory(system_prompt, max_messages, summarize_threshold)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message"""
        self.memory.add_user_message(content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an assistant message"""
        self.memory.add_assistant_message(content, metadata)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption"""
        messages = []
        
        # Add system prompt if available
        if self.memory.system_prompt:
            messages.append({"role": "system", "content": self.memory.system_prompt})
        
        # Add conversation messages
        for msg in self.memory.get_messages():
            if msg.role == "system" and not self.memory.system_prompt:
                # Include system messages (like summaries) if no global system prompt
                messages.append({"role": msg.role, "content": msg.content})
            elif msg.role in ["user", "assistant"]:
                messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def get_conversation_length(self) -> int:
        """Get number of messages in conversation"""
        return self.memory.get_conversation_length()
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.memory.clear()
    
    def save_to_file(self, filepath: str) -> None:
        """Save conversation to file"""
        data = self.memory.to_dict()
        data["memory_type"] = self.memory_type
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """Load conversation from file"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        memory_type = data.get("memory_type", "window")
        
        # Reconstruct memory based on type
        if memory_type == "buffer":
            self.memory = BufferMemory.from_dict(data)
        elif memory_type == "window":
            self.memory = WindowMemory.from_dict(data)
        elif memory_type == "token":
            self.memory = TokenBasedMemory.from_dict(data)
        elif memory_type == "summary":
            self.memory = SummaryMemory.from_dict(data)
        
        self.memory_type = memory_type
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message content"""
        for msg in reversed(self.memory.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message content"""
        for msg in reversed(self.memory.messages):
            if msg.role == "assistant":
                return msg.content
        return None