"""
Session Persistence Utilities

Handles saving and loading conversation sessions with support for
multiple formats and metadata tracking.
"""

import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

from memory.conversation_memory import ConversationMemory, Message, SummaryMemory


class SessionManager:
    """Manages session persistence with multiple format support"""
    
    def __init__(self, default_format: str = "json"):
        """
        Initialize session manager
        
        Args:
            default_format: Default file format ('json' or 'pickle')
        """
        self.default_format = default_format.lower()
        if self.default_format not in ["json", "pickle"]:
            raise ValueError("Format must be 'json' or 'pickle'")
    
    def save_session(self, 
                    memory: ConversationMemory,
                    filepath: Union[str, Path],
                    metadata: Optional[Dict[str, Any]] = None,
                    format_type: Optional[str] = None) -> bool:
        """
        Save conversation session to file
        
        Args:
            memory: ConversationMemory instance to save
            filepath: Path to save the session
            metadata: Additional metadata to include
            format_type: File format override ('json' or 'pickle')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            format_type = format_type or self._detect_format(filepath) or self.default_format
            
            # Prepare session data
            session_data = self._prepare_session_data(memory, metadata)
            
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            if format_type == "json":
                return self._save_json(session_data, filepath)
            elif format_type == "pickle":
                return self._save_pickle(session_data, filepath)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self,
                    filepath: Union[str, Path],
                    format_type: Optional[str] = None) -> Optional[ConversationMemory]:
        """
        Load conversation session from file
        
        Args:
            filepath: Path to load the session from
            format_type: File format override ('json' or 'pickle')
            
        Returns:
            ConversationMemory instance or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                print(f"Session file not found: {filepath}")
                return None
            
            format_type = format_type or self._detect_format(filepath) or self.default_format
            
            # Load based on format
            if format_type == "json":
                session_data = self._load_json(filepath)
            elif format_type == "pickle":
                session_data = self._load_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            if session_data is None:
                return None
            
            # Reconstruct memory from session data
            return self._reconstruct_memory(session_data)
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def list_sessions(self, directory: Union[str, Path]) -> list:
        """
        List available session files in directory
        
        Args:
            directory: Directory to search for session files
            
        Returns:
            List of session file information
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        sessions = []
        
        # Look for json and pickle files
        patterns = ["*.json", "*.pkl", "*.pickle"]
        
        for pattern in patterns:
            for filepath in directory.glob(pattern):
                try:
                    # Get file info
                    stat = filepath.stat()
                    
                    # Try to load metadata
                    metadata = self._get_session_metadata(filepath)
                    
                    session_info = {
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "format": self._detect_format(filepath),
                        "metadata": metadata
                    }
                    
                    sessions.append(session_info)
                    
                except Exception as e:
                    print(f"Error reading session file {filepath}: {e}")
                    continue
        
        # Sort by modification time (newest first)
        sessions.sort(key=lambda x: x["modified"], reverse=True)
        
        return sessions
    
    def _prepare_session_data(self, 
                             memory: ConversationMemory, 
                             metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare session data for saving"""
        session_data = {
            "version": "2.0.0",
            "created": datetime.now().isoformat(),
            "memory_data": memory.memory.to_dict(),
            "metadata": metadata or {}
        }
        
        # Add some basic statistics
        session_data["stats"] = {
            "message_count": memory.get_conversation_length(),
            "memory_type": memory.memory_type,
            "last_user_message": memory.get_last_user_message(),
            "last_assistant_message": memory.get_last_assistant_message()
        }
        
        return session_data
    
    def _reconstruct_memory(self, session_data: Dict[str, Any]) -> ConversationMemory:
        """Reconstruct ConversationMemory from session data"""
        memory_data = session_data.get("memory_data", {})
        memory_type = memory_data.get("type", "window")
        
        # Map class names to memory types
        type_mapping = {
            "BufferMemory": "buffer",
            "WindowMemory": "window", 
            "TokenBasedMemory": "token",
            "SummaryMemory": "summary"
        }
        
        if memory_type in type_mapping:
            memory_type = type_mapping[memory_type]
        
        # Create appropriate memory instance
        kwargs = {}
        if memory_type == "window":
            kwargs["window_size"] = memory_data.get("window_size", 10)
        elif memory_type == "token":
            kwargs["max_tokens"] = memory_data.get("max_tokens", 2000)
            kwargs["tokens_per_message"] = memory_data.get("tokens_per_message", 50)
        elif memory_type == "summary":
            kwargs["max_messages"] = memory_data.get("max_messages", 8)
            kwargs["summarize_threshold"] = memory_data.get("summarize_threshold", 12)
        
        # Create memory instance
        memory = ConversationMemory(
            memory_type=memory_type,
            system_prompt=memory_data.get("system_prompt", ""),
            **kwargs
        )
        
        # Restore messages
        messages_data = memory_data.get("messages", [])
        for msg_data in messages_data:
            message = Message.from_dict(msg_data)
            memory.memory.messages.append(message)
        
        # Restore summary if available (for summary memory)
        if memory_type == "summary" and "summary" in memory_data:
            # Only assign summary for SummaryMemory instances
            if hasattr(memory.memory, 'summary'):
                memory.memory.summary = memory_data["summary"]
        
        return memory
    
    def _save_json(self, data: Dict[str, Any], filepath: Path) -> bool:
        """Save data as JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    
    def _save_pickle(self, data: Dict[str, Any], filepath: Path) -> bool:
        """Save data as pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    
    def _load_json(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load data from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_pickle(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load data from pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _detect_format(self, filepath: Path) -> Optional[str]:
        """Detect file format from extension"""
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix in [".pkl", ".pickle"]:
            return "pickle"
        return None
    
    def _get_session_metadata(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Get metadata from session file"""
        try:
            format_type = self._detect_format(filepath)
            
            if format_type == "json":
                data = self._load_json(filepath)
            elif format_type == "pickle":
                data = self._load_pickle(filepath)
            else:
                return None
            
            if data and isinstance(data, dict):
                return {
                    "version": data.get("version"),
                    "created": data.get("created"),
                    "stats": data.get("stats", {}),
                    "user_metadata": data.get("metadata", {})
                }
                
        except Exception:
            pass
        
        return None


class AutoSessionManager:
    """Automatic session management with configurable intervals"""
    
    def __init__(self, 
                 session_manager: SessionManager,
                 auto_save_interval: int = 300):  # 5 minutes
        """
        Initialize auto session manager
        
        Args:
            session_manager: SessionManager instance
            auto_save_interval: Auto-save interval in seconds
        """
        self.session_manager = session_manager
        self.auto_save_interval = auto_save_interval
        self.last_save_time = datetime.now()
        self.session_file: Optional[str] = None
    
    def set_session_file(self, filepath: str) -> None:
        """Set the session file for auto-saving"""
        self.session_file = filepath
    
    def should_auto_save(self) -> bool:
        """Check if auto-save should trigger"""
        if not self.session_file:
            return False
        
        now = datetime.now()
        elapsed = (now - self.last_save_time).total_seconds()
        return elapsed >= self.auto_save_interval
    
    def auto_save(self, memory: ConversationMemory) -> bool:
        """Perform auto-save if conditions are met"""
        if not self.should_auto_save():
            return False
        
        if self.session_file and self.session_manager.save_session(memory, self.session_file):
            self.last_save_time = datetime.now()
            return True
        
        return False
    
    def force_save(self, memory: ConversationMemory) -> bool:
        """Force save regardless of interval"""
        if self.session_file:
            result = self.session_manager.save_session(memory, self.session_file)
            if result:
                self.last_save_time = datetime.now()
            return result
        return False


# Convenience functions
def save_conversation(memory: ConversationMemory, 
                     filepath: str,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to save a conversation
    
    Args:
        memory: ConversationMemory to save
        filepath: File path to save to
        metadata: Optional metadata
        
    Returns:
        bool: Success status
    """
    manager = SessionManager()
    return manager.save_session(memory, filepath, metadata)


def load_conversation(filepath: str) -> Optional[ConversationMemory]:
    """
    Convenience function to load a conversation
    
    Args:
        filepath: File path to load from
        
    Returns:
        ConversationMemory or None
    """
    manager = SessionManager()
    return manager.load_session(filepath)


def list_saved_conversations(directory: str = ".") -> list:
    """
    Convenience function to list saved conversations
    
    Args:
        directory: Directory to search
        
    Returns:
        List of session information
    """
    manager = SessionManager()
    return manager.list_sessions(directory)