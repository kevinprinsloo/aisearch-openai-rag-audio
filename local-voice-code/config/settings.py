"""
Configuration Management for Chatbot

Centralized configuration handling with environment variable support,
validation, and easy customization.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM services"""
    provider: str = "qwen"  # qwen, openai
    model_name: Optional[str] = None  # Uses provider default if None
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    device: Optional[str] = None  # Auto-detect if None
    
    # API keys for external providers
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set default model names based on provider
        if self.model_name is None:
            if self.provider == "qwen":
                self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            elif self.provider == "openai":
                self.model_name = "gpt-3.5-turbo"
        
        # Load API keys from environment
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.provider not in ["qwen", "openai"]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("Top_p must be between 0.0 and 1.0")
        
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        return True


@dataclass
class TTSConfig:
    """Configuration for TTS services"""
    provider: str = "gtts"  # gtts for now, extensible for future providers
    language: str = "en"
    speed: float = 1.2
    tld: str = "com"  # Top-level domain for GTTS accents
    chunk_length: int = 100
    crossfade_length: int = 10
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.language = os.getenv("GTTS_LANGUAGE", self.language)
        self.speed = float(os.getenv("GTTS_SPEED", str(self.speed)))
        self.tld = os.getenv("GTTS_TLD", self.tld)
        self.chunk_length = int(os.getenv("GTTS_CHUNK_LENGTH", str(self.chunk_length)))
        self.crossfade_length = int(os.getenv("GTTS_CROSSFADE", str(self.crossfade_length)))
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError("Speed must be between 0.5 and 2.0")
        
        valid_tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]
        if self.tld not in valid_tlds:
            raise ValueError(f"TLD must be one of: {valid_tlds}")
        
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        
        if self.crossfade_length < 0:
            raise ValueError("crossfade_length must be non-negative")
        
        return True


@dataclass
class MemoryConfig:
    """Configuration for conversation memory"""
    memory_type: str = "window"  # buffer, window, token, summary
    
    # Window memory settings
    window_size: int = 10
    
    # Token-based memory settings
    max_tokens: int = 2000
    tokens_per_message: int = 50
    
    # Summary memory settings
    max_messages: int = 8
    summarize_threshold: int = 12
    
    # System prompt
    system_prompt: str = (
        "You are a helpful AI voice assistant. Keep your responses conversational, "
        "concise, and engaging since they will be spoken aloud. Avoid overly long "
        "responses unless specifically asked for detailed explanations. "
        "Be friendly, helpful, and informative."
    )
    
    def validate(self) -> bool:
        """Validate configuration"""
        valid_types = ["buffer", "window", "token", "summary"]
        if self.memory_type not in valid_types:
            raise ValueError(f"memory_type must be one of: {valid_types}")
        
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.max_messages <= 0:
            raise ValueError("max_messages must be positive")
        
        if self.summarize_threshold <= self.max_messages:
            raise ValueError("summarize_threshold must be greater than max_messages")
        
        return True


@dataclass
class SpeechConfig:
    """Configuration for speech recognition services"""
    model_name: str = "openai/whisper-large-v3"
    device: Optional[str] = None  # Auto-detect if None
    sample_rate: int = 16000
    chunk_duration: float = 1.0
    silence_threshold: float = 0.01
    silence_duration: float = 2.0
    language: Optional[str] = "english"
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.model_name = os.getenv("SPEECH_MODEL_NAME", self.model_name)
        if os.getenv("SPEECH_DEVICE"):
            self.device = os.getenv("SPEECH_DEVICE")
        self.sample_rate = int(os.getenv("SPEECH_SAMPLE_RATE", str(self.sample_rate)))
        self.silence_threshold = float(os.getenv("SPEECH_SILENCE_THRESHOLD", str(self.silence_threshold)))
        self.silence_duration = float(os.getenv("SPEECH_SILENCE_DURATION", str(self.silence_duration)))
        if os.getenv("SPEECH_LANGUAGE"):
            self.language = os.getenv("SPEECH_LANGUAGE")
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        
        if not (0.0 <= self.silence_threshold <= 1.0):
            raise ValueError("silence_threshold must be between 0.0 and 1.0")
        
        if self.silence_duration <= 0:
            raise ValueError("silence_duration must be positive")
        
        return True


@dataclass
class ChatbotConfig:
    """Main configuration for the chatbot"""
    
    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    
    # Session management
    session_file: Optional[str] = None  # Save/load conversations
    auto_save: bool = True
    
    # Logging and output
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    show_thinking: bool = True  # Show "🤔 Thinking..." message
    show_speaking: bool = True  # Show "🔊 Speaking..." message
    
    # Voice commands and shortcuts
    enable_voice_commands: bool = True
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Load session file from environment
        if self.session_file is None:
            self.session_file = os.getenv("CHATBOT_SESSION_FILE")
    
    def validate(self) -> bool:
        """Validate all configurations"""
        self.llm.validate()
        self.tts.validate()
        self.memory.validate()
        self.speech.validate()
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "max_new_tokens": self.llm.max_new_tokens,
                "temperature": self.llm.temperature,
                "do_sample": self.llm.do_sample,
                "top_p": self.llm.top_p,
                "device": self.llm.device,
            },
            "tts": {
                "provider": self.tts.provider,
                "language": self.tts.language,
                "speed": self.tts.speed,
                "tld": self.tts.tld,
                "chunk_length": self.tts.chunk_length,
                "crossfade_length": self.tts.crossfade_length,
            },
            "memory": {
                "memory_type": self.memory.memory_type,
                "window_size": self.memory.window_size,
                "max_tokens": self.memory.max_tokens,
                "tokens_per_message": self.memory.tokens_per_message,
                "max_messages": self.memory.max_messages,
                "summarize_threshold": self.memory.summarize_threshold,
                "system_prompt": self.memory.system_prompt,
            },
            "session_file": self.session_file,
            "auto_save": self.auto_save,
            "log_level": self.log_level,
            "show_thinking": self.show_thinking,
            "show_speaking": self.show_speaking,
            "enable_voice_commands": self.enable_voice_commands,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatbotConfig':
        """Create configuration from dictionary"""
        # Extract component configs
        llm_config = LLMConfig(**data.get("llm", {}))
        tts_config = TTSConfig(**data.get("tts", {}))
        memory_config = MemoryConfig(**data.get("memory", {}))
        
        # Create main config
        config = cls(
            llm=llm_config,
            tts=tts_config,
            memory=memory_config,
            session_file=data.get("session_file"),
            auto_save=data.get("auto_save", True),
            log_level=data.get("log_level", "INFO"),
            show_thinking=data.get("show_thinking", True),
            show_speaking=data.get("show_speaking", True),
            enable_voice_commands=data.get("enable_voice_commands", True),
        )
        
        return config
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        # LLM settings
        if os.getenv("LLM_PROVIDER"):
            self.llm.provider = os.getenv("LLM_PROVIDER", self.llm.provider)
        if os.getenv("LLM_MODEL_NAME"):
            self.llm.model_name = os.getenv("LLM_MODEL_NAME")
        if os.getenv("LLM_MAX_TOKENS"):
            self.llm.max_new_tokens = int(os.getenv("LLM_MAX_TOKENS", str(self.llm.max_new_tokens)))
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE", str(self.llm.temperature)))
        
        # Memory settings
        if os.getenv("MEMORY_TYPE"):
            self.memory.memory_type = os.getenv("MEMORY_TYPE", self.memory.memory_type)
        if os.getenv("MEMORY_WINDOW_SIZE"):
            self.memory.window_size = int(os.getenv("MEMORY_WINDOW_SIZE", str(self.memory.window_size)))
        if os.getenv("SYSTEM_PROMPT"):
            self.memory.system_prompt = os.getenv("SYSTEM_PROMPT", self.memory.system_prompt)
        
        # Session settings
        if os.getenv("AUTO_SAVE"):
            self.auto_save = os.getenv("AUTO_SAVE", "true").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            self.log_level = os.getenv("LOG_LEVEL", self.log_level)


# Default configuration instance
default_config = ChatbotConfig()


def get_default_config() -> ChatbotConfig:
    """Get default configuration"""
    return ChatbotConfig()


def create_config_from_env() -> ChatbotConfig:
    """Create configuration with environment variable overrides"""
    config = ChatbotConfig()
    config.update_from_env()
    config.validate()
    return config