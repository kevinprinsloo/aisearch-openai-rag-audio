"""
Text-to-Speech Service Module

Provides a unified interface for different TTS providers
with configuration management and voice switching capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from gtts import gTTS
import pygame
import tempfile
import os


@dataclass
class TTSConfig:
    """Configuration for TTS services"""
    language: str = "en"
    speed: float = 1.2
    tld: str = "com"  # Top-level domain for GTTS accents
    chunk_length: int = 50
    crossfade_length: int = 10
    
    def validate(self) -> None:
        """Validate configuration values"""
        if self.speed <= 0:
            raise ValueError("Speed must be positive")
        if self.chunk_length <= 0:
            raise ValueError("Chunk length must be positive")
        if self.crossfade_length < 0:
            raise ValueError("Crossfade length must be non-negative")


class BaseTTSService(ABC):
    """Abstract base class for TTS services"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
    
    @abstractmethod
    def speak(self, text: str, blocking: bool = True) -> None:
        """Speak the given text"""
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        """Stop current speech"""
        raise NotImplementedError
    
    @abstractmethod
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        raise NotImplementedError
    
    @abstractmethod
    def update_config(self, new_config: TTSConfig) -> None:
        """Update TTS configuration"""
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        raise NotImplementedError


class GTTSTTSService(BaseTTSService):
    """Google Text-to-Speech service implementation using basic gtts"""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
    
    def speak(self, text: str, blocking: bool = True) -> None:
        """Speak the given text"""
        if not text.strip():
            return
            
        try:
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=self.config.language,
                tld=self.config.tld,
                slow=False
            )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Save to temporary file
            tts.save(temp_filename)
            
            # Play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            if blocking:
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except OSError:
                pass  # Ignore cleanup errors
                
        except Exception as e:
            print(f"⚠ TTS Error: {e}")
    
    def stop(self) -> None:
        """Stop current speech"""
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        try:
            return pygame.mixer.music.get_busy()
        except pygame.error:
            return False
    
    def update_config(self, new_config: TTSConfig) -> None:
        """Update TTS configuration"""
        new_config.validate()
        self.config = new_config
        # No engine reinitialization needed for basic gtts
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            pygame.mixer.quit()
        except pygame.error:
            pass


class TTSService:
    """
    Main TTS service interface that manages different TTS providers
    """
    
    def __init__(self, 
                 config: TTSConfig = None,
                 provider: str = "gtts"):
        """
        Initialize TTS service
        
        Args:
            config: TTS configuration
            provider: TTS provider ('gtts' for now, extensible for future providers)
        """
        self.config = config or TTSConfig()
        self.provider = provider
        self.service = None
        
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize the appropriate TTS service"""
        if self.provider == "gtts":
            self.service = GTTSTTSService(self.config)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")
    
    def speak(self, text: str, blocking: bool = True) -> None:
        """Speak text using the configured TTS service"""
        if self.service:
            self.service.speak(text, blocking)
    
    def stop(self) -> None:
        """Stop current speech"""
        if self.service:
            self.service.stop()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        if self.service:
            return self.service.is_speaking()
        return False
    
    def update_config(self, new_config: TTSConfig) -> None:
        """Update TTS configuration"""
        if self.service:
            self.service.update_config(new_config)
        self.config = new_config
    
    def cleanup(self) -> None:
        """Clean up TTS service resources"""
        if self.service:
            self.service.cleanup()


# Convenience function for backward compatibility
def create_tts_service(config: TTSConfig = None) -> TTSService:
    """Create a TTS service with default configuration"""
    return TTSService(config or TTSConfig())