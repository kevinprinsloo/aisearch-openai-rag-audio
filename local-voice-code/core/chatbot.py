"""
Main Chatbot Class

Orchestrates all components: memory, LLM, TTS, and command handling.
Designed for easy extensibility with RAG and other features.
"""

import os
import time
import platform
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.settings import ChatbotConfig, create_config_from_env
from memory.conversation_memory import ConversationMemory
from services.llm_service import LLMService
from services.tts_service import TTSService, TTSConfig
from services.speech_service import WhisperSpeechService


@dataclass
class CommandResult:
    """Result of a command execution"""
    success: bool
    message: str
    should_continue: bool = True


class Chatbot:
    """Main chatbot class that orchestrates all components"""
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """
        Initialize the chatbot
        
        Args:
            config: Configuration object (creates default if None)
        """
        # Configuration
        self.config = config or create_config_from_env()
        self.config.validate()
        
        # Components (initialized lazily)
        self.memory: Optional[ConversationMemory] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.speech_service: Optional[WhisperSpeechService] = None
        
        # Voice input mode
        self.voice_input_enabled = getattr(config, 'enable_voice_commands', False) if config else False
        
        # State
        self.is_initialized = False
        self.session_active = False
        
        # Command handlers
        self.command_handlers: Dict[str, Callable[[str], CommandResult]] = {
            "exit": self._handle_exit,
            "quit": self._handle_exit,
            "speed": self._handle_speed_change,
            "accent": self._handle_accent_change,
            "lang": self._handle_language_change,
            "voices": self._handle_show_voices,
            "system": self._handle_show_system_info,
            "memory": self._handle_memory_info,
            "config": self._handle_show_config,
            "clear": self._handle_clear_memory,
            "save": self._handle_save_session,
            "load": self._handle_load_session,
            "help": self._handle_help,
            "voice": self._handle_voice_toggle,
            "mic": self._handle_mic_test,
        }
        
        print("Chatbot initialized with modular architecture")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        if self.is_initialized:
            return True
        
        try:
            # Initialize memory
            self._initialize_memory()
            
            # Initialize LLM service
            self._initialize_llm()
            
            # Initialize TTS service
            self._initialize_tts()
            
            # Initialize Speech service (if voice input enabled)
            if self.voice_input_enabled:
                self._initialize_speech()
            
            # Load session if specified
            if self.config.session_file and self.config.auto_save:
                self._load_session()
            
            self.is_initialized = True
            print("✓ All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize chatbot: {e}")
            return False
    
    def _initialize_memory(self) -> None:
        """Initialize conversation memory"""
        memory_config = self.config.memory
        
        self.memory = ConversationMemory(
            memory_type=memory_config.memory_type,
            system_prompt=memory_config.system_prompt,
            window_size=memory_config.window_size,
            max_tokens=memory_config.max_tokens,
            tokens_per_message=memory_config.tokens_per_message,
            max_messages=memory_config.max_messages,
            summarize_threshold=memory_config.summarize_threshold
        )
        
        print(f"✓ Memory initialized - Type: {memory_config.memory_type}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM service"""
        llm_config = self.config.llm
        
        self.llm_service = LLMService(
            provider=llm_config.provider,
            model_name=llm_config.model_name,
            max_new_tokens=llm_config.max_new_tokens,
            temperature=llm_config.temperature,
            do_sample=llm_config.do_sample,
            top_p=llm_config.top_p,
            device=llm_config.device,
            api_key=llm_config.openai_api_key
        )
        
        # Load model if needed
        if not self.llm_service.load_model():
            raise RuntimeError("Failed to load LLM model")
        
        print(f"✓ LLM initialized - Provider: {llm_config.provider}, Model: {llm_config.model_name}")
    
    def _initialize_tts(self) -> None:
        """Initialize TTS service"""
        tts_config_obj = TTSConfig(
            language=self.config.tts.language,
            speed=self.config.tts.speed,
            tld=self.config.tts.tld,
            chunk_length=self.config.tts.chunk_length,
            crossfade_length=self.config.tts.crossfade_length
        )
        
        self.tts_service = TTSService(
            provider=self.config.tts.provider,
            config=tts_config_obj
        )
        
        print(f"✓ TTS initialized - Provider: {self.config.tts.provider}")
    
    def _initialize_speech(self) -> None:
        """Initialize Speech recognition service"""
        try:
            # Use configuration values if available, otherwise defaults
            speech_config = getattr(self.config, 'speech', None)
            
            # Use speech config if available
            if speech_config:
                self.speech_service = WhisperSpeechService(
                    model_name=speech_config.model_name,
                    device=speech_config.device,
                    sample_rate=speech_config.sample_rate,
                    chunk_duration=speech_config.chunk_duration,
                    silence_threshold=speech_config.silence_threshold,
                    silence_duration=speech_config.silence_duration,
                    language=speech_config.language
                )
            else:
                # Default configuration
                self.speech_service = WhisperSpeechService()
            
            # Load model
            if not self.speech_service.load_model():
                raise RuntimeError("Failed to load speech recognition model")
            
            print(f"✓ Speech recognition initialized - Model: {self.speech_service.model_name}")
            
        except Exception as e:
            print(f"⚠ Speech recognition initialization failed: {e}")
            print("Voice input will be disabled.")
            self.speech_service = None
            self.voice_input_enabled = False
    
    def start_conversation(self) -> None:
        """Start the main conversation loop"""
        if not self.is_initialized:
            if not self.initialize():
                return
        
        self.session_active = True
        
        print("\n" + "="*60)
        print("AI VOICE ASSISTANT READY!")
        print("="*60)
        self._show_welcome_message()
        
        # Main conversation loop
        while self.session_active:
            try:
                user_input = self._get_user_input()
                
                if not user_input:
                    continue
                
                # Process user input
                self._process_user_input(user_input)
                
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        self._cleanup()
    
    def _process_user_input(self, user_input: str) -> None:
        """Process user input (command or conversation)"""
        # Check if it's a command
        if self._is_command(user_input):
            result = self._handle_command(user_input)
            if not result.should_continue:
                self.session_active = False
            return
        
        # Handle conversation
        self._handle_conversation(user_input)
    
    def _is_command(self, user_input: str) -> bool:
        """Check if input is a command"""
        if not self.config.enable_voice_commands:
            return False
        
        # Commands start with keywords or have specific patterns
        words = user_input.lower().split()
        if not words:
            return False
        
        # Check for direct command matches
        if words[0] in self.command_handlers:
            return True
        
        # Check for command patterns
        command_patterns = [
            ("speed", lambda w: len(w) >= 2 and w[0] == "speed"),
            ("accent", lambda w: len(w) >= 2 and w[0] == "accent"),
            ("lang", lambda w: len(w) >= 2 and w[0] == "lang"),
        ]
        
        for cmd, pattern in command_patterns:
            if pattern(words):
                return True
        
        return False
    
    def _handle_command(self, user_input: str) -> CommandResult:
        """Handle command input"""
        words = user_input.lower().split()
        command = words[0]
        
        if command in self.command_handlers:
            return self.command_handlers[command](user_input)
        
        return CommandResult(False, f"Unknown command: {command}")
    
    def _handle_conversation(self, user_input: str) -> None:
        """Handle conversational input"""
        if not self.memory or not self.llm_service or not self.tts_service:
            print("Error: Components not initialized")
            return
        
        try:
            # Add user message to memory
            self.memory.add_user_message(user_input)
            
            # Show thinking indicator
            if self.config.show_thinking:
                print("🤔 Thinking...")
            
            # Generate response
            response = self.llm_service.generate_response(self.memory)
            
            # Add assistant response to memory
            self.memory.add_assistant_message(response)
            
            # Display response
            print(f"Assistant: {response}")
            
            # Convert to speech
            if self.config.show_speaking:
                print("🔊 Speaking...")
            
            self.tts_service.speak(response, blocking=True)
            
            # Auto-save session if enabled
            if self.config.auto_save and self.config.session_file:
                self._save_session()
                
        except Exception as e:
            print(f"Error processing conversation: {e}")
    
    # Command handlers
    def _handle_exit(self, command: str) -> CommandResult:
        """Handle exit/quit command"""
        print("Assistant: Goodbye! Have a great day!")
        return CommandResult(True, "Exiting...", should_continue=False)
    
    def _handle_speed_change(self, command: str) -> CommandResult:
        """Handle speed change command"""
        try:
            parts = command.split()
            if len(parts) < 2:
                return CommandResult(False, "Usage: speed <value> (e.g., speed 1.5)")
            
            new_speed = float(parts[1])
            if not (0.5 <= new_speed <= 2.0):
                return CommandResult(False, "Speed must be between 0.5 and 2.0")
            
            if self.tts_service:
                self.tts_service.update_voice_setting("speed", new_speed)
                self.config.tts.speed = new_speed
                return CommandResult(True, f"✓ Voice speed changed to {new_speed}x")
            
            return CommandResult(False, "TTS service not available")
            
        except (ValueError, IndexError):
            return CommandResult(False, "Invalid speed format. Use: speed 1.5")
    
    def _handle_accent_change(self, command: str) -> CommandResult:
        """Handle accent change command"""
        try:
            parts = command.split()
            if len(parts) < 2:
                return CommandResult(False, "Usage: accent <tld> (e.g., accent co.uk)")
            
            new_tld = parts[1]
            valid_tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]
            
            if new_tld not in valid_tlds:
                return CommandResult(False, f"Invalid accent. Valid options: {', '.join(valid_tlds)}")
            
            if self.tts_service:
                self.tts_service.update_voice_setting("tld", new_tld)
                self.config.tts.tld = new_tld
                return CommandResult(True, f"✓ Accent changed to {new_tld}")
            
            return CommandResult(False, "TTS service not available")
            
        except IndexError:
            return CommandResult(False, "Invalid accent format. Use: accent co.uk")
    
    def _handle_language_change(self, command: str) -> CommandResult:
        """Handle language change command"""
        try:
            parts = command.split()
            if len(parts) < 2:
                return CommandResult(False, "Usage: lang <code> (e.g., lang es)")
            
            new_lang = parts[1]
            
            if self.tts_service:
                self.tts_service.update_voice_setting("language", new_lang)
                self.config.tts.language = new_lang
                return CommandResult(True, f"✓ Language changed to {new_lang}")
            
            return CommandResult(False, "TTS service not available")
            
        except IndexError:
            return CommandResult(False, "Invalid language format. Use: lang es")
    
    def _handle_show_voices(self, command: str) -> CommandResult:
        """Handle show voices command"""
        voice_options = TTSService.get_voice_options()
        
        print("\n=== Voice Configuration Options ===")
        print(f"Languages: {', '.join(voice_options['languages'][:10])}... (and more)")
        print("TLD Accents:")
        for tld, desc in voice_options['tld_accents'].items():
            print(f"  {tld:<8} - {desc}")
        print(f"Speed: {voice_options['speed_range']['min']} to {voice_options['speed_range']['max']} (default: {voice_options['speed_range']['default']})")
        print("=====================================\n")
        
        return CommandResult(True, "Voice options displayed")
    
    def _handle_show_system_info(self, command: str) -> CommandResult:
        """Handle show system info command"""
        print(f"\n=== System Information ===")
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA: Available (GPU: {torch.cuda.get_device_name()})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS: Available (Apple Silicon GPU)")
        else:
            print("GPU: Not available, using CPU")
        
        if self.llm_service:
            llm_info = self.llm_service.get_provider_info()
            print(f"LLM: {llm_info['provider']} - {llm_info['model_name']}")
        
        if self.tts_service:
            tts_info = self.tts_service.get_provider_info()
            print(f"TTS: {tts_info['provider']}")
        
        print("===========================\n")
        
        return CommandResult(True, "System information displayed")
    
    def _handle_memory_info(self, command: str) -> CommandResult:
        """Handle memory info command"""
        if not self.memory:
            return CommandResult(False, "Memory not initialized")
        
        print(f"\n=== Memory Information ===")
        print(f"Type: {self.memory.memory_type}")
        print(f"Messages: {self.memory.get_conversation_length()}")
        
        if hasattr(self.memory.memory, 'summary') and self.memory.memory.summary:
            print(f"Summary: {self.memory.memory.summary[:100]}...")
        
        last_user = self.memory.get_last_user_message()
        last_assistant = self.memory.get_last_assistant_message()
        
        if last_user:
            print(f"Last user: {last_user[:50]}...")
        if last_assistant:
            print(f"Last assistant: {last_assistant[:50]}...")
        
        print("==========================\n")
        
        return CommandResult(True, "Memory information displayed")
    
    def _handle_show_config(self, command: str) -> CommandResult:
        """Handle show config command"""
        config_dict = self.config.to_dict()
        
        print("\n=== Configuration ===")
        for section, settings in config_dict.items():
            if isinstance(settings, dict):
                print(f"{section.upper()}:")
                for key, value in settings.items():
                    if "key" in key.lower() and value:
                        value = "***"  # Hide API keys
                    print(f"  {key}: {value}")
            else:
                print(f"{section}: {settings}")
        print("=====================\n")
        
        return CommandResult(True, "Configuration displayed")
    
    def _handle_clear_memory(self, command: str) -> CommandResult:
        """Handle clear memory command"""
        if not self.memory:
            return CommandResult(False, "Memory not initialized")
        
        self.memory.clear()
        return CommandResult(True, "✓ Memory cleared")
    
    def _handle_save_session(self, command: str) -> CommandResult:
        """Handle save session command"""
        try:
            parts = command.split()
            if len(parts) > 1:
                filename = parts[1]
            else:
                filename = self.config.session_file or "chatbot_session.json"
            
            if self.memory:
                self.memory.save_to_file(filename)
                return CommandResult(True, f"✓ Session saved to {filename}")
            
            return CommandResult(False, "Memory not initialized")
            
        except Exception as e:
            return CommandResult(False, f"Failed to save session: {e}")
    
    def _handle_load_session(self, command: str) -> CommandResult:
        """Handle load session command"""
        try:
            parts = command.split()
            if len(parts) > 1:
                filename = parts[1]
            else:
                filename = self.config.session_file or "chatbot_session.json"
            
            if self.memory:
                self.memory.load_from_file(filename)
                return CommandResult(True, f"✓ Session loaded from {filename}")
            
            return CommandResult(False, "Memory not initialized")
            
        except Exception as e:
            return CommandResult(False, f"Failed to load session: {e}")
    
    def _handle_help(self, command: str) -> CommandResult:
        """Handle help command"""
        print("\n=== Available Commands ===")
        print("Conversation:")
        print("  Just type naturally to chat with the AI")
        print("\nVoice Settings:")
        print("  speed X     - Change voice speed (0.5-2.0)")
        print("  accent X    - Change accent (e.g., 'co.uk', 'com.au')")
        print("  lang X      - Change language (e.g., 'es', 'fr')")
        print("  voices      - Show available voice options")
        print("\nSession Management:")
        print("  save [file] - Save conversation to file")
        print("  load [file] - Load conversation from file")
        print("  clear       - Clear conversation memory")
        print("\nInformation:")
        print("  system      - Show system information")
        print("  memory      - Show memory status")
        print("  config      - Show configuration")
        print("  help        - Show this help")
        print("\nControl:")
        print("  exit/quit   - End session")
        print("============================\n")
        
        return CommandResult(True, "Help displayed")
    
    def _show_welcome_message(self) -> None:
        """Show welcome message"""
        print("Ask me any question and I'll respond with voice!")
        print("\nSpecial commands:")
        print("  Type 'help' for all available commands")
        print("  Type 'exit' or 'quit' to end the session")
        print("\nNote: Internet connection required for TTS synthesis.")
        print("="*60 + "\n")
    
    def _save_session(self) -> None:
        """Save current session"""
        if self.memory and self.config.session_file:
            try:
                self.memory.save_to_file(self.config.session_file)
            except Exception as e:
                print(f"Warning: Failed to auto-save session: {e}")
    
    def _load_session(self) -> None:
        """Load session from file"""
        if self.memory and self.config.session_file:
            try:
                self.memory.load_from_file(self.config.session_file)
                print(f"✓ Session loaded from {self.config.session_file}")
            except Exception as e:
                print(f"Note: Could not load previous session: {e}")
    
    def _get_user_input(self) -> str:
        """Get user input via text or voice"""
        if self.voice_input_enabled and self.speech_service:
            # Check if voice-only mode
            if getattr(self.config, 'voice_only_mode', False):
                return self._get_voice_input()
            
            # Natural voice assistant mode - always listen by default
            print("🎤 Listening... (speak now, or press Ctrl+C to switch to text mode)")
            
            try:
                return self._get_voice_input()
            except KeyboardInterrupt:
                # User interrupted - switch to text input
                print("\nSwitching to text input...")
                return input("You (text): ").strip()
        else:
            return input("You: ").strip()
    
    def _get_voice_input(self) -> str:
        """Get voice input using speech recognition"""
        if not self.speech_service:
            print("Voice input not available.")
            return ""
        
        try:
            print("🎤 Listening... (speak now, I'll detect when you're done)")
            
            # Use shorter silence duration for more responsive interaction
            transcription = self.speech_service.record_and_transcribe(
                use_silence_detection=True
            )
            
            if transcription and transcription.strip():
                print(f"You said: {transcription}")
                return transcription.strip()
            else:
                print("🔇 I didn't catch that. Let me try again...")
                # Give one more chance
                print("🎤 Please speak again...")
                transcription = self.speech_service.record_and_transcribe(
                    use_silence_detection=True
                )
                
                if transcription and transcription.strip():
                    print(f"You said: {transcription}")
                    return transcription.strip()
                else:
                    print("No speech detected. You can type your message instead:")
                    return input("You: ").strip()
                
        except Exception as e:
            print(f"Voice input error: {e}")
            print("Falling back to text input:")
            return input("You: ").strip()
    
    def _handle_voice_toggle(self, command: str) -> CommandResult:
        """Handle voice input toggle command"""
        if not self.speech_service:
            return CommandResult(False, "Voice input not available (speech service not initialized)")
        
        self.voice_input_enabled = not self.voice_input_enabled
        status = "enabled" if self.voice_input_enabled else "disabled"
        return CommandResult(True, f"✓ Voice input {status}")
    
    def _handle_mic_test(self, command: str) -> CommandResult:
        """Handle microphone test command"""
        if not self.speech_service:
            return CommandResult(False, "Speech service not available")
        
        success = self.speech_service.test_microphone()
        if success:
            return CommandResult(True, "✓ Microphone test completed")
        else:
            return CommandResult(False, "⚠ Microphone test failed")
    
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        print("\nShutting down...")
        
        # Save session if auto-save is enabled
        if self.config.auto_save and self.config.session_file:
            self._save_session()
        
        # Stop TTS if speaking
        if self.tts_service and self.tts_service.is_speaking():
            self.tts_service.stop_speaking()
        
        # Cleanup services
        if self.llm_service:
            self.llm_service.cleanup()
        
        if self.tts_service:
            self.tts_service.cleanup()
        
        if self.speech_service:
            self.speech_service.cleanup()
        
        print("Application closed.")
    
    # Public API methods
    def send_message(self, message: str) -> str:
        """Send a message and get response (for programmatic use)"""
        if not self.is_initialized:
            self.initialize()
        
        if not self.memory or not self.llm_service:
            return "Error: Components not initialized"
        
        # Add message and generate response
        self.memory.add_user_message(message)
        response = self.llm_service.generate_response(self.memory)
        self.memory.add_assistant_message(response)
        
        return response
    
    def speak_response(self, text: str, blocking: bool = True) -> None:
        """Speak text using TTS (for programmatic use)"""
        if self.tts_service:
            self.tts_service.speak(text, blocking)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration (for programmatic use)"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.config.validate()
    
    def get_conversation_history(self) -> list:
        """Get conversation history (for programmatic use)"""
        if self.memory:
            return self.memory.get_messages_for_llm()
        return []