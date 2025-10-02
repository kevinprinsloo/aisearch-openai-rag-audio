#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Set environment variables early
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add both current and parent directories to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

from chatbot import Chatbot
from chatbot.config.settings import ChatbotConfig, MemoryConfig, LLMConfig, TTSConfig, SpeechConfig

def create_minimal_config():
    """Create minimal config for testing"""
    memory_config = MemoryConfig()
    llm_config = LLMConfig(provider="qwen", model_name="Qwen/Qwen2.5-3B-Instruct")
    tts_config = TTSConfig(provider="gtts")
    speech_config = SpeechConfig(model_name="openai/whisper-large-v3")
    
    config = ChatbotConfig(
        memory=memory_config,
        llm=llm_config,
        tts=tts_config,
        speech=speech_config,
        session_file="debug_session.json"
    )
    return config

print("Creating chatbot with debug...")
config = create_minimal_config()
chatbot = Chatbot(config)

print("Initializing...")
success = chatbot.initialize()
print(f"Initialization success: {success}")

print(f"Speech service exists: {chatbot.speech_service is not None}")
if chatbot.speech_service:
    print(f"Speech service loaded: {chatbot.speech_service.is_loaded()}")
else:
    print("Speech service is None - checking why...")
    
    # Try to initialize speech service manually
    try:
        from chatbot.services.speech_service import WhisperSpeechService
        print("Attempting manual speech service creation...")
        speech = WhisperSpeechService()
        success = speech.load_model()
        print(f"Manual speech service load: {success}")
    except Exception as e:
        print(f"Manual speech service failed: {e}")
        import traceback
        traceback.print_exc()

