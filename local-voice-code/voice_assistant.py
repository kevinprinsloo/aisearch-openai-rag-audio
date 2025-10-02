#!/usr/bin/env python3
"""
Simple Voice Assistant (Self-Contained)

A simplified version that works entirely within the chatbot folder
without complex import issues.
"""

import os
import sys
import time
import platform
from pathlib import Path

# Set environment variables early
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the local modules (we're now in the chatbot directory)
from core.chatbot import Chatbot
from config.settings import ChatbotConfig, MemoryConfig, LLMConfig, TTSConfig, SpeechConfig

import torch


def create_voice_config():
    """Create configuration optimized for voice interaction"""
    
    # Memory configuration - optimized for voice conversation
    memory_config = MemoryConfig(
        memory_type="window",
        system_prompt="""You are a natural voice assistant. Keep responses conversational, 
concise, and under 3 sentences unless asked for more detail. Speak naturally like 
you're having a face-to-face conversation. Be helpful, friendly, and engaging.""",
        window_size=8,  # Shorter context for faster responses
        max_tokens=1500,
        tokens_per_message=40,
        max_messages=20,  # Reduced for faster voice responses
        summarize_threshold=30  # Must be greater than max_messages
    )
    
    # LLM configuration - optimized for quick voice responses
    llm_config = LLMConfig(
        provider="qwen",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=256,  # Shorter responses for voice
        temperature=0.8,     # Slightly more creative
        do_sample=True,
        top_p=0.9,
        device=None  # Auto-detect
    )
    
    # TTS configuration - faster speech for natural conversation
    tts_config = TTSConfig(
        provider="gtts",
        language="en",
        speed=1.3,  # Slightly faster for natural flow
        tld="com",
        chunk_length=80,
        crossfade_length=5
    )
    
    # Speech recognition configuration
    speech_config = SpeechConfig(
        model_name="openai/whisper-large-v3",
        device=None,  # Auto-detect
        sample_rate=16000,
        chunk_duration=1.0,
        silence_threshold=0.01,
        silence_duration=2.0,
        language="english"
    )
    
    # Main chatbot configuration
    config = ChatbotConfig(
        memory=memory_config,
        llm=llm_config,
        tts=tts_config,
        speech=speech_config,
        session_file="voice_session.json",
        auto_save=True,
        enable_voice_commands=True,
        show_thinking=False,  # Hide thinking for smoother flow
        show_speaking=False   # Hide speaking indicator
    )
    
    return config


def print_system_info():
    """Print system information"""
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"⚡ CUDA: Available (GPU: {torch.cuda.get_device_name()})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 MPS: Available (Apple Silicon GPU)")
    else:
        print("💻 GPU: Not available, using CPU")
    print()


class VoiceAssistant:
    """Complete voice assistant with continuous listening"""
    
    def __init__(self):
        self.config = create_voice_config()
        self.chatbot = Chatbot(self.config)
        self.conversation_active = False
    
    def start_conversation(self):
        """Start natural voice conversation"""
        print("=== 🎤 Voice Assistant (Self-Contained) ===")
        print("Always listening mode - speak naturally!")
        print("=========================================\n")
        
        print_system_info()
        
        # Initialize chatbot
        print("🤖 Initializing voice assistant...")
        if not self.chatbot.initialize():
            print("❌ Failed to initialize. Exiting.")
            return
        
        print("\n🎤 Voice Assistant Ready!")
        print("="*40)
        print("💡 Tips:")
        print("  - Speak naturally after you hear the instruction")
        print("  - Say 'exit' or 'goodbye' to end")
        print("  - Say 'repeat' if you didn't hear the response")
        print("  - Pause between sentences for clarity")
        print("="*40 + "\n")
        
        # Check speech service
        if not self.chatbot.speech_service:
            print("❌ Speech service not available. Cannot run voice mode.")
            print("💡 Try running: python text_assistant.py for text mode")
            return
        
        print("🔍 Testing microphone...")
        if not self._test_microphone():
            print("⚠️ Microphone test failed. Check your audio settings.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        self.conversation_active = True
        
        # Greeting
        greeting = "Hi! I'm your voice assistant. What would you like to talk about?"
        print(f"🤖 {greeting}")
        if self.chatbot.tts_service:
            self.chatbot.tts_service.speak(greeting, blocking=True)
        
        # Main conversation loop
        conversation_count = 0
        last_response = ""
        
        try:
            while self.conversation_active:
                print(f"\n--- Turn {conversation_count + 1} ---")
                
                # Get voice input
                print("🎤 Listening...")
                user_input = self._get_voice_input()
                
                if not user_input:
                    print("No input detected. Continuing to listen...")
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['exit', 'quit', 'goodbye', 'bye']):
                    farewell = "Goodbye! Have a great day!"
                    print(f"🤖 {farewell}")
                    if self.chatbot.tts_service:
                        self.chatbot.tts_service.speak(farewell, blocking=True)
                    break
                
                # Check for repeat request
                if 'repeat' in user_input.lower() and last_response:
                    print(f"🤖 {last_response}")
                    if self.chatbot.tts_service:
                        self.chatbot.tts_service.speak(last_response, blocking=True)
                    continue
                
                # Process input through chatbot
                try:
                    # Use the modular chatbot's send_message method
                    response = self.chatbot.send_message(user_input)
                    
                    # Display and speak response
                    print(f"🤖 {response}")
                    if self.chatbot.tts_service:
                        self.chatbot.tts_service.speak(response, blocking=True)
                    
                    last_response = response
                    conversation_count += 1
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    print(f"❌ {error_msg}")
                    if self.chatbot.tts_service:
                        self.chatbot.tts_service.speak(error_msg, blocking=True)
        
        except KeyboardInterrupt:
            print("\n\n👋 Voice assistant interrupted. Goodbye!")
        
        finally:
            self._cleanup()
    
    def _get_voice_input(self) -> str:
        """Get voice input from user"""
        try:
            # Use the speech service to record and transcribe
            return self.chatbot.speech_service.record_and_transcribe()
        except Exception as e:
            print(f"❌ Voice input error: {e}")
            return ""
    
    def _test_microphone(self) -> bool:
        """Test microphone functionality"""
        try:
            print("🎤 Testing... (say something briefly)")
            test_input = self.chatbot.speech_service.record_and_transcribe(duration=1.0)
            if test_input and test_input.strip():
                print(f"✅ Microphone test successful! Heard: '{test_input}'")
                return True
            else:
                print("❌ Microphone test failed - no speech detected")
                return False
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False
    
    def _cleanup(self):
        """Clean up resources"""
        print("\nShutting down...")
        if self.chatbot:
            self.chatbot.cleanup()
        print("Application closed.")


def main():
    """Main entry point"""
    print("🚀 Self-Contained Voice Assistant")
    print("="*50)
    print("🎯 True voice mode - listens to your speech!")
    print("📁 Running from isolated chatbot/ folder")
    print("="*50 + "\n")
    
    try:
        assistant = VoiceAssistant()
        assistant.start_conversation()
    
    except KeyboardInterrupt:
        print("\n👋 Voice assistant interrupted. Goodbye!")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
