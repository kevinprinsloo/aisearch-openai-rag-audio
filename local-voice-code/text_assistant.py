#!/usr/bin/env python3
"""
Simple Text Assistant (Self-Contained)

A text-based version for testing the chatbot functionality
before trying voice mode.
"""

import os
import sys
import platform
from pathlib import Path

# Set environment variables early
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the local modules (we're now in the chatbot directory)
from core.chatbot import Chatbot
from config.settings import ChatbotConfig, MemoryConfig, LLMConfig, TTSConfig

import torch


def create_text_config():
    """Create configuration optimized for text interaction"""
    
    # Memory configuration
    memory_config = MemoryConfig(
        memory_type="window",
        system_prompt="""You are a helpful AI assistant. Provide informative, 
accurate, and engaging responses. Be conversational but concise.""",
        window_size=10,
        max_tokens=2000,
        tokens_per_message=50,
        max_messages=30,
        summarize_threshold=40
    )
    
    # LLM configuration
    llm_config = LLMConfig(
        provider="qwen",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        device=None  # Auto-detect
    )
    
    # TTS configuration (optional for text mode)
    tts_config = TTSConfig(
        provider="gtts",
        language="en",
        speed=1.2,
        tld="com",
        chunk_length=100,
        crossfade_length=5
    )
    
    # Main chatbot configuration
    config = ChatbotConfig(
        memory=memory_config,
        llm=llm_config,
        tts=tts_config,
        speech=None,  # No speech for text mode
        session_file="text_session.json",
        auto_save=True,
        enable_voice_commands=False,  # Disable voice for text mode
        show_thinking=True,
        show_speaking=True
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


class TextAssistant:
    """Text-based assistant for testing functionality"""
    
    def __init__(self):
        self.config = create_text_config()
        self.chatbot = Chatbot(self.config)
        self.conversation_active = False
    
    def start_conversation(self):
        """Start text-based conversation"""
        print("=== ⌨️  Text Assistant (Self-Contained) ===")
        print("Type your messages - TTS available with 'speak' command")
        print("=========================================\n")
        
        print_system_info()
        
        # Initialize chatbot
        print("🤖 Initializing text assistant...")
        if not self.chatbot.initialize():
            print("❌ Failed to initialize. Exiting.")
            return
        
        print("\n⌨️  Text Assistant Ready!")
        print("="*40)
        print("💡 Commands:")
        print("  - Type your message and press Enter")
        print("  - Type 'speak <message>' to hear TTS")
        print("  - Type 'exit' or 'quit' to end")
        print("  - Type 'help' for more commands")
        print("="*40 + "\n")
        
        self.conversation_active = True
        
        # Greeting
        greeting = "Hi! I'm your text assistant. How can I help you today?"
        print(f"🤖 {greeting}\n")
        
        # Main conversation loop
        conversation_count = 0
        
        try:
            while self.conversation_active:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = "Goodbye! Thanks for chatting!"
                    print(f"🤖 {farewell}")
                    break
                
                # Check for speak command
                if user_input.lower().startswith('speak '):
                    message = user_input[6:]  # Remove 'speak ' prefix
                    if message and self.chatbot.tts_service:
                        print(f"🔊 Speaking: {message}")
                        self.chatbot.tts_service.speak(message, blocking=True)
                    else:
                        print("❌ Nothing to speak or TTS not available")
                    continue
                
                # Process input through chatbot
                try:
                    print("🤔 Thinking...")
                    response = self.chatbot.send_message(user_input)
                    print(f"🤖 {response}\n")
                    
                    conversation_count += 1
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    print(f"❌ {error_msg}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Text assistant interrupted. Goodbye!")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        print("\nShutting down...")
        if self.chatbot:
            self.chatbot.cleanup()
        print("Application closed.")


def main():
    """Main entry point"""
    print("⌨️  Self-Contained Text Assistant")
    print("="*50)
    print("🎯 Text mode - test functionality before voice!")
    print("📁 Running from isolated chatbot/ folder")
    print("="*50 + "\n")
    
    try:
        assistant = TextAssistant()
        assistant.start_conversation()
    
    except KeyboardInterrupt:
        print("\n👋 Text assistant interrupted. Goodbye!")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
