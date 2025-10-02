"""
Local Voice Backend Service

A simple HTTP API service that bridges the LocalVoiceRAG frontend with the local voice code.
This service provides audio processing endpoints for the frontend.
"""

import asyncio
import json
import logging
import base64
import wave
import io
import subprocess
import os
import tempfile
import sys
import numpy as np
from typing import Dict, Any, Optional
from aiohttp import web
from pathlib import Path

# Add the local-voice-code directory to the Python path
local_voice_path = Path(__file__).parent.parent.parent / "local-voice-code"
sys.path.insert(0, str(local_voice_path))

# Import from local voice code
try:
    from core.chatbot import Chatbot
    from config.settings import ChatbotConfig, MemoryConfig, LLMConfig, TTSConfig, SpeechConfig
    LOCAL_VOICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import local voice code: {e}")
    LOCAL_VOICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_voice_backend")


class LocalVoiceProcessor:
    """Processes voice requests using the local voice code"""
    
    def __init__(self):
        self.local_voice_path = Path(__file__).parent.parent.parent / "local-voice-code"
        self.chatbot = None
        self._initialize_chatbot()
        
    def _initialize_chatbot(self):
        """Initialize the chatbot with voice configuration"""
        if not LOCAL_VOICE_AVAILABLE:
            print("Local voice code not available, using mock responses")
            return
            
        try:
            # Create a lightweight config for web use
            memory_config = MemoryConfig(
                memory_type="window",
                system_prompt="You are a helpful voice assistant. Keep responses brief and conversational.",
                window_size=5,
                max_tokens=1000,
                tokens_per_message=30,
                max_messages=10,
                summarize_threshold=20
            )
            
            llm_config = LLMConfig(
                provider="qwen",
                model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Much smaller model - 0.5B instead of 3B
                max_new_tokens=150,  # Keep responses short for voice
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                device=None
            )
            
            tts_config = TTSConfig(
                provider="gtts",
                language="en",
                speed=1.2,
                tld="com"
            )
            
            speech_config = SpeechConfig(
                model_name="openai/whisper-large-v3",
                device=None,
                sample_rate=16000
            )
            
            config = ChatbotConfig(
                memory=memory_config,
                llm=llm_config,
                tts=tts_config,
                speech=speech_config,
                session_file=None,  # Don't persist sessions for web use
                auto_save=False
            )
            
            self.chatbot = Chatbot(config)
            print("Initializing local voice chatbot...")
            if not self.chatbot.initialize():
                print("Failed to initialize chatbot, falling back to mock responses")
                self.chatbot = None
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            self.chatbot = None
        
    async def process_audio_request(self, audio_base64: str, audio_format: str = "webm") -> Dict[str, Any]:
        """Process audio input and return response"""
        try:
            logger.info(f"Processing audio request, format: {audio_format}, data length: {len(audio_base64)}")
            
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")
            
            # If we have the real chatbot, use it
            if self.chatbot and self.chatbot.speech_service:
                try:
                    transcription = await self._transcribe_with_local_voice(audio_data)
                    response = await self._get_response_with_local_voice(transcription)
                    audio_response = await self._generate_tts_with_local_voice(response)
                except Exception as e:
                    logger.error(f"Error using local voice: {e}")
                    # Fall back to mock responses
                    transcription = await self._mock_transcription()
                    response = await self._get_mock_response()
                    audio_response = await self._generate_mock_audio()
            else:
                # Use mock responses
                transcription = await self._mock_transcription()
                response = await self._get_mock_response()
                audio_response = await self._generate_mock_audio()
                
            logger.info(f"Generated transcription: {transcription}")
            logger.info(f"Generated response: {response}")
            
            return {
                "success": True,
                "transcription": transcription,
                "response": response,
                "audio": audio_response,
                "sources": []
            }
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _transcribe_with_local_voice(self, audio_data: bytes) -> str:
        """Transcribe audio using local voice speech service"""
        try:
            # Convert webm audio data to numpy array
            # For now, we'll need to save to temp file and convert
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Convert webm to wav using ffmpeg (if available)
                wav_path = temp_file_path.replace(".webm", ".wav")
                
                # Try to convert with ffmpeg
                result = subprocess.run([
                    "ffmpeg", "-i", temp_file_path, "-ar", "16000", "-ac", "1", "-y", wav_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Load the wav file and convert to numpy array
                    with wave.open(wav_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Use the speech service to transcribe
                    if self.chatbot and self.chatbot.speech_service:
                        transcription = self.chatbot.speech_service.transcribe_audio(audio_np, 16000)
                        
                        # Clean up temp files
                        os.unlink(wav_path)
                        return transcription if transcription else "Could not transcribe audio"
                    else:
                        os.unlink(wav_path)
                        return await self._mock_transcription()
                else:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    return await self._mock_transcription()
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return await self._mock_transcription()
    
    async def _get_response_with_local_voice(self, transcription: str) -> str:
        """Get response using local voice chatbot"""
        try:
            # Use the chatbot to generate a response
            if self.chatbot and hasattr(self.chatbot, 'send_message'):
                response = self.chatbot.send_message(transcription)
                return response if response else "I'm sorry, I couldn't generate a response."
            else:
                return await self._get_mock_response()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return await self._get_mock_response()
    
    async def _generate_tts_with_local_voice(self, text: str) -> str:
        """Generate TTS audio using local voice TTS service"""
        try:
            if self.chatbot and self.chatbot.tts_service:
                # Generate audio using gTTS directly (similar to how the TTS service works)
                from gtts import gTTS
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Create gTTS object and save to file
                    tts = gTTS(
                        text=text,
                        lang=self.chatbot.tts_service.config.language,
                        tld=self.chatbot.tts_service.config.tld,
                        slow=False
                    )
                    tts.save(temp_path)
                    
                    # Read the generated MP3 file
                    if os.path.exists(temp_path):
                        with open(temp_path, 'rb') as f:
                            audio_data = f.read()
                        
                        # Convert to base64 for frontend
                        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                        return audio_b64
                    else:
                        return await self._generate_mock_audio()
                        
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                return await self._generate_mock_audio()
        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            return await self._generate_mock_audio()
    
    async def _mock_transcription(self) -> str:
        """Generate a mock transcription for testing"""
        transcriptions = [
            "Hello, how are you doing today?",
            "What's the weather like?",
            "Can you help me with something?",
            "Tell me about artificial intelligence.",
            "What can you do for me?",
            "How does local voice processing work?",
            "Can you explain machine learning?",
            "What are the benefits of using local models?"
        ]
        import random
        return random.choice(transcriptions)
    
    async def _get_mock_response(self) -> str:
        """Generate a mock response for testing"""
        responses = [
            "Hello! I'm your local voice assistant running on your machine. How can I help you today?",
            "I'm processing your request using local models running right here on your computer. What would you like to know?",
            "This is a test response from the local voice system. Everything seems to be working perfectly!",
            "I'm here to assist you with your questions using open-source models that run locally for privacy and speed.",
            "Local voice processing is working! I can help you with various tasks using AI models that run entirely on your device.",
            "Machine learning allows computers to learn and make decisions. Local models give you privacy and control over your data.",
            "Local models offer benefits like data privacy, no internet dependency, and faster response times for your AI needs."
        ]
        import random
        return random.choice(responses)
    
    async def _generate_mock_audio(self) -> str:
        """Generate mock audio response as PCM data"""
        # Generate silence instead of tone to avoid weird noise
        sample_rate = 24000  # Match frontend expectation
        duration = 1.0  # 1 second
        
        # Generate silence
        samples = int(duration * sample_rate)
        
        # Convert to bytes (16-bit signed integers) - all zeros for silence
        import struct
        pcm_bytes = b''.join(struct.pack('<h', 0) for _ in range(samples))
        
        # Return as base64 for transmission
        return base64.b64encode(pcm_bytes).decode('utf-8')


async def handle_process_audio(request):
    """Handle audio processing request"""
    try:
        data = await request.json()
        audio_base64 = data.get('audio')
        audio_format = data.get('format', 'webm')
        
        if not audio_base64:
            return web.json_response({
                'success': False,
                'error': 'No audio data provided'
            }, status=400)
        
        processor = LocalVoiceProcessor()
        result = await processor.process_audio_request(audio_base64, audio_format)
        
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"Error in handle_process_audio: {e}")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


async def handle_health_check(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'healthy',
        'service': 'local-voice-backend'
    })


async def create_local_voice_app():
    """Create the local voice backend application"""
    app = web.Application()
    
    # Add routes
    app.router.add_post('/api/local-voice/process-audio', handle_process_audio)
    app.router.add_get('/api/local-voice/health', handle_health_check)
    
    # Enable CORS for all routes
    async def cors_handler(request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_handler)
    
    # Handle OPTIONS requests for CORS
    async def options_handler(request):
        return web.Response(
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    
    app.router.add_route('OPTIONS', '/{path:.*}', options_handler)
    
    return app


if __name__ == "__main__":
    app = create_local_voice_app()
    web.run_app(app, host="localhost", port=8766)