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
    _instance = None
    _chatbot = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalVoiceProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.initialized = False
            logger.info("LocalVoiceProcessor instance created")
    
    @property
    def chatbot(self):
        return self._chatbot
    
    def is_initialized(self):
        return self._initialized and self._chatbot is not None
    
    async def initialize_models(self):
        """Initialize models once and reuse them"""
        if self._initialized and self._chatbot is not None:
            logger.info("Models already initialized, skipping...")
            return True
            
        try:
            logger.info("🚀 Initializing local voice models (one-time setup)...")
            
            # Import and set up configuration
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'local-voice-code'))
            from core.chatbot import Chatbot
            from config.settings import get_default_config
            
            config = get_default_config()
            self._chatbot = Chatbot(config)
            
            logger.info("Initializing local voice chatbot...")
            if not self._chatbot.initialize():
                logger.error("Failed to initialize chatbot")
                self._chatbot = None
                return False
            
            self._initialized = True
            logger.info("✅ Local voice models initialized successfully and ready for reuse!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._chatbot = None
            self._initialized = False
            return False
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._chatbot = None
            self._initialized = False
            return False
        
    async def process_audio_request(self, audio_base64: str, audio_format: str = "webm") -> Dict[str, Any]:
        """Process audio input and return response"""
        import time
        
        try:
            logger.info(f"Processing audio request, format: {audio_format}, data length: {len(audio_base64)}")
            
            # Track timing metrics
            start_time = time.time()
            metrics = {}
            
            # For streaming response, we'll return partial results
            partial_response = {
                "success": True,
                "transcription": None,
                "response_text": None,
                "audio": None,
                "sources": [],
                "metrics": {},
                "status": "processing"
            }
            
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")
            
            # Ensure models are initialized (this will be fast after first time)
            if not self.is_initialized():
                logger.info("Models not initialized, initializing now...")
                if not await self.initialize_models():
                    logger.error("Failed to initialize models, using mock responses")
                    transcription = await self._mock_transcription()
                    response = await self._get_mock_response()
                    audio_response = await self._generate_mock_audio()
                else:
                    logger.info("Models initialized successfully")
            
            # Try to use local voice processing
            if self.is_initialized() and self._chatbot and self._chatbot.speech_service:
                try:
                    # Transcription timing
                    transcription_start = time.time()
                    transcription = await self._transcribe_with_local_voice(audio_data)
                    transcription_time = (time.time() - transcription_start) * 1000  # Convert to ms
                    metrics['transcription_ms'] = round(transcription_time, 2)
                    
                    # LLM response timing
                    llm_start = time.time()
                    response = await self._get_response_with_local_voice(transcription)
                    llm_time = (time.time() - llm_start) * 1000  # Convert to ms
                    metrics['llm_ms'] = round(llm_time, 2)
                    
                    # TTS timing
                    tts_start = time.time()
                    audio_response = await self._generate_tts_with_local_voice(response)
                    tts_time = (time.time() - tts_start) * 1000  # Convert to ms
                    metrics['tts_ms'] = round(tts_time, 2)
                    
                except Exception as e:
                    logger.error(f"Error using local voice: {e}")
                    # Fall back to mock responses
                    transcription = await self._mock_transcription()
                    response = await self._get_mock_response()
                    audio_response = await self._generate_mock_audio()
                    metrics['error'] = str(e)
            else:
                # Use mock responses
                transcription = await self._mock_transcription()
                response = await self._get_mock_response()
                audio_response = await self._generate_mock_audio()
                metrics['mode'] = 'mock'
            
            # Total processing time
            total_time = (time.time() - start_time) * 1000
            metrics['total_ms'] = round(total_time, 2)
                
            logger.info(f"Generated transcription: {transcription}")
            logger.info(f"Generated response: {response}")
            logger.info(f"Performance metrics: {metrics}")
            
            return {
                "success": True,
                "transcription": transcription,
                "response": response,
                "response_text": response,  # Add text response for immediate display
                "audio": audio_response,
                "sources": [],
                "metrics": metrics
            }
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {}
            }
    
    async def process_audio_request_streaming(self, audio_base64: str, audio_format: str = "webm"):
        """Process audio input with streaming updates - yields chunks as they become available"""
        import time
        import re
        
        try:
            logger.info(f"Processing audio request with streaming, format: {audio_format}, data length: {len(audio_base64)}")
            
            # Track timing metrics
            start_time = time.time()
            metrics = {}
            
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")
            
            # Ensure models are initialized
            if not self.is_initialized():
                logger.info("Models not initialized, initializing now...")
                if not await self.initialize_models():
                    logger.error("Failed to initialize models, using mock responses")
                    yield {
                        "type": "error",
                        "error": "Failed to initialize models"
                    }
                    return
            
            # Step 1: Process transcription first
            if self.is_initialized() and self._chatbot and self._chatbot.speech_service:
                try:
                    # Transcription timing
                    transcription_start = time.time()
                    transcription = await self._transcribe_with_local_voice(audio_data)
                    transcription_time = (time.time() - transcription_start) * 1000
                    metrics['transcription_ms'] = round(transcription_time, 2)
                    
                    # Send transcription immediately
                    yield {
                        "type": "transcription",
                        "transcription": transcription
                    }
                    
                    # Step 2: Get LLM response
                    llm_start = time.time()
                    response = await self._get_response_with_local_voice(transcription)
                    llm_time = (time.time() - llm_start) * 1000
                    metrics['llm_ms'] = round(llm_time, 2)
                    
                    # Send response text immediately
                    yield {
                        "type": "response_text",
                        "response_text": response
                    }
                    
                    # Step 3: Generate TTS in chunks (sentence by sentence)
                    tts_start = time.time()
                    
                    # Split response into sentences for streaming TTS
                    sentences = re.split(r'(?<=[.!?])\s+', response)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    logger.info(f"🎵 Streaming TTS for {len(sentences)} sentences...")
                    
                    for i, sentence in enumerate(sentences):
                        # Generate TTS for this sentence
                        audio_chunk = await self._generate_tts_chunk(sentence)
                        
                        if audio_chunk:
                            # Send audio chunk immediately
                            yield {
                                "type": "audio_delta",
                                "delta": audio_chunk,
                                "sentence_index": i,
                                "total_sentences": len(sentences)
                            }
                    
                    tts_time = (time.time() - tts_start) * 1000
                    metrics['tts_ms'] = round(tts_time, 2)
                    
                except Exception as e:
                    logger.error(f"Error using local voice: {e}")
                    yield {
                        "type": "error",
                        "error": str(e)
                    }
                    return
            else:
                yield {
                    "type": "error",
                    "error": "Local voice service not available"
                }
                return
            
            # Total processing time
            total_time = (time.time() - start_time) * 1000
            metrics['total_ms'] = round(total_time, 2)
            
            # Send final metrics
            yield {
                "type": "metrics",
                "metrics": metrics
            }
            
            # Send completion signal
            yield {
                "type": "done"
            }
                
        except Exception as e:
            logger.error(f"Error processing audio streaming: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def _transcribe_with_local_voice(self, audio_data: bytes) -> str:
        """Transcribe audio using local voice speech service"""
        try:
            # Debug: Check audio data
            logger.info(f"Audio data length: {len(audio_data)} bytes")
            logger.info(f"First 20 bytes: {audio_data[:20].hex() if len(audio_data) >= 20 else audio_data.hex()}")
            
            # The audio data is actually PCM data from the browser, not WebM
            # Let's try to process it directly as PCM
            try:
                # Convert bytes to numpy array (assuming 16-bit PCM)
                # The frontend sends Int16 data as bytes
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check if we have valid audio data
                if len(audio_np) == 0:
                    logger.error("No audio data after PCM processing")
                    return await self._mock_transcription()
                
                # Check if audio is not all zeros
                if np.all(audio_np == 0):
                    logger.warning("Audio data is all zeros - no speech detected")
                    return "No speech detected. Please try speaking louder."
                
                logger.info(f"Successfully processed PCM audio: {len(audio_np)} samples, RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")
                
                # Use the speech service to transcribe
                if self._chatbot and self._chatbot.speech_service:
                    transcription = self._chatbot.speech_service.transcribe_audio(audio_np, 24000)  # Browser uses 24kHz
                    return transcription if transcription else "Could not transcribe audio"
                else:
                    return await self._mock_transcription()
                    
            except Exception as pcm_error:
                logger.error(f"PCM processing failed: {pcm_error}")
                # Try librosa as fallback
                return await self._transcribe_with_librosa(audio_data)
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return await self._mock_transcription()
    
    async def _transcribe_with_librosa(self, audio_data: bytes) -> str:
        """Try to transcribe using librosa (fallback method)"""
        try:
            # Convert bytes to BytesIO for librosa
            audio_buffer = io.BytesIO(audio_data)
            
            # Try to load with librosa directly
            import librosa
            audio_np, _ = librosa.load(audio_buffer, sr=16000, mono=True)
            
            # Ensure we have valid audio data
            if len(audio_np) == 0:
                logger.error("No audio data after librosa processing")
                return await self._mock_transcription()
            
            logger.info(f"Successfully processed audio with librosa: {len(audio_np)} samples")
            
            # Use the speech service to transcribe
            if self.chatbot and self.chatbot.speech_service:
                transcription = self.chatbot.speech_service.transcribe_audio(audio_np, 16000)
                return transcription if transcription else "Could not transcribe audio"
            else:
                return await self._mock_transcription()
                
        except Exception as librosa_error:
            logger.error(f"Librosa processing failed: {librosa_error}")
            # Fall back to FFmpeg approach
            return await self._transcribe_with_ffmpeg(audio_data)
    
    async def _transcribe_with_ffmpeg(self, audio_data: bytes) -> str:
        """Fallback transcription using FFmpeg"""
        try:
            # Convert webm audio data to wav using FFmpeg
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Convert webm to wav using ffmpeg (if available)
                wav_path = temp_file_path.replace(".webm", ".wav")
                
                # Try to convert with ffmpeg - check multiple possible locations
                ffmpeg_paths = [
                    "/opt/homebrew/bin/ffmpeg",  # Homebrew ARM Mac
                    "/usr/local/bin/ffmpeg",     # Homebrew Intel Mac
                    "ffmpeg"                     # System PATH
                ]
                
                ffmpeg_cmd = None
                for path in ffmpeg_paths:
                    try:
                        if os.path.exists(path) or path == "ffmpeg":
                            # Test if the command works
                            test_result = subprocess.run([path, "-version"], 
                                                       capture_output=True, 
                                                       text=True, 
                                                       timeout=5,
                                                       check=False)
                            if test_result.returncode == 0:
                                ffmpeg_cmd = path
                                break
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                        continue
                
                if ffmpeg_cmd is None:
                    logger.error("FFmpeg not found in any expected location")
                    return await self._mock_transcription()
                
                # Try to convert with ffmpeg
                result = subprocess.run([
                    ffmpeg_cmd, "-i", temp_file_path, "-ar", "16000", "-ac", "1", "-y", wav_path
                ], capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    # Load the wav file and convert to numpy array
                    with wave.open(wav_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Use the speech service to transcribe
                    if self._chatbot and self._chatbot.speech_service:
                        transcription = self._chatbot.speech_service.transcribe_audio(audio_np, 16000)
                        
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
            logger.error(f"Error in FFmpeg transcription: {e}")
            return await self._mock_transcription()
    
    async def _get_response_with_local_voice(self, transcription: str) -> str:
        """Get response from local LLM"""
        try:
            if self._chatbot:
                response = self._chatbot.send_message(transcription)
                return response if response else "I couldn't generate a response."
            else:
                return "Local voice service not available."
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "Sorry, I encountered an error processing your request."
    
    async def _generate_tts_chunk(self, text: str) -> str:
        """Generate TTS audio for a single chunk/sentence (optimized for streaming)"""
        try:
            if self._chatbot and self._chatbot.tts_service:
                from gtts import gTTS
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                pcm_path = temp_path.replace('.mp3', '.pcm')
                
                try:
                    # Create gTTS object and save to file
                    tts = gTTS(
                        text=text,
                        lang=self._chatbot.tts_service.config.language,
                        tld=self._chatbot.tts_service.config.tld,
                        slow=False
                    )
                    tts.save(temp_path)
                    
                    # Convert MP3 to PCM using ffmpeg
                    ffmpeg_paths = [
                        '/opt/homebrew/bin/ffmpeg',
                        '/usr/local/bin/ffmpeg', 
                        '/usr/bin/ffmpeg',
                        'ffmpeg'
                    ]
                    
                    ffmpeg_cmd = None
                    for path in ffmpeg_paths:
                        try:
                            if os.path.exists(path) or path == "ffmpeg":
                                test_result = subprocess.run([path, "-version"], 
                                                           capture_output=True, 
                                                           text=True, 
                                                           timeout=5,
                                                           check=False)
                                if test_result.returncode == 0:
                                    ffmpeg_cmd = path
                                    break
                        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                            continue
                    
                    if ffmpeg_cmd is None:
                        logger.warning("FFmpeg not found, falling back to mock audio")
                        return await self._generate_mock_audio()
                    
                    # Convert MP3 to 16-bit PCM at 24kHz (matching frontend expectations)
                    result = subprocess.run([
                        ffmpeg_cmd, "-i", temp_path, 
                        "-f", "s16le",
                        "-ar", "24000",
                        "-ac", "1",
                        "-y", pcm_path
                    ], capture_output=True, text=True, check=False)
                    
                    if result.returncode == 0 and os.path.exists(pcm_path):
                        with open(pcm_path, 'rb') as f:
                            pcm_data = f.read()
                        
                        audio_b64 = base64.b64encode(pcm_data).decode('utf-8')
                        logger.info(f"✅ Generated TTS chunk: {len(pcm_data)} bytes for text: {text[:50]}...")
                        return audio_b64
                    else:
                        logger.error(f"FFmpeg conversion failed: {result.stderr}")
                        return await self._generate_mock_audio()
                        
                finally:
                    for file_path in [temp_path, pcm_path]:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
            else:
                return await self._generate_mock_audio()
        except Exception as e:
            logger.error(f"Error generating TTS chunk: {e}")
            return await self._generate_mock_audio()
    
    async def _generate_tts_with_local_voice(self, text: str) -> str:
        """Generate TTS audio using local voice TTS service"""
        try:
            if self._chatbot and self._chatbot.tts_service:
                # Generate audio using gTTS directly (similar to how the TTS service works)
                from gtts import gTTS
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Output PCM file path
                pcm_path = temp_path.replace('.mp3', '.pcm')
                
                try:
                    # Create gTTS object and save to file
                    tts = gTTS(
                        text=text,
                        lang=self._chatbot.tts_service.config.language,
                        tld=self._chatbot.tts_service.config.tld,
                        slow=False
                    )
                    tts.save(temp_path)
                    
                    # Convert MP3 to PCM using ffmpeg (same approach as transcription)
                    ffmpeg_paths = [
                        '/opt/homebrew/bin/ffmpeg',
                        '/usr/local/bin/ffmpeg', 
                        '/usr/bin/ffmpeg',
                        'ffmpeg'
                    ]
                    
                    ffmpeg_cmd = None
                    for path in ffmpeg_paths:
                        try:
                            if os.path.exists(path) or path == "ffmpeg":
                                test_result = subprocess.run([path, "-version"], 
                                                           capture_output=True, 
                                                           text=True, 
                                                           timeout=5,
                                                           check=False)
                                if test_result.returncode == 0:
                                    ffmpeg_cmd = path
                                    break
                        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                            continue
                    
                    if ffmpeg_cmd is None:
                        logger.warning("FFmpeg not found, falling back to mock audio")
                        return await self._generate_mock_audio()
                    
                    # Convert MP3 to 16-bit PCM at 24kHz (matching frontend expectations)
                    result = subprocess.run([
                        ffmpeg_cmd, "-i", temp_path, 
                        "-f", "s16le",  # 16-bit signed little-endian PCM
                        "-ar", "24000",  # 24kHz sample rate (matching frontend)
                        "-ac", "1",      # Mono
                        "-y", pcm_path
                    ], capture_output=True, text=True, check=False)
                    
                    if result.returncode == 0 and os.path.exists(pcm_path):
                        # Read the PCM data
                        with open(pcm_path, 'rb') as f:
                            pcm_data = f.read()
                        
                        # Convert to base64 for transmission
                        audio_b64 = base64.b64encode(pcm_data).decode('utf-8')
                        logger.info(f"✅ Generated TTS audio: {len(pcm_data)} bytes PCM data")
                        return audio_b64
                    else:
                        logger.error(f"FFmpeg conversion failed: {result.stderr}")
                        return await self._generate_mock_audio()
                        
                finally:
                    # Clean up temp files
                    for file_path in [temp_path, pcm_path]:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
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
    """Handle audio processing request with enhanced status updates"""
    try:
        data = await request.json()
        audio_base64 = data.get('audio')
        audio_format = data.get('format', 'webm')
        
        if not audio_base64:
            return web.json_response({
                'success': False,
                'error': 'No audio data provided'
            }, status=400)
        
        # Use the global singleton instance
        processor = LocalVoiceProcessor()
        result = await processor.process_audio_request(audio_base64, audio_format)
        
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"Error in handle_process_audio: {e}")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


async def handle_process_audio_streaming(request):
    """Handle audio processing with streaming updates using Server-Sent Events"""
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
        
        # Create streaming response
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        await response.prepare(request)
        
        # Process with streaming updates
        async for chunk in processor.process_audio_request_streaming(audio_base64, audio_format):
            # Send each chunk as a Server-Sent Event
            event_data = f"data: {json.dumps(chunk)}\n\n"
            await response.write(event_data.encode('utf-8'))
        
        await response.write_eof()
        return response
        
    except Exception as e:
        logger.error(f"Error in handle_process_audio_streaming: {e}")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)


async def handle_warmup(request):
    """Warm up the models - initialize them ahead of time"""
    try:
        logger.info("🔥 Warmup request received - initializing models...")
        processor = LocalVoiceProcessor()
        
        if processor.is_initialized():
            return web.json_response({
                'success': True,
                'message': 'Models already initialized',
                'initialized': True
            })
        
        success = await processor.initialize_models()
        
        if success:
            return web.json_response({
                'success': True,
                'message': 'Models initialized successfully',
                'initialized': True
            })
        else:
            return web.json_response({
                'success': False,
                'error': 'Failed to initialize models',
                'initialized': False
            }, status=500)
        
    except Exception as e:
        logger.error(f"Error in handle_warmup: {e}")
        return web.json_response({
            'success': False,
            'error': str(e),
            'initialized': False
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
    app.router.add_post('/api/local-voice/process-audio-streaming', handle_process_audio_streaming)
    app.router.add_post('/api/local-voice/warmup', handle_warmup)
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