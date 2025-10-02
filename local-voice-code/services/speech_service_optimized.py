"""
Optimized Speech Service Module

Provides speech recognition capabilities using OpenAI Whisper models
with HuggingFace pipeline approach for better performance and chunking support.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
import numpy as np
import torch
import gc
import threading
import time
import queue
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import librosa
from io import BytesIO
import wave


class BaseSpeechService(ABC):
    """Abstract base class for speech recognition services"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text"""
        raise NotImplementedError
    
    @abstractmethod
    def record_and_transcribe(self, duration: Optional[float] = None, **kwargs) -> str:
        """Record audio from microphone and transcribe"""
        raise NotImplementedError
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        raise NotImplementedError


class WhisperSpeechService(BaseSpeechService):
    """
    Optimized Speech-to-text service using OpenAI's Whisper model.
    Uses HuggingFace pipeline with chunking support for better performance.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        language: Optional[str] = "english"
    ):
        """
        Initialize the Whisper speech service.
        
        Args:
            model_name: Whisper model to use
            device: Device to run the model on (auto-detected if None)
            sample_rate: Audio sample rate for recording
            chunk_duration: Duration of each audio chunk for recording
            silence_threshold: RMS threshold below which audio is considered silence
            silence_duration: Duration of silence before stopping recording
            language: Language for speech recognition (None for auto-detection)
        """
        super().__init__(model_name)
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.language = language
        
        # Model components - using pipeline approach
        self.pipeline = None
        self.device = device or self._detect_device()
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        print(f"Optimized Whisper Speech Service initialized - Device: {self.device}")
        print(f"Model: {self.model_name}")
        if self.language:
            print(f"Language: {self.language}")
    
    def _detect_device(self) -> str:
        """Detect the best available device"""
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """Load the Whisper model using the optimized pipeline approach"""
        if self.is_loaded():
            return True
            
        try:
            print(f"Loading {self.model_name} with pipeline approach...")
            print("This may take a few minutes on first run...")
            
            # Determine optimal settings based on device
            torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            
            # Load model and processor separately for better control
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Create pipeline with chunking support for longer audio
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,  # Optimal chunk length for large-v3
                batch_size=1,  # Conservative for real-time use
                torch_dtype=torch_dtype,
                device=self.device,
            )
            
            print(f"✅ Pipeline loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pipeline: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded"""
        return self.pipeline is not None
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data to text using the optimized pipeline.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Transcribed text or error message
        """
        if not self.is_loaded():
            return "Sorry, the speech model is not loaded."
        
        try:
            # Ensure audio is the correct type and format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample if necessary (Whisper expects 16kHz)
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000,
                    res_type='kaiser_fast'
                )
            
            # Prepare generation kwargs based on HuggingFace recommendations
            generate_kwargs = {
                "max_new_tokens": 256,  # Optimal for large-v3
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.0,
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            # Add language specification if provided
            if self.language:
                generate_kwargs["language"] = self.language
                generate_kwargs["task"] = "transcribe"
            
            # Use pipeline for transcription with chunking support
            result = self.pipeline(
                audio_data,
                generate_kwargs=generate_kwargs
            )
            
            # Extract text from result
            transcription = result.get("text", "").strip()
            
            # Clean up GPU memory
            if self.device != "cpu":
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if hasattr(torch.backends, 'mps'):
                    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            
            # Validate transcription
            if not transcription or len(transcription.strip()) == 0:
                return "Sorry, I couldn't understand that. Please try again."
            
            print(f"🎤 Transcribed: '{transcription}'")
            return transcription
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"Sorry, I couldn't process that audio. Error: {str(e)}"
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording"""
        if status:
            print(f"Audio status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def _record_audio_continuous(self) -> np.ndarray:
        """
        Record audio continuously until silence is detected.
        
        Returns:
            Recorded audio data as numpy array
        """
        print("🎤 Listening... (speak now)")
        
        # Recording parameters
        chunk_size = int(self.sample_rate * self.chunk_duration)
        audio_chunks = []
        silence_chunks = 0
        max_silence_chunks = int(self.silence_duration / self.chunk_duration)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_size,
                callback=self._audio_callback
            ):
                self.is_recording = True
                
                while self.is_recording:
                    try:
                        # Get audio chunk with timeout
                        chunk = self.audio_queue.get(timeout=self.chunk_duration * 2)
                        audio_chunks.append(chunk.flatten())
                        
                        # Check for silence
                        rms = np.sqrt(np.mean(chunk.flatten() ** 2))
                        
                        if rms < self.silence_threshold:
                            silence_chunks += 1
                            if silence_chunks >= max_silence_chunks:
                                print("🔇 Silence detected, stopping recording")
                                break
                        else:
                            silence_chunks = 0  # Reset silence counter
                            
                    except queue.Empty:
                        print("⚠️ Audio queue timeout")
                        break
                        
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return np.array([])
        finally:
            self.is_recording = False
            
            # Clear any remaining audio in queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        if not audio_chunks:
            print("⚠️ No audio recorded")
            return np.array([])
        
        # Combine all chunks
        audio_data = np.concatenate(audio_chunks)
        print(f"📊 Recorded {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.1f}s)")
        print(f"📊 Audio RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        
        return audio_data
    
    def record_and_transcribe(self, duration: Optional[float] = None, **kwargs) -> str:
        """
        Record audio from microphone and transcribe it.
        
        Args:
            duration: Maximum recording duration (None for silence detection)
            **kwargs: Additional arguments (maintained for compatibility)
            
        Returns:
            Transcribed text
        """
        if not self.is_loaded():
            if not self.load_model():
                return "Sorry, I couldn't load the speech recognition model."
        
        try:
            if duration is not None:
                # Fixed duration recording
                print(f"🎤 Recording for {duration} seconds...")
                audio_data = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()  # Wait for recording to complete
                audio_data = audio_data.flatten()
            else:
                # Continuous recording with silence detection
                audio_data = self._record_audio_continuous()
            
            if len(audio_data) == 0:
                return "Sorry, no audio was recorded."
            
            # Transcribe the recorded audio
            return self.transcribe_audio(audio_data, self.sample_rate)
            
        except Exception as e:
            print(f"❌ Recording and transcription error: {e}")
            return f"Sorry, there was an error with recording: {str(e)}"
    
    def cleanup(self) -> None:
        """Clean up resources"""
        print("🧹 Cleaning up speech service...")
        
        # Stop any ongoing recording
        self.is_recording = False
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clean up model
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        gc.collect()
        print("✅ Speech service cleanup complete")


def create_speech_service(config=None):
    """Factory function to create a speech service instance"""
    return WhisperSpeechService(config)