"""
LLM Service Module

Provides a unified interface for different Language Model providers
with conversation memory integration and streaming support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory.conversation_memory import ConversationMemory


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    def generate_response(self, memory: ConversationMemory, **kwargs) -> str:
        """Generate response using conversation memory"""
        raise NotImplementedError
    
    @abstractmethod
    def generate_stream(self, memory: ConversationMemory, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using conversation memory"""
        raise NotImplementedError
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        raise NotImplementedError


class QwenLLMService(BaseLLMService):
    """Qwen model service with memory integration"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 do_sample: bool = True,
                 top_p: float = 0.9,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = device or self._detect_device()
        
        print(f"Qwen LLM Service initialized - Device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """Load the Qwen model and tokenizer"""
        if self.is_loaded():
            return True
            
        try:
            print(f"Loading {self.model_name}...")
            print("This may take a few minutes on first run...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure model loading based on device
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            }
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to device
            if self.device != "cpu" and self.model is not None:
                self.model = self.model.to(self.device)
            
            print(f"✓ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def generate_response(self, memory: ConversationMemory, **kwargs) -> str:
        """Generate response using conversation memory"""
        if not self.is_loaded():
            if not self.load_model():
                return "Error: Model failed to load."
        
        # Type guards for None checks
        if self.model is None or self.tokenizer is None:
            return "Error: Model or tokenizer not loaded."
        
        try:
            # Get messages from memory
            messages = memory.get_messages_for_llm()
            
            if not messages:
                return "Error: No conversation context available."
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt")
            if self.device != "cpu":
                model_inputs = model_inputs.to(self.device)
            
            # Override generation parameters if provided
            generation_params = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": kwargs.get("do_sample", self.do_sample),
                "top_p": kwargs.get("top_p", self.top_p),
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    **generation_params
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up GPU memory
            self._cleanup_memory()
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def generate_stream(self, memory: ConversationMemory, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response (placeholder for future implementation)"""
        # For now, yield the complete response
        # Future implementation could use TextIteratorStreamer
        response = self.generate_response(memory, **kwargs)
        
        # Simulate streaming by yielding chunks
        words = response.split()
        current_text = ""
        
        for word in words:
            current_text += word + " "
            yield word + " "
    
    def _cleanup_memory(self) -> None:
        """Clean up GPU memory after generation"""
        if self.device in ["cuda", "mps"]:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def cleanup(self) -> None:
        """Clean up model and free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self._cleanup_memory()
        print("LLM model cleaned up")
    
    def update_generation_params(self, **kwargs) -> None:
        """Update generation parameters"""
        if "max_new_tokens" in kwargs:
            self.max_new_tokens = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "do_sample" in kwargs:
            self.do_sample = kwargs["do_sample"]
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]


class OpenAILLMService(BaseLLMService):
    """OpenAI API service (placeholder for future implementation)"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    def is_loaded(self) -> bool:
        """OpenAI API doesn't need model loading"""
        return bool(self.api_key)
    
    def generate_response(self, memory: ConversationMemory, **kwargs) -> str:
        """Generate response using OpenAI API"""
        if not self.is_loaded():
            return "Error: OpenAI API key not configured."
        
        # Placeholder - would implement OpenAI API calls here
        return "OpenAI integration coming soon..."
    
    def generate_stream(self, memory: ConversationMemory, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using OpenAI API"""
        response = self.generate_response(memory, **kwargs)
        yield response
    
    def cleanup(self) -> None:
        """No cleanup needed for API-based service"""
        pass


class LLMService:
    """Main LLM service that manages different providers"""
    
    def __init__(self, 
                 provider: str = "qwen",
                 model_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize LLM service
        
        Args:
            provider: LLM provider ('qwen', 'openai')
            model_name: Model name (optional, uses provider defaults)
            **kwargs: Additional configuration for the provider
        """
        self.provider = provider.lower()
        
        # Initialize appropriate service
        if self.provider == "qwen":
            default_model = "Qwen/Qwen2.5-3B-Instruct"
            self.service = QwenLLMService(
                model_name=model_name or default_model,
                **kwargs
            )
        elif self.provider == "openai":
            default_model = "gpt-3.5-turbo"
            self.service = OpenAILLMService(
                model_name=model_name or default_model,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        print(f"LLM Service initialized with provider: {self.provider}")
    
    def load_model(self) -> bool:
        """Load the model (if applicable)"""
        if hasattr(self.service, 'load_model') and callable(getattr(self.service, 'load_model')):
            return self.service.load_model()  # type: ignore
        return True
    
    def generate_response(self, memory: ConversationMemory, **kwargs) -> str:
        """Generate response using conversation memory"""
        return self.service.generate_response(memory, **kwargs)
    
    def generate_stream(self, memory: ConversationMemory, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response"""
        yield from self.service.generate_stream(memory, **kwargs)
    
    def is_loaded(self) -> bool:
        """Check if service is ready"""
        return self.service.is_loaded()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.service.cleanup()
    
    def update_generation_params(self, **kwargs) -> None:
        """Update generation parameters (if supported)"""
        if hasattr(self.service, 'update_generation_params') and callable(getattr(self.service, 'update_generation_params')):
            self.service.update_generation_params(**kwargs)  # type: ignore
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        return {
            "provider": self.provider,
            "model_name": self.service.model_name,
            "is_loaded": self.is_loaded(),
            "config": self.service.config
        }