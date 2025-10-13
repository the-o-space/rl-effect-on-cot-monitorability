from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelProvider(ABC):
    """Abstract interface for model providers (OpenRouter, RunPod, local, etc.)."""
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the model provider.
        
        Args:
            model_id: Identifier for the model (e.g., "anthropic/claude-3.5-sonnet")
            **kwargs: Provider-specific configuration
        """
        self.model_id = model_id
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the model with metadata.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing:
                - response: Generated text
                - metadata: Provider-specific metadata (tokens, latency, etc.)
        """
        pass
