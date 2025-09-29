import os
import time
from typing import Any, Dict, Optional
import requests

from .provider import ModelProvider


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider implementation."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            site_url: Optional site URL for OpenRouter rankings
            site_name: Optional site name for OpenRouter rankings
            **kwargs: Additional configuration
        """
        super().__init__(model_id, **kwargs)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required (set OPENROUTER_API_KEY env var)")
        
        self.site_url = site_url
        self.site_name = site_name
    
    def _make_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text response from OpenRouter."""
        result = self.generate_with_metadata(prompt, temperature, max_tokens, **kwargs)
        return result["response"]
    
    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with metadata.
        
        Returns:
            Dict with keys:
                - response: Generated text
                - metadata: Dict with tokens_used, latency_ms, model, etc.
        """
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        start_time = time.time()
        response = requests.post(
            self.BASE_URL,
            headers=self._make_headers(),
            json=payload,
            timeout=60
        )
        latency_ms = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        # Extract response text
        content = data["choices"][0]["message"]["content"]
        
        # Build metadata
        usage = data.get("usage", {})
        metadata = {
            "latency_ms": latency_ms,
            "model": data.get("model", self.model_id),
            "tokens_prompt": usage.get("prompt_tokens", 0),
            "tokens_completion": usage.get("completion_tokens", 0),
            "tokens_total": usage.get("total_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
        
        return {
            "response": content,
            "metadata": metadata
        }
