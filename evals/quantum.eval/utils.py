"""Utility functions for Quantum Reasoning eval"""

import os
import yaml
import json
import re
from typing import Any, Optional, Dict, List
from pathlib import Path
import requests
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class Config:
    """Load and manage configuration from YAML"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        """
        Initialize config from file or dictionary
        
        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary to use as config (for programmatic setup)
        """
        if config_dict is not None:
            self.config = config_dict
            self.config_path = None
        elif config_path is not None:
            self.config_path = Path(config_path)
            self.config = self._load_yaml(config_path)
        else:
            raise Exception("Either config_path or config_dict should be passed")

 
    def _load_yaml(self, path: str) -> dict:
        """Load YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get config value with dot notation (e.g., 'api.base_url')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    @property
    def actor_models(self):
        return self.get('models.actor_models', [])
    
    @property
    def judge_model(self):
        return self.get('models.judge_model')
    
    @property
    def api_base_url(self):
        return self.get('api.base_url', 'https://openrouter.ai/api/v1')


class PromptLoader:
    """Load prompts from YAML files or dictionaries"""

    def __init__(self, prompts_dir: Optional[str] = "prompts", prompts_dict: Optional[dict] = None):
        """
        Initialize from directory or dictionary

        Args:
            prompts_dir: Directory containing actor.yaml and judges.yaml (default: "prompts")
            prompts_dict: Dictionary with 'actor' and 'judge' keys (overrides prompts_dir if provided)
        """
        if prompts_dict is not None:
            self.actor_prompts = prompts_dict.get('actor', {})
            self.judge_prompts = prompts_dict.get('judge', {})
        else:
            # Default to prompts_dir (with default value "prompts")
            self.prompts_dir = Path(prompts_dir)
            self.actor_prompts = self._load_yaml(self.prompts_dir / "actor.yaml")
            self.judge_prompts = self._load_yaml(self.prompts_dir / "judges.yaml")
           
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML file"""
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_actor_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get actor prompt with formatting"""
        template = self.actor_prompts.get(prompt_type)
        if template is None:
            raise ValueError(f"Unknown actor prompt type: {prompt_type}")
        return template.format(**kwargs)
    
    def get_judge_prompt(self, judge_type: str, **kwargs) -> str:
        """Get judge prompt with formatting"""
        template = self.judge_prompts.get(judge_type)
        if template is None:
            raise ValueError(f"Unknown judge prompt type: {judge_type}")
        return template.format(**kwargs)


class OpenRouterClient:
    """Wrapper for OpenRouter API with rate limiting and error handling"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _wait_for_rate_limit(self):
        """Simple rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def complete(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = None,
        reasoning_config: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Get completion from OpenRouter using OpenAI SDK
        
        Args:
            model: Model name (e.g., 'anthropic/claude-sonnet-4')
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            reasoning_config: Dict with reasoning params like:
                {'effort': 'high'} or {'max_tokens': 2000, 'exclude': False}
            **kwargs: Additional API parameters
            
        Returns:
            Dict with 'output' (content) and 'reasoning' (if available)
        """
    
        self._wait_for_rate_limit()
        
        try:
            extra_body = kwargs.pop("extra_body", {}) or {}
            if reasoning_config:
                rc = reasoning_config.copy()

                # OpenRouter expects a top-level `reasoning` object.
                # Common fields supported across providers include:
                #   effort: "low" | "medium" | "high"
                #   max_tokens: int (budget for reasoning tokens)
                #   exclude: bool (hide thinking in the response)
                # `enabled` is not part of the spec — drop it if present.
                rc.pop("enabled", None)

                # If caller passed max_tokens (response), also mirror into reasoning.max_tokens if provided in rc
                # (Some models look at reasoning.max_tokens for thinking budget.)
                if "max_tokens" not in rc and max_tokens is not None:
                    # optional: don’t set this if you want strict separation of output vs reasoning budgets
                    pass

                extra_body["reasoning"] = rc

                # Many OpenRouter models require this flag to return thinking text.
                # If user set exclude=True, we *don’t* include reasoning text.
                if "exclude" in rc:
                    extra_body["include_reasoning"] = not bool(rc["exclude"])
                else:
                    extra_body.setdefault("include_reasoning", True)

            # Build request
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,                 # do NOT inject reasoning here
                temperature=temperature,
                max_tokens=max_tokens,             # output-token cap
                extra_body=extra_body,             # pass reasoning + include_reasoning here
                **kwargs
            )

            # Extract primary content and optional reasoning (varies by model)
            choice = completion.choices[0].message
            return {
                "output": choice.content,
                "reasoning": getattr(choice, "reasoning", None) or getattr(completion, "reasoning", None),
                "full_response": completion,
            }

            
        except Exception as e:
            raise Exception(f"OpenRouter API call failed for model {model}: {str(e)}")

    
    def get_text_response(self, *args, **kwargs) -> Dict[str, Optional[str]]:
        """
        Get text response with reasoning (if available)

        Returns:
            Dict with 'output' (main content) and 'reasoning' (reasoning trace if available)
        """
        result = self.complete(*args, **kwargs)
        return {
            'output': result['output'],
            'reasoning': result['reasoning']
        }

    async def complete_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = None,
        reasoning_config: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Async version of complete - runs in thread pool to avoid blocking

        Args:
            Same as complete()

        Returns:
            Same as complete()
        """
        # Run the synchronous complete method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default executor
            lambda: self.complete(model, messages, temperature, max_tokens, reasoning_config, **kwargs)
        )

    async def get_text_response_async(self, *args, **kwargs) -> Dict[str, Optional[str]]:
        """
        Async version of get_text_response

        Returns:
            Dict with 'output' (main content) and 'reasoning' (reasoning trace if available)
        """
        result = await self.complete_async(*args, **kwargs)
        return {
            'output': result['output'],
            'reasoning': result['reasoning']
        }

    async def batch_get_text_responses(
        self,
        requests: List[Dict],
        max_concurrent: int = 5
    ) -> List[Dict[str, Optional[str]]]:
        """
        Process multiple requests in parallel with concurrency limit

        Args:
            requests: List of dicts, each containing args for get_text_response
                      Format: {'model': ..., 'messages': ..., 'temperature': ..., etc}
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of results in same order as requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(req_dict):
            async with semaphore:
                return await self.get_text_response_async(**req_dict)

        # Create all tasks
        tasks = [limited_request(req) for req in requests]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'output': None,
                    'reasoning': None,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        return processed_results



def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Extract the final button sequence from a model response.
    Returns a normalized string like 'A → B → C', or None if not found.
    Takes the LAST match to handle cases where the model revises its answer.
    """
    text = response
    
    # 1) Strip common LaTeX wrappers
    text = re.sub(r'\\\[|\\\]', '', text)  # Remove \[ and \]
    text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)  # Extract from \boxed{}
    
    # 2) Normalize ALL delimiter variants to ' → '
    # LaTeX arrows
    text = re.sub(r'\\to\b', '→', text)  # \to
    text = re.sub(r'\\\\to\b', '→', text)  # \\to
    text = re.sub(r'\\rightarrow', '→', text)  # \rightarrow
    
    # ASCII arrows
    text = re.sub(r'\s*-\s*>\s*', ' → ', text)  # -> or - >
    
    # Unicode arrows (normalize spacing)
    text = re.sub(r'\s*→\s*', ' → ', text)
    
    # Other common separators (comma, pipe, etc.)
    text = re.sub(r'\s*[,|]\s*', ' → ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 3) Patterns to match sequences
    patterns = [
        # After normalization, primary pattern with arrows
        r'(?:final answer|answer|solution)\s*(?:is|:)?\s*\*?\*?\s*([A-C](?:\s*→\s*[A-C])+)',
        
        # Bare sequence with arrows
        r'\b([A-C](?:\s*→\s*[A-C]){2,})\b',  # at least 3 buttons
        
        # Fallback: space-separated sequence (at least 3 letters)
        r'\b([A-C](?:\s+[A-C]){2,})\b',
    ]
    
    matches = []
    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        matches.extend(found)
    
    if not matches:
        return None
    
    # Take the LAST match (in case model revises its answer)
    seq = matches[-1].strip()
    
    # If it's space-separated without arrows, convert to arrows
    if '→' not in seq and re.fullmatch(r'[A-C](?:\s+[A-C])+', seq, flags=re.IGNORECASE):
        seq = re.sub(r'\s+', ' → ', seq)
    
    # Final cleanup
    seq = re.sub(r'\s*→\s*', ' → ', seq)
    
    return seq.upper()


def normalize_path(path: str) -> List[str]:
    """
    Normalize a path string to list of buttons
    
    Args:
        path: Path string like "A → B → C" or "A->B->C"
        
    Returns:
        List of button names like ['A', 'B', 'C']
    """
    # Extract just the letters A, B, C
    return [c.upper() for c in re.findall(r'[A-C]', path.upper())]


def paths_match(path1: str, path2: str) -> bool:
    """
    Check if two paths are equivalent
    
    Args:
        path1: First path string
        path2: Second path string
        
    Returns:
        True if paths match (after normalization)
    """
    return normalize_path(path1) == normalize_path(path2)


def format_path(path: List[str]) -> str:
    """
    Format a path list as a string
    
    Args:
        path: List of button names like ['A', 'B', 'C']
        
    Returns:
        Formatted string like "A → B → C"
    """
    return ' → '.join(path)


# Convenience function for quick setup
def quick_setup(api_key: Optional[str] = None) -> tuple:
    """
    Quick setup for Jupyter notebooks
    
    Returns:
        (config, prompt_loader, client)
    """
    config = Config()
    prompt_loader = PromptLoader()
    client = OpenRouterClient(api_key=api_key)
    
    return config, prompt_loader, client