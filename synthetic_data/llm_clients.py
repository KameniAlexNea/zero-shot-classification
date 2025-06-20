import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from groq import Groq
from loguru import logger
from ollama import Client


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.8
    max_tokens: int = 2048
    top_p: float = 0.95
    max_retries: int = 3
    retry_delay: float = 1.0
    repeat_penalty: float = 1.1


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts: {e}"
                    )
                    raise e

                delay = self.config.retry_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)


class GroqClient(BaseLLMClient):
    """Groq cloud client with retry capability."""

    def __init__(self, config: LLMConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.client = Groq(api_key=api_key) if api_key else Groq()
        logger.info(f"Initialized Groq client with model: {config.model}")

    def _make_groq_request(self, prompt: str) -> str:
        """Make a single request to Groq API."""
        temperature = random.uniform(0.1, 0.3)
        seed = random.randint(1, 100000)
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            seed=seed,
            stream=False,
            # stop=None,
            timeout=30
        )

        return completion.choices[0].message.content

    def generate_text(self, prompt: str) -> str:
        """Generate text using Groq with retry logic."""
        return self._retry_with_backoff(self._make_groq_request, prompt)


class OllamaClient(BaseLLMClient):
    """Ollama local client with retry capability."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = Client()
        logger.info(f"Initialized Ollama client with model: {config.model}")

    def _make_ollama_request(self, prompt: str) -> str:
        """Make a single request to Ollama."""
        # Randomize for diversity
        temperature = random.uniform(0.7, 0.9)
        seed = random.randint(1, 100000)

        response = self.client.generate(
            model=self.config.model,
            prompt=prompt,
            options={
                "seed": seed,
                "temperature": temperature,
                "repeat_penalty": self.config.repeat_penalty,
                "max_tokens": self.config.max_tokens,
            },
        )

        return response.response.strip()

    def generate_text(self, prompt: str) -> str:
        """Generate text using Ollama with retry logic."""
        return self._retry_with_backoff(self._make_ollama_request, prompt)


def create_llm_client(
    provider: str, model: str, api_key: Optional[str] = None
) -> BaseLLMClient:
    """Factory function to create LLM clients."""
    config = LLMConfig(model=model)

    if provider.lower() == "groq":
        return GroqClient(config, api_key)
    elif provider.lower() == "ollama":
        return OllamaClient(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
