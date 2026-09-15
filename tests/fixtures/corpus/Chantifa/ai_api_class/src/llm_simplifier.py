import logging
import time
import backoff
from typing import Optional, Dict, List, Generator
import requests
import openai
import anthropic
from google.generativeai import GenerativeModel, configure as google_configure
from huggingface_hub import InferenceClient
from ratelimit import limits, sleep_and_retry

class LLMSimplifier:
    def __init__(self, api_key: str, provider: str = "openai",
                 temperature: float = 0.7, max_tokens: int = 512,
                 log_file: str = "llm_api.log", model: str = None):
        """Initialize the LLM API client with an API key, provider, and parameters."""
        self.provider = provider.lower()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model or self._default_model()
        self.api_base = {
            "xai": "https://api.x.ai/v1",
            "huggingface": "https://api-inference.huggingface.co/models"
        }.get(self.provider, None)
        self._setup_logging(log_file)
        self._setup_client()
        self.call_count = 0
        self.rate_limit_period = 60  # seconds
        self.rate_limit_calls = 50   # calls per period

    def _default_model(self) -> str:
        """Return default model for each provider."""
        defaults = {
            "openai": "gpt-4",
            "xai": "grok-3",
            "anthropic": "claude-3-opus-20240229",
            "google": "gemini-1.5-pro",
            "huggingface": "meta-llama/Llama-3-8b"
        }
        return defaults.get(self.provider, "gpt-4")

    def _setup_logging(self, log_file: str) -> None:
        """Configure file-based logging for API interactions."""
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LLMSimplifier for {self.provider} with model {self.model}")

    def _setup_client(self) -> None:
        """Set up the API client based on the provider."""
        if self.provider == "openai":
            openai.api_key = self.api_key
            self.client = openai
        elif self.provider == "xai":
            self.client = requests.Session()
            self.client.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            google_configure(api_key=self.api_key)
            self.client = GenerativeModel(model_name=self.model)
        elif self.provider == "huggingface":
            self.client = InferenceClient(model=self.model, token=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Extend _setup_client for new providers.")

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, requests.exceptions.RequestException, anthropic.APIError), max_tries=3)
    @sleep_and_retry
    @limits(calls=50, period=60)
    def generate_text(self, prompt: str,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> str:
        """Generate text from a prompt with customizable parameters."""
        self.call_count += 1
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        try:
            self.logger.info(f"Sending request {self.call_count}: {prompt[:50]}...")
            if self.provider == "openai":
                response = self.client.Completion.create(
                    model=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
                text = response.choices[0].text.strip()
            elif self.provider == "xai":
                response = self.client.post(
                    f"{self.api_base}/completions",
                    json={"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["text"].strip()
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
            elif self.provider == "google":
                response = self.client.generate_content(
                    contents=prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
                )
                text = response.text.strip()
            elif self.provider == "huggingface":
                response = self.client.text_generation(
                    prompt, max_new_tokens=max_tokens, temperature=temperature
                )
                text = response.strip()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            self.logger.info(f"Received response for request {self.call_count}")
            return text
        except Exception as e:
            self.logger.error(f"Error in request {self.call_count}: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=50, period=60)
    def batch_generate(self, prompts: List[str],
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> List[str]:
        """Process multiple prompts in a batch, respecting rate limits."""
        results = []
        self.logger.info(f"Starting batch processing for {len(prompts)} prompts")
        for i, prompt in enumerate(prompts, 1):
            try:
                result = self.generate_text(prompt, temperature, max_tokens)
                results.append(result)
                self.logger.info(f"Completed prompt {i}/{len(prompts)}")
            except Exception as e:
                self.logger.error(f"Failed prompt {i}: {str(e)}")
                results.append(f"Error: {str(e)}")
        self.logger.info(f"Finished batch processing: {len(results)} results")
        return results

    def stream_text(self, prompt: str,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Handle streaming responses for real-time applications."""
        self.call_count += 1
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        try:
            self.logger.info(f"Streaming request {self.call_count}: {prompt[:50]}...")
            if self.provider == "openai":
                response = self.client.Completion.create(
                    model=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=True
                )
                for chunk in response:
                    if chunk.choices and chunk.choices[0].text:
                        text = chunk.choices[0].text.strip()
                        self.logger.info(f"Received stream chunk for request {self.call_count}")
                        yield text
            elif self.provider == "xai":
                response = self.client.post(
                    f"{self.api_base}/completions",
                    json={"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "stream": True},
                    stream=True
                )
                response.raise_for_status()
                for chunk in response.iter_lines():
                    if chunk:
                        text = chunk.decode("utf-8").strip()
                        if text:
                            self.logger.info(f"Received stream chunk for request {self.call_count}")
                            yield text
            elif self.provider == "anthropic":
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        self.logger.info(f"Received stream chunk for request {self.call_count}")
                        yield text.strip()
            elif self.provider == "google":
                response = self.client.generate_content(
                    contents=prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    stream=True
                )
                for chunk in response:
                    text = chunk.text.strip()
                    if text:
                        self.logger.info(f"Received stream chunk for request {self.call_count}")
                        yield text
            elif self.provider == "huggingface":
                # Hugging Face Inference API does not support streaming, fallback to non-streaming
                text = self.generate_text(prompt, temperature, max_tokens)
                self.logger.info(f"Non-streaming fallback for request {self.call_count}")
                yield text
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            self.logger.error(f"Error in streaming request {self.call_count}: {str(e)}")
            raise