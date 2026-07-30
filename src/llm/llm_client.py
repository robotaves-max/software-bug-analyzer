import os
import re
import json
import time
from typing import Dict, Any, Optional
from src.utils import logger

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

class LLMClient:
    """Abstract multi-provider LLM client supporting Groq, OpenAI, Gemini, and Ollama."""

    def __init__(self, provider: str = "groq", model: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.provider = (provider or "groq").lower()
        self.config = config or {}
        self.temperature = float(self.config.get("temperature", 0.2))
        self.max_tokens = int(self.config.get("max_tokens", 1500))

        if self.provider == "groq":
            self.model = model or self.config.get("model", "llama-3.1-70b-versatile")
        elif self.provider == "openai":
            self.model = model or self.config.get("model", "gpt-4o-mini")
        elif self.provider == "gemini":
            self.model = model or self.config.get("model", "gemini-1.5-flash")
        elif self.provider == "ollama":
            self.model = model or self.config.get("model", "llama3")
        else:
            self.model = model or "llama-3.1-70b-versatile"

        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes provider specific SDK client."""
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if GROQ_AVAILABLE and api_key:
                self.client = Groq(api_key=api_key)
                logger.info(f"Initialized Groq client with model {self.model}.")
            else:
                logger.warning("Groq API key or SDK missing.")

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if OPENAI_AVAILABLE and api_key:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model {self.model}.")
            else:
                logger.warning("OpenAI API key or SDK missing.")

        elif self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if GEMINI_AVAILABLE and api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Initialized Gemini client with model {self.model}.")
            else:
                logger.warning("Google Gemini API key or SDK missing.")

        elif self.provider == "ollama":
            if OLLAMA_AVAILABLE:
                self.client = ollama
                logger.info(f"Initialized Ollama client with model {self.model}.")
            else:
                logger.warning("Ollama library missing.")

    def is_available(self) -> bool:
        """Returns True if client SDK and API credentials are live."""
        return self.client is not None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generates raw text completion across active LLM provider."""
        if not self.is_available():
            raise RuntimeError(f"LLM Provider '{self.provider}' is not available or missing API credentials.")

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            response = self.client.generate_content(full_prompt, generation_config=generation_config)
            return response.text

        elif self.provider == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            return response['message']['content']

        else:
            raise NotImplementedError(f"Unsupported provider: {self.provider}")

    def generate_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generates response and enforces JSON parsing with automatic retry on malformed JSON."""
        attempts = 0
        current_user_prompt = user_prompt

        while attempts < max_retries:
            attempts += 1
            try:
                raw_text = self.generate(system_prompt, current_user_prompt)
                cleaned_text = raw_text.strip()

                if "```" in cleaned_text:
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
                    if match:
                        cleaned_text = match.group(1).strip()

                first_brace = cleaned_text.find("{")
                last_brace = cleaned_text.rfind("}")

                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    cleaned_text = cleaned_text[first_brace:last_brace+1]

                parsed_json = json.loads(cleaned_text)
                return parsed_json

            except Exception as e:
                logger.warning(f"JSON parsing attempt {attempts}/{max_retries} failed: {e}")
                if attempts < max_retries:
                    current_user_prompt = f"{user_prompt}\n\nCRITICAL FIX: Your previous response was invalid JSON ({e}). Output ONLY valid JSON starting with {{ and ending with }}."
                    time.sleep(1)

        raise ValueError(f"Failed to obtain valid JSON from {self.provider} LLM after {max_retries} attempts.")

    def test_connection(self) -> bool:
        """Tests basic LLM API connection."""
        if not self.is_available():
            return False
        try:
            res = self.generate("You are a helpful assistant.", "Say 'OK'")
            return "OK" in res or len(res) > 0
        except Exception:
            return False
