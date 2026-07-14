import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from app.infra.logging import logger


OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def _ollama_get(url: str, timeout: int = 5) -> dict:
    resp = urllib.request.urlopen(url, timeout=timeout)
    return json.loads(resp.read())


def _ollama_post(url: str, data: dict, timeout: int = 300) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


class LLMRunner:
    def __init__(
        self,
        enabled: bool = False,
        model_path: Optional[Path] = None,
        n_ctx: int = 2048,
        n_threads: int = 2,
        max_tokens: int = 512,
        temperature: float = 0.1,
        use_gpu: bool = False,
        ollama_base_url: str = OLLAMA_DEFAULT_BASE,
        ollama_model: str = "llama3:8b",
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_gpu = use_gpu
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self._llm = None

    def _check_ollama(self) -> bool:
        try:
            data = _ollama_get(f"{self.ollama_base_url}/api/tags", timeout=5)
            models = [m["name"] for m in data.get("models", [])]
            if self.ollama_model not in models:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self.ollama_model,
                    ", ".join(models),
                )
                return False
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            logger.warning(
                "Ollama not reachable at %s. Is the Ollama service running?",
                self.ollama_base_url,
            )
            return False
        except Exception as e:
            logger.warning("Ollama check failed: %s", str(e))
            return False

    def load(self) -> None:
        if not self.enabled:
            logger.info("LLM is disabled. Enable via settings or .env file.")
            return
        if self._llm is not None:
            return
        if not self._check_ollama():
            logger.warning(
                "Ollama unavailable or model missing. "
                "Run: ollama pull %s",
                self.ollama_model,
            )
            self.enabled = False
            return
        logger.info(
            "LLM ready (Ollama: %s, model: %s)",
            self.ollama_base_url,
            self.ollama_model,
        )
        self._llm = True

    def generate(self, prompt: str, stop: Optional[list[str]] = None) -> Optional[str]:
        if not self.enabled:
            return None
        if self._llm is None:
            self.load()
        if self._llm is None:
            return None
        try:
            body = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            if stop:
                body["options"]["stop"] = stop
            data = _ollama_post(
                f"{self.ollama_base_url}/api/generate",
                body,
                timeout=300,
            )
            text = data.get("response", "").strip()
            return text
        except urllib.error.HTTPError as e:
            logger.error("LLM generation returned %s: %s", e.code, e.read().decode())
            return None
        except Exception as e:
            logger.error("LLM generation failed: %s", str(e))
            return None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None
