import logging
from typing import Any, Dict, List, Optional
import httpx
from gateway.stt.base import BaseSTTEngine

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqSTTEngine(BaseSTTEngine):
    """
    High-performance Cloud STT using Groq's Whisper API with raw audio passthrough.
    """

    engine_name = "groq"

    def __init__(
        self,
        api_key: str,
        default_model: str = "whisper-large-v3-turbo",
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key.strip()
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Any:
        if not self.api_key:
            raise RuntimeError("Groq API key is not configured.")

        url = f"{GROQ_BASE_URL}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        use_model = model or self.default_model
        # Normalize model name for Groq if generic whisper requested
        if use_model in ("whisper-1", "groq", "groq-whisper"):
            use_model = self.default_model

        data: Dict[str, Any] = {
            "model": use_model,
            "response_format": response_format,
            "temperature": str(temperature),
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if timestamp_granularities:
            for tg in timestamp_granularities:
                data.setdefault("timestamp_granularities[]", []).append(tg)

        files = {
            "file": (filename, audio_bytes, "application/octet-stream"),
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, headers=headers, data=data, files=files)
            if resp.status_code != 200:
                logger.warning(
                    "Groq STT transcription failed [%d]: %s",
                    resp.status_code,
                    resp.text,
                )
                resp.raise_for_status()

            if response_format in ("json", "verbose_json"):
                return resp.json()
            return resp.text

    async def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Any:
        if not self.api_key:
            raise RuntimeError("Groq API key is not configured.")

        url = f"{GROQ_BASE_URL}/audio/translations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        use_model = model or self.default_model
        if use_model in ("whisper-1", "groq", "groq-whisper"):
            use_model = self.default_model

        data: Dict[str, Any] = {
            "model": use_model,
            "response_format": response_format,
            "temperature": str(temperature),
        }
        if prompt:
            data["prompt"] = prompt

        files = {
            "file": (filename, audio_bytes, "application/octet-stream"),
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, headers=headers, data=data, files=files)
            if resp.status_code != 200:
                logger.warning(
                    "Groq STT translation failed [%d]: %s",
                    resp.status_code,
                    resp.text,
                )
                resp.raise_for_status()

            if response_format in ("json", "verbose_json"):
                return resp.json()
            return resp.text

    async def health_check(self) -> bool:
        return bool(self.api_key)
