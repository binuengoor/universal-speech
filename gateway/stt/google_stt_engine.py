import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from gateway.stt.base import BaseSTTEngine

logger = logging.getLogger(__name__)


class GoogleSTTEngine(BaseSTTEngine):
    """
    Cloud STT using Google Cloud Speech-to-Text API.
    """

    engine_name = "google-cloud"

    def __init__(
        self,
        credentials_path: Optional[str] = "credentials/google-service-account.json",
        language_code: str = "en-US",
        timeout_seconds: float = 10.0,
    ):
        self.credentials_path = credentials_path
        self.default_model = "google-cloud"
        self.language_code = language_code
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        from google.cloud import speech

        if self.credentials_path and Path(self.credentials_path).exists():
            self._client = speech.SpeechClient.from_service_account_file(self.credentials_path)
        else:
            self._client = speech.SpeechClient()
        return self._client

    def _sync_transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        from google.cloud import speech

        client = self._get_client()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code=language or self.language_code,
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio, timeout=self.timeout_seconds)
        transcript_parts = [result.alternatives[0].transcript for result in response.results if result.alternatives]
        return " ".join(transcript_parts).strip()

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
        full_text = await asyncio.to_thread(self._sync_transcribe, audio_bytes, language)

        if response_format == "text":
            return full_text
        return {"text": full_text}

    async def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Any:
        # Standard recognize with English language target
        full_text = await asyncio.to_thread(self._sync_transcribe, audio_bytes, "en-US")
        if response_format == "text":
            return full_text
        return {"text": full_text}

    async def health_check(self) -> bool:
        if not self.credentials_path:
            return False
        return Path(self.credentials_path).exists()
