from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSTTEngine(ABC):
    """
    Abstract base class for all Speech-to-Text engine drivers.
    """

    engine_name: str
    default_model: str

    @abstractmethod
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
        """
        Transcribe audio to text.
        Returns text, dict (for json/verbose_json), or formatted string (for srt/vtt).
        """
        pass

    @abstractmethod
    async def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Any:
        """
        Translate foreign audio to English text.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the engine is ready and operational.
        """
        pass
