import asyncio
import logging
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from google.cloud import texttospeech

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)

_GOOGLE_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-[A-Za-z0-9]+(-[A-Za-z0-9]+)*$")

# Google rejects SynthesisInput.text above 5000 bytes; keep a safety margin.
MAX_INPUT_BYTES = 4800


def split_text_for_google(text: str, max_bytes: int = MAX_INPUT_BYTES) -> List[str]:
    """Split text into chunks under `max_bytes` (UTF-8), preferring sentence then word boundaries."""
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    chunks: List[str] = []
    current = ""

    def _flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for piece in pieces:
        if not piece:
            continue
        candidate = f"{current} {piece}" if current else piece
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
            continue
        _flush()
        if len(piece.encode("utf-8")) <= max_bytes:
            current = piece
            continue
        # A single overlong sentence: fall back to word boundaries.
        for word in piece.split():
            candidate = f"{current} {word}" if current else word
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
            else:
                _flush()
                current = word
    _flush()
    return chunks


class GoogleTTSEngine(BaseTTSEngine):
    engine_name = "google-cloud"
    native_format = "mp3"
    min_speed = 0.25
    max_speed = 4.0

    def is_valid_voice(self, voice: str) -> bool:
        if self._cached_voices and self.client is not None:
            return any(v.id == voice for v in self._cached_voices)
        return bool(_GOOGLE_VOICE_RE.match(voice)) and not voice.endswith("Neural")

    def __init__(
        self,
        credentials_path: Optional[str] = "credentials/google-service-account.json",
        default_voice: str = "en-US-Neural2-F",
        timeout_seconds: Optional[float] = 8.0,
    ):
        self.credentials_path = credentials_path
        self.default_voice = default_voice
        self.timeout_seconds = timeout_seconds
        self._cached_voices: Optional[List[VoiceObject]] = None
        self._voice_lang_map: Dict[str, str] = {}
        self.client: Optional[texttospeech.TextToSpeechClient] = None

        self._init_client()

    def _init_client(self) -> None:
        try:
            if self.credentials_path and Path(self.credentials_path).exists():
                self.client = texttospeech.TextToSpeechClient.from_service_account_json(
                    self.credentials_path
                )
                logger.info("Google Cloud TTS client initialized from %s", self.credentials_path)
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).exists():
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Google Cloud TTS client initialized via GOOGLE_APPLICATION_CREDENTIALS")
            else:
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Google Cloud TTS client initialized with default credentials")
        except Exception as e:
            logger.warning("Failed to initialize Google Cloud TTS client: %s", str(e))
            self.client = None

    def _extract_language_code(self, voice_name: str) -> str:
        if voice_name in self._voice_lang_map:
            return self._voice_lang_map[voice_name]
        parts = voice_name.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "en-US"

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        audio_bytes, _ = await self.synthesize_bytes(
            text=text,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        if self.client is None:
            raise RuntimeError("Google Cloud TTS client is not initialized or credentials missing")

        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        language_code = self._extract_language_code(target_voice)
        clamped_speed = self.clamp_speed(speed)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=target_voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=clamped_speed,
        )

        def _call_synthesize(chunk: str) -> bytes:
            response = self.client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=chunk),
                voice=voice_params,
                audio_config=audio_config,
                timeout=self.timeout_seconds,
            )
            return response.audio_content

        chunks = split_text_for_google(text)
        try:
            parts: List[bytes] = []
            for chunk in chunks:
                if self.timeout_seconds:
                    async with asyncio.timeout(self.timeout_seconds):
                        parts.append(await asyncio.to_thread(_call_synthesize, chunk))
                else:
                    parts.append(await asyncio.to_thread(_call_synthesize, chunk))
            # MP3 frames are self-delimiting, so concatenating segments yields a playable stream.
            audio_bytes = b"".join(parts)
        except asyncio.TimeoutError:
            logger.error("Google Cloud TTS timed out after %s seconds", self.timeout_seconds)
            raise TimeoutError(f"Google Cloud TTS timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error("Google Cloud TTS synthesis error: %s", str(e))
            raise

        if not audio_bytes:
            raise RuntimeError("Google Cloud TTS returned empty audio data")

        # Zero-transcode fast path
        if target_fmt == self.native_format:
            return audio_bytes, "mp3"

        converted = await convert_audio_ffmpeg(audio_bytes, "mp3", target_fmt)
        return converted, target_fmt

    def _get_fallback_voices(self) -> List[VoiceObject]:
        fallback = [
            VoiceObject(
                id="en-US-Neural2-F",
                name="Google en-US (Neural2 Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Neural2-D",
                name="Google en-US (Neural2 Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
            VoiceObject(
                id="en-US-Journey-F",
                name="Google en-US (Journey Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Journey-D",
                name="Google en-US (Journey Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
            VoiceObject(
                id="en-US-Studio-O",
                name="Google en-US (Studio Female)",
                engine=self.engine_name,
                language="en-US",
                gender="Female",
            ),
            VoiceObject(
                id="en-US-Wavenet-D",
                name="Google en-US (WaveNet Male)",
                engine=self.engine_name,
                language="en-US",
                gender="Male",
            ),
        ]
        for v in fallback:
            self._voice_lang_map[v.id] = v.language
        return fallback

    async def get_voices(self) -> List[VoiceObject]:
        if self._cached_voices is not None:
            return self._cached_voices

        if self.client is None:
            fallback = self._get_fallback_voices()
            self._cached_voices = fallback
            return fallback

        try:
            response = await asyncio.to_thread(self.client.list_voices)
            voices: List[VoiceObject] = []
            for v in response.voices:
                lang = v.language_codes[0] if v.language_codes else "en-US"
                try:
                    gender_enum = texttospeech.SsmlVoiceGender(v.ssml_gender)
                    gender_str = gender_enum.name.capitalize()
                    if gender_str == "Ssml_voice_gender_unspecified":
                        gender = None
                    else:
                        gender = gender_str
                except Exception:
                    gender = None

                sample_rate = getattr(v, "natural_sample_rate_hertz", None)
                voices.append(
                    VoiceObject(
                        id=v.name,
                        name=v.name,
                        engine=self.engine_name,
                        language=lang,
                        gender=gender,
                        sample_rate=sample_rate,
                    )
                )
                self._voice_lang_map[v.name] = lang

            self._cached_voices = voices
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Google Cloud voices online: %s", str(e))
            fallback = self._get_fallback_voices()
            self._cached_voices = fallback
            return fallback

    async def health_check(self) -> bool:
        return self.client is not None
