import asyncio
import io
import logging
import os
import re
from typing import AsyncGenerator, List, Optional, Tuple
import edge_tts

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)

# e.g. en-US-AriaNeural, en-US-AvaMultilingualNeural, zh-CN-liaoning-XiaobeiNeural
_EDGE_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Za-z]{2,4}(-[A-Za-z]+)*-[A-Za-z]+Neural$")


class EdgeTTSEngine(BaseTTSEngine):
    engine_name = "edge-tts"
    native_format = "mp3"
    min_speed = 0.5
    max_speed = 3.0

    def __init__(self, default_voice: str = "en-US-AriaNeural", timeout_seconds: float = 10.0):
        self.default_voice = default_voice
        # Idle timeout: maximum silence between two chunks from the Edge service.
        self.timeout_seconds = timeout_seconds
        self._cached_voices: Optional[List[VoiceObject]] = None
        self._proxy = os.getenv("EDGE_TTS_PROXY") or None

    def is_valid_voice(self, voice: str) -> bool:
        if self._cached_voices:
            return any(v.id == voice for v in self._cached_voices)
        return bool(_EDGE_VOICE_RE.match(voice))

    def _format_rate(self, speed: float) -> str:
        pct = int(round((self.clamp_speed(speed) - 1.0) * 100))
        return f"{pct:+d}%"

    def _communicate(self, text: str, voice: str, rate: str) -> "edge_tts.Communicate":
        if self._proxy:
            return edge_tts.Communicate(text=text, voice=voice, rate=rate, proxy=self._proxy)
        return edge_tts.Communicate(text=text, voice=voice, rate=rate)

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        rate_str = self._format_rate(speed)

        # If zero-transcode (target format is native mp3), stream directly chunk-by-chunk
        if target_fmt == self.native_format:
            communicate = self._communicate(text, target_voice, rate_str)
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
            except Exception as e:
                logger.error("Edge-TTS streaming failed: %s", str(e))
                raise
        else:
            # Need conversion: accumulate native MP3 and convert
            full_mp3, _ = await self.synthesize_bytes(text, target_voice, speed, "mp3")
            converted = await convert_audio_ffmpeg(full_mp3, "mp3", target_fmt)
            yield converted

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        target_voice = voice or self.default_voice
        target_fmt = (response_format or self.native_format).lower()
        rate_str = self._format_rate(speed)

        communicate = self._communicate(text, target_voice, rate_str)
        buffer = io.BytesIO()

        stream = communicate.stream()
        try:
            while True:
                # Idle timeout per chunk, so long inputs that stream steadily never time out.
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=self.timeout_seconds)
                except StopAsyncIteration:
                    break
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
        except asyncio.TimeoutError:
            logger.error("Edge-TTS stalled for more than %s seconds", self.timeout_seconds)
            raise TimeoutError(f"Edge-TTS stalled for more than {self.timeout_seconds}s")
        except Exception as e:
            logger.error("Edge-TTS synthesis error: %s", str(e))
            raise

        mp3_bytes = buffer.getvalue()
        if not mp3_bytes:
            raise RuntimeError("Edge-TTS returned empty audio data")

        if target_fmt == self.native_format:
            return mp3_bytes, "mp3"

        converted = await convert_audio_ffmpeg(mp3_bytes, "mp3", target_fmt)
        return converted, target_fmt

    async def get_voices(self) -> List[VoiceObject]:
        if self._cached_voices is not None:
            return self._cached_voices

        try:
            voices_data = await edge_tts.list_voices()
            voices: List[VoiceObject] = []
            for v in voices_data:
                voices.append(
                    VoiceObject(
                        id=v.get("ShortName", ""),
                        name=v.get("FriendlyName", v.get("ShortName", "")),
                        engine=self.engine_name,
                        language=v.get("Locale", "en-US"),
                        gender=v.get("Gender", None),
                    )
                )
            self._cached_voices = voices
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Edge-TTS online voices: %s", str(e))
            # Fallback default list
            return [
                VoiceObject(
                    id="en-US-AriaNeural",
                    name="Microsoft Aria (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Female",
                ),
                VoiceObject(
                    id="en-US-GuyNeural",
                    name="Microsoft Guy (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Male",
                ),
                VoiceObject(
                    id="en-US-JennyNeural",
                    name="Microsoft Jenny (Neural)",
                    engine=self.engine_name,
                    language="en-US",
                    gender="Female",
                ),
            ]

    async def health_check(self) -> bool:
        # Edge is a remote service; availability is tracked by the router's
        # circuit breaker rather than probed on every health request.
        return True
