import asyncio
import logging
import struct
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)

# Phonon (Gradium) English voice catalogue, pinned from
# https://huggingface.co/gradium/phonon-voices @ da6c9ec0d47be8df5c87c730a3aa97187f3229c3
# (name, locale, gender). The server owns the authoritative list; this is the
# offline copy used for validation and for GET /v1/voices without a round trip.
PHONON_EN_VOICES: List[Tuple[str, str, Optional[str]]] = [
    ("Zoey", "en-US", "female"),
    ("Juniper", "en-US", "female"),
    ("Everly", "en-US", "female"),
    ("Harper", "en-US", "female"),
    ("Hadley", "en-US", "female"),
    ("Delia", "en-US", "female"),
    ("Easton", "en-US", "male"),
    ("Micah", "en-US", "male"),
    ("Trevor", "en-US", "male"),
    ("Russell", "en-US", "male"),
    ("Preston", "en-US", "male"),
    ("Miles", "en-US", "male"),
    ("Maeve", "en-GB", "female"),
    ("Freya", "en-GB", "female"),
    ("Pippa", "en-GB", "female"),
    ("Elodie-Rose", "en-GB", "female"),
    ("Imogen", "en-GB", "female"),
    ("Lottie", "en-GB", "female"),
    ("Alfie", "en-GB", "male"),
    ("Hugh", "en-GB", "male"),
    ("Edwin", "en-GB", "male"),
    ("Spencer", "en-GB", "male"),
    ("Toby", "en-GB", "male"),
    ("Teddy", "en-GB", "male"),
    ("Bryony", "en-GB", "female"),
    ("Captain Silas Blackwater", "en-US", None),
    ("Morwenna of the Deep", "en-US", None),
    ("Old Mortlake", "en-US", None),
    ("Azgorath", "en-US", None),
    ("Fennick the Young Elf", "en-US", None),
    ("The Thing Under the Stairs", "en-US", None),
    ("Vyrmathrax the Ancient", "en-US", None),
    ("Grub the Ogre", "en-US", None),
    ("Master Aldric", "en-US", None),
    ("The Snow Queen", "en-US", None),
    ("Sir Bramwell", "en-US", None),
    ("Fairy Godmother", "en-US", None),
]

# Formats ptts-openai-server produces natively; anything else is fetched as WAV
# and transcoded locally, so the gateway never asks for a format the server rejects.
PHONON_NATIVE_FORMATS = ("wav", "mp3", "opus")


def normalize_wav_header(data: bytes) -> bytes:
    """
    Rewrite the RIFF/data size fields of a WAV buffer with the real lengths.

    ptts-openai-server streams, so it writes the 0xFFFFFFFF "length unknown"
    sentinel in both the RIFF and `data` size fields. Players that read the
    header literally then report a bogus duration (or refuse to seek), which is
    exactly what a reader UI or a Home Assistant media player does. The audio
    frames themselves are correct; only the two integers are wrong.
    """
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    idx = data.find(b"data", 12)
    if idx < 0 or idx + 8 > len(data):
        return data
    header = bytearray(data)
    header[4:8] = struct.pack("<I", len(data) - 8)
    header[idx + 4 : idx + 8] = struct.pack("<I", len(data) - idx - 8)
    return bytes(header)


class PhononEngine(BaseTTSEngine):
    """
    Gradium Phonon, served by the `ptts-openai-server` sidecar.

    Phonon is a streaming flow-matching model over the Mimi codec: ~31M params,
    ~64 MB of q8 weights, English-only, no on-device voice cloning. It exposes
    the OpenAI `/v1/audio/speech` contract, so this driver is a passthrough with
    two adaptations:

    * **Speed.** The model has no rate control; the server answers 400 for any
      speed but 1.0. The gateway therefore asks for 1.0 and applies `atempo`
      locally when a caller asks for something else.
    * **Formats.** wav/mp3/opus are served directly; aac/flac/pcm are transcoded
      from WAV.

    Streaming note: `synthesize()` buffers like the other drivers because the
    gateway returns whole responses and caches them. The sidecar itself streams
    from the first 80 ms frame, so point latency-sensitive clients (Home
    Assistant via Wyoming, a reader UI) straight at it when first-audio time
    matters more than caching.
    """

    engine_name = "phonon"
    native_format = "wav"
    # atempo's useful single-filter range; outside it the artefacting is worse
    # than simply refusing the speed.
    min_speed = 0.5
    max_speed = 2.0

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8880",
        default_voice: str = "Easton",
        timeout_seconds: Optional[float] = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_voice = default_voice
        self.timeout_seconds = timeout_seconds
        self._canonical: Dict[str, str] = {n.lower(): n for n, _, _ in PHONON_EN_VOICES}

    def canonical_voice(self, voice: str) -> Optional[str]:
        return self._canonical.get((voice or "").strip().lower())

    def is_valid_voice(self, voice: str) -> bool:
        return self.canonical_voice(voice) is not None

    def _server_format(self, target_format: str) -> str:
        return target_format if target_format in PHONON_NATIVE_FORMATS else "wav"

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/audio/speech"

    async def _post(self, text: str, voice: str, fmt: str) -> bytes:
        payload = {
            "input": text,
            "voice": voice,
            "response_format": fmt,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self._endpoint(), json=payload)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Phonon timed out after {self.timeout_seconds}s") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Phonon request failed: {e}") from e

        if response.status_code != 200:
            detail = response.text[:300]
            if 400 <= response.status_code < 500:
                # Unknown voice / bad format is the caller's fault: never trip the breaker.
                from gateway.errors import InvalidRequestError

                raise InvalidRequestError(f"Phonon rejected the request ({response.status_code}): {detail}")
            raise RuntimeError(f"Phonon returned HTTP {response.status_code}: {detail}")

        if not response.content:
            raise RuntimeError("Phonon returned empty audio data")
        return response.content

    async def _apply_speed(self, wav_bytes: bytes, speed: float, target_format: str) -> bytes:
        """Re-time WAV audio with ffmpeg atempo, then transcode to the target format."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-filter:a",
            f"atempo={speed:.3f}",
        ]
        out_args = ["-f", "wav", "-c:a", "pcm_s16le"]
        if target_format == "pcm":
            out_args = ["-f", "s16le", "-c:a", "pcm_s16le", "-ac", "1", "-ar", "24000"]
        elif target_format == "mp3":
            out_args = ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"]
        elif target_format == "opus":
            out_args = ["-f", "opus", "-c:a", "libopus", "-b:a", "64k"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            *out_args,
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=wav_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo failed: {stderr.decode(errors='ignore')[:300]}")
        if target_format == "wav":
            # ffmpeg writes the same 0xFFFFFFFF size sentinel when its output is a pipe.
            return normalize_wav_header(stdout)
        if target_format in ("flac", "aac"):
            return await convert_audio_ffmpeg(normalize_wav_header(stdout), "wav", target_format)
        return stdout

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        audio_bytes, _ = await self.synthesize_bytes(
            text=text, voice=voice, speed=speed, response_format=response_format
        )
        for i in range(0, len(audio_bytes), 4096):
            yield audio_bytes[i : i + 4096]

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        target_voice = self.canonical_voice(voice or self.default_voice)
        if target_voice is None:
            raise ValueError(f"Voice '{voice}' is not a Phonon voice")
        target_format = (response_format or self.native_format).lower()
        clamped_speed = self.clamp_speed(speed)
        retime = abs(clamped_speed - 1.0) > 0.01

        # Re-timing always happens on WAV, so the atempo pass sees a lossless source.
        server_format = "wav" if retime else self._server_format(target_format)
        audio_bytes = await self._post(text, target_voice, server_format)
        if server_format == "wav":
            audio_bytes = normalize_wav_header(audio_bytes)

        if retime:
            return await self._apply_speed(audio_bytes, clamped_speed, target_format), target_format

        if server_format == target_format:
            return audio_bytes, target_format
        converted = await convert_audio_ffmpeg(audio_bytes, "wav", target_format)
        return converted, target_format

    async def get_voices(self) -> List[VoiceObject]:
        return [
            VoiceObject(
                id=name,
                name=name,
                engine=self.engine_name,
                language=locale,
                gender=gender,
                sample_rate=24000,
            )
            for name, locale, gender in PHONON_EN_VOICES
        ]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug("Phonon health check failed: %s", e)
            return False