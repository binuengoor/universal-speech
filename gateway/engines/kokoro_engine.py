import asyncio
import io
import logging
import re
import threading
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple
import soundfile as sf
import numpy as np

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)

_KOKORO_VOICE_RE = re.compile(r"^[a-z][fm]_[a-z]+$")


class KokoroEngine(BaseTTSEngine):
    engine_name = "kokoro"
    native_format = "wav"
    min_speed = 0.5
    max_speed = 2.0

    def __init__(
        self,
        model_path: str = "models/kokoro/kokoro-v1.0.onnx",
        voices_path: str = "models/kokoro/voices-v1.0.bin",
        default_voice: str = "af_heart",
    ):
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.default_voice = default_voice
        self._kokoro_instance = None
        self._voice_names: Optional[List[str]] = None
        self._lock = threading.Lock()

    def is_valid_voice(self, voice: str) -> bool:
        if self._voice_names:
            return voice in self._voice_names
        return bool(_KOKORO_VOICE_RE.match(voice))

    def _ensure_loaded(self):
        if self._kokoro_instance is not None:
            return self._kokoro_instance
        with self._lock:
            if self._kokoro_instance is None:
                self._kokoro_instance = self._load()
                try:
                    self._voice_names = list(self._kokoro_instance.get_voices())
                except Exception:
                    self._voice_names = None
        return self._kokoro_instance

    def _load(self):

        if not self.model_path.exists() or not self.voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found: {self.model_path} and/or {self.voices_path}. "
                f"Please run 'python scripts/download_models.py --engine kokoro' to download them."
            )

        try:
            from kokoro_onnx import Kokoro
            logger.info("Loading Kokoro ONNX model from %s...", self.model_path)
            return Kokoro(
                model_path=str(self.model_path),
                voices_path=str(self.voices_path),
            )
        except Exception as e:
            logger.error("Failed to initialize Kokoro engine: %s", str(e))
            raise

    def _determine_language(self, voice: str) -> str:
        # Kokoro voice prefixes: 'a' for American English, 'b' for British English, 'j' for Japanese, 'z' for Mandarin, etc.
        if voice.startswith("af_") or voice.startswith("am_"):
            return "en-us"
        elif voice.startswith("bf_") or voice.startswith("bm_"):
            return "en-gb"
        elif voice.startswith("jf_") or voice.startswith("jm_"):
            return "ja"
        elif voice.startswith("zf_") or voice.startswith("zm_"):
            return "zh"
        elif voice.startswith("ef_") or voice.startswith("em_"):
            return "es"
        elif voice.startswith("ff_") or voice.startswith("fm_"):
            return "fr"
        elif voice.startswith("if_") or voice.startswith("im_"):
            return "it"
        elif voice.startswith("pf_") or voice.startswith("pm_"):
            return "pt-br"
        elif voice.startswith("hf_") or voice.startswith("hm_"):
            return "hi"
        return "en-us"

    def _synthesize_sync(self, text: str, voice: Optional[str], speed: float) -> bytes:
        kokoro = self._ensure_loaded()
        target_voice = voice or self.default_voice
        lang = self._determine_language(target_voice)

        samples, sample_rate = kokoro.create(
            text=text,
            voice=target_voice,
            speed=self.clamp_speed(speed),
            lang=lang,
        )

        out_buf = io.BytesIO()
        sf.write(out_buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        wav_bytes, fmt = await self.synthesize_bytes(text, voice, speed, response_format)
        chunk_size = 32768
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i : i + chunk_size]

    async def synthesize_bytes(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        target_fmt = (response_format or self.native_format).lower()
        wav_bytes = await asyncio.to_thread(self._synthesize_sync, text, voice, speed)

        if target_fmt == self.native_format:
            return wav_bytes, "wav"

        converted = await convert_audio_ffmpeg(wav_bytes, "wav", target_fmt)
        return converted, target_fmt

    async def get_voices(self) -> List[VoiceObject]:
        known_voices = [
            "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah", "af_sky",
            "am_adam", "am_michael", "am_eric", "am_fenrir", "am_liam", "am_onyx", "am_puck",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis"
        ]

        if self.model_path.exists() and self.voices_path.exists():
            try:
                # Loading the model is slow and CPU-bound: keep it off the event loop.
                await asyncio.to_thread(self._ensure_loaded)
                if self._voice_names:
                    known_voices = self._voice_names
            except Exception:
                pass

        voices: List[VoiceObject] = []
        for v in known_voices:
            gender = "Female" if "_f" in v or v.startswith("af_") or v.startswith("bf_") else "Male"
            lang = self._determine_language(v)
            voices.append(
                VoiceObject(
                    id=v,
                    name=f"Kokoro {v}",
                    engine=self.engine_name,
                    language=lang,
                    gender=gender,
                    sample_rate=24000,
                )
            )
        return voices

    async def health_check(self) -> bool:
        return self.model_path.exists() and self.voices_path.exists()
