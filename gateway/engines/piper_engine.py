import asyncio
import io
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import wave

from gateway.audio import convert_audio_ffmpeg
from gateway.engines.base import BaseTTSEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


class PiperEngine(BaseTTSEngine):
    engine_name = "piper"
    native_format = "wav"

    def is_valid_voice(self, voice: str) -> bool:
        self._discover_models()
        return voice in self._voice_models_map

    def __init__(self, models_dir: str = "models/piper", default_voice: str = "en_US-ryan-medium"):
        self.models_dir = Path(models_dir)
        self.default_voice = default_voice
        self._loaded_voices: Dict[str, Any] = {}
        self._voice_models_map: Dict[str, Tuple[Path, Path]] = {}
        self._discover_models()

    def _discover_models(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._voice_models_map.clear()
        for onnx_file in self.models_dir.glob("*.onnx"):
            config_file = onnx_file.with_suffix(".onnx.json")
            if not config_file.exists():
                # Check alternative naming: model_name.json
                alt_config = onnx_file.with_suffix(".json")
                if alt_config.exists():
                    config_file = alt_config

            voice_id = onnx_file.stem
            self._voice_models_map[voice_id] = (onnx_file, config_file if config_file.exists() else None)
            
            # If name has format en_US-ryan-medium, also map 'ryan'
            parts = voice_id.split("-")
            if len(parts) >= 2:
                short_name = parts[1]
                if short_name not in self._voice_models_map:
                    self._voice_models_map[short_name] = (onnx_file, config_file if config_file.exists() else None)

    def _resolve_model_paths(self, voice: Optional[str]) -> Optional[Tuple[Path, Optional[Path]]]:
        self._discover_models()
        target = voice or self.default_voice
        # Never silently substitute another voice: callers cache by voice name.
        return self._voice_models_map.get(target)

    def _get_piper_voice(self, voice: Optional[str]):
        paths = self._resolve_model_paths(voice)
        if not paths:
            raise FileNotFoundError(
                f"Piper voice '{voice or self.default_voice}' not found in {self.models_dir}. "
                f"Please run 'python scripts/download_models.py --engine piper' to download default models."
            )
        onnx_path, config_path = paths
        key = str(onnx_path)
        if key not in self._loaded_voices:
            try:
                import piper  # provided by the optional 'piper' extra (piper-tts)
                self._loaded_voices[key] = piper.PiperVoice.load(
                    model_path=str(onnx_path),
                    config_path=str(config_path) if config_path else None,
                )
            except Exception as e:
                logger.error("Failed to load Piper model %s: %s", onnx_path, e)
                raise
        return self._loaded_voices[key]

    def _synthesize_sync(self, text: str, voice: Optional[str], speed: float) -> bytes:
        piper_voice = self._get_piper_voice(voice)
        import piper

        length_scale = 1.0 / max(speed, 0.25)
        syn_config = piper.SynthesisConfig(
            length_scale=length_scale,
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            piper_voice.synthesize_wav(text, wav_file, syn_config=syn_config)

        return buf.getvalue()

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        wav_bytes, fmt = await self.synthesize_bytes(text, voice, speed, response_format)
        # Yield in chunks for streaming simulation or direct delivery
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
        self._discover_models()
        voices: List[VoiceObject] = []
        seen = set()
        for v_id, (onnx_path, _) in self._voice_models_map.items():
            if onnx_path.stem in seen:
                continue
            seen.add(onnx_path.stem)
            
            # Guess language from filename e.g. en_US-ryan-medium -> en-US
            lang = "en-US"
            parts = onnx_path.stem.split("-")
            if parts and "_" in parts[0]:
                lang = parts[0].replace("_", "-")

            voices.append(
                VoiceObject(
                    id=onnx_path.stem,
                    name=f"Piper {onnx_path.stem}",
                    engine=self.engine_name,
                    language=lang,
                    sample_rate=22050,
                )
            )
        if not voices:
            # Provide placeholder if not downloaded yet
            voices.append(
                VoiceObject(
                    id=self.default_voice,
                    name=f"Piper {self.default_voice} (Download required)",
                    engine=self.engine_name,
                    language="en-US",
                    sample_rate=22050,
                )
            )
        return voices

    async def health_check(self) -> bool:
        self._discover_models()
        return len(self._voice_models_map) > 0
