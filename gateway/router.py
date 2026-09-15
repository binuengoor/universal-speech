import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from gateway.config import AppSettings
from gateway.engines.base import BaseTTSEngine
from gateway.engines.edge_engine import EdgeTTSEngine
from gateway.engines.google_engine import GoogleTTSEngine
from gateway.engines.kokoro_engine import KokoroEngine
from gateway.engines.piper_engine import PiperEngine
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


class TTSRouter:
    """
    Dynamic Router and Circuit Breaker for TTS synthesis.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.engines: Dict[str, BaseTTSEngine] = {}
        self._cooldowns: Dict[str, float] = {}  # engine_name -> cooldown_until_timestamp
        self._init_engines()

    def _init_engines(self) -> None:
        if self.settings.engines.edge_tts.enabled:
            self.engines["edge-tts"] = EdgeTTSEngine(
                default_voice=self.settings.engines.edge_tts.default_voice,
                timeout_seconds=self.settings.engines.edge_tts.timeout_seconds or 5.0,
            )

        if self.settings.engines.piper.enabled:
            self.engines["piper"] = PiperEngine(
                models_dir=self.settings.paths.piper_models_dir,
                default_voice=self.settings.engines.piper.default_voice,
            )

        if self.settings.engines.kokoro.enabled:
            self.engines["kokoro"] = KokoroEngine(
                model_path=self.settings.paths.kokoro_model_path,
                voices_path=self.settings.paths.kokoro_voices_path,
                default_voice=self.settings.engines.kokoro.default_voice,
            )

        if getattr(self.settings.engines, "google_cloud", None) and self.settings.engines.google_cloud.enabled:
            self.engines["google-cloud"] = GoogleTTSEngine(
                credentials_path=self.settings.engines.google_cloud.credentials_path,
                default_voice=self.settings.engines.google_cloud.default_voice,
                timeout_seconds=self.settings.engines.google_cloud.timeout_seconds or 8.0,
            )

    def is_engine_available(self, name: str) -> bool:
        if name not in self.engines:
            return False
        cooldown_until = self._cooldowns.get(name, 0.0)
        return time.time() >= cooldown_until

    def set_engine_cooldown(self, name: str) -> None:
        cooldown_duration = self.settings.circuit_breaker.cooldown_seconds
        self._cooldowns[name] = time.time() + cooldown_duration
        logger.warning(
            "⚡ TTS Engine '%s' placed in cooldown for %.0f seconds.",
            name,
            cooldown_duration,
        )

    def clear_engine_cooldown(self, name: str) -> None:
        if name in self._cooldowns:
            del self._cooldowns[name]

    def resolve_engine_and_voice(
        self,
        model: Optional[str],
        voice: Optional[str],
    ) -> Tuple[BaseTTSEngine, str]:
        req_model = (model or "").lower().strip()
        req_voice = (voice or "").strip()

        # Step 1: Map OpenAI voice names (alloy, echo, etc.)
        if req_voice.lower() in self.settings.openai_voice_map:
            req_voice = self.settings.openai_voice_map[req_voice.lower()]

        # Step 2: Map short aliases (ryan, heart, aria, etc.)
        if req_voice.lower() in self.settings.short_aliases:
            req_voice = self.settings.short_aliases[req_voice.lower()]

        # Step 3: Determine engine based on explicit model or voice heuristics
        target_engine_name = None

        if req_model in ("edge-tts", "edge_tts"):
            target_engine_name = "edge-tts"
        elif req_model in ("piper", "piper-tts"):
            target_engine_name = "piper"
        elif req_model in ("kokoro", "kokoro-tts"):
            target_engine_name = "kokoro"
        elif req_model in ("google", "google-tts", "google-cloud", "gcp-tts"):
            target_engine_name = "google-cloud"
        else:
            # Model was None, empty, "auto", "tts-1", "tts-1-hd", or unrecognized.
            # Check voice heuristics first:
            if any(k in req_voice for k in ("Neural2", "Journey", "Studio", "Wavenet", "Standard", "Chirp")):
                target_engine_name = "google-cloud"
            elif any(req_voice.startswith(p) for p in ("af_", "am_", "bf_", "bm_", "jf_", "jm_", "zf_", "zm_", "ef_", "ff_", "if_", "pf_", "hf_")):
                target_engine_name = "kokoro"
            elif "Neural" in req_voice or (req_voice.count("-") >= 2 and not req_voice.endswith(("-low", "-medium", "-high"))):
                target_engine_name = "edge-tts"
            elif req_voice.endswith(("-low", "-medium", "-high")) or "piper" in req_voice.lower():
                target_engine_name = "piper"
            else:
                target_engine_name = self.settings.defaults.engine

        # If primary engine is not available or registered, pick next available
        if target_engine_name not in self.engines:
            target_engine_name = next(iter(self.engines)) if self.engines else "edge-tts"

        engine = self.engines.get(target_engine_name)
        if not engine:
            raise RuntimeError(f"Engine '{target_engine_name}' is not configured or available")

        # Resolve voice default if empty
        resolved_voice = req_voice if req_voice else engine.default_voice
        return engine, resolved_voice

    def resolve_format(self, engine: BaseTTSEngine, requested_format: Optional[str]) -> str:
        if requested_format:
            return requested_format.lower()
        return engine.native_format

    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Synthesize audio with automatic circuit breaker cooldown fallback.
        Returns: (audio_bytes, format_name, resolved_engine_name)
        """
        primary_engine, resolved_voice = self.resolve_engine_and_voice(model, voice)
        target_fmt = self.resolve_format(primary_engine, response_format)

        # Check if primary engine is cooling down
        if not self.is_engine_available(primary_engine.engine_name):
            logger.info(
                "⚡ Primary engine '%s' is in cooldown. Short-circuiting to fallback...",
                primary_engine.engine_name,
            )
            fallback_name = self.settings.circuit_breaker.fallback_engine
            fallback_engine = self.engines.get(fallback_name)
            if fallback_engine:
                fb_voice = self.settings.circuit_breaker.fallback_voice
                audio_bytes, out_fmt = await fallback_engine.synthesize_bytes(
                    text=text,
                    voice=fb_voice,
                    speed=speed,
                    response_format=target_fmt,
                )
                return audio_bytes, out_fmt, fallback_name

        try:
            audio_bytes, out_fmt = await primary_engine.synthesize_bytes(
                text=text,
                voice=resolved_voice,
                speed=speed,
                response_format=target_fmt,
            )
            self.clear_engine_cooldown(primary_engine.engine_name)
            return audio_bytes, out_fmt, primary_engine.engine_name

        except Exception as e:
            logger.warning(
                "Primary engine '%s' failed: %s",
                primary_engine.engine_name,
                str(e),
            )
            self.set_engine_cooldown(primary_engine.engine_name)

            # Circuit breaker fallback
            if self.settings.circuit_breaker.enabled:
                fallback_name = self.settings.circuit_breaker.fallback_engine
                fallback_engine = self.engines.get(fallback_name)
                if fallback_engine and fallback_engine.engine_name != primary_engine.engine_name:
                    fallback_voice = self.settings.circuit_breaker.fallback_voice
                    logger.info(
                        "⚡ Circuit breaker triggered: Falling back from %s to %s (voice: %s)",
                        primary_engine.engine_name,
                        fallback_name,
                        fallback_voice,
                    )
                    try:
                        audio_bytes, out_fmt = await fallback_engine.synthesize_bytes(
                            text=text,
                            voice=fallback_voice,
                            speed=speed,
                            response_format=target_fmt,
                        )
                        return audio_bytes, out_fmt, fallback_name
                    except Exception as fb_err:
                        logger.error("Fallback engine '%s' also failed: %s", fallback_name, fb_err)

            raise e

    async def get_all_voices(self) -> List[VoiceObject]:
        all_voices: List[VoiceObject] = []
        for engine in self.engines.values():
            try:
                voices = await engine.get_voices()
                all_voices.extend(voices)
            except Exception as e:
                logger.warning("Error fetching voices from engine %s: %s", engine.engine_name, e)
        return all_voices

    async def get_curated_voices(
        self,
        all: bool = False,
        locale: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> List[VoiceObject]:
        """
        Return voices, either the curated subset (default) or the full catalog (if all=True).
        Supports optional filtering by locale and engine.
        """
        def _matches_filter(v: VoiceObject) -> bool:
            if engine and v.engine.lower() != engine.lower():
                return False
            if locale:
                target_loc = locale.lower().replace("_", "-")
                v_loc = v.language.lower().replace("_", "-")
                if v_loc != target_loc and not v_loc.startswith(target_loc):
                    return False
            return True

        # Full catalog requested or curated voices disabled in config
        if all or not getattr(self.settings.curated_voices, "enabled", True):
            all_voices = await self.get_all_voices()
            return [v for v in all_voices if _matches_filter(v)]

        # Curated subset
        curated_list = self.settings.curated_voices.voices
        curated_objects: List[VoiceObject] = []

        for item in curated_list:
            # Only include if the engine is enabled and active in this deployment
            if item.engine not in self.engines:
                continue

            v_obj = VoiceObject(
                id=item.id,
                name=item.name or item.id,
                engine=item.engine,
                language=item.language,
                gender=item.gender,
                sample_rate=item.sample_rate,
            )
            if _matches_filter(v_obj):
                curated_objects.append(v_obj)

        return curated_objects

    async def get_engine_health(self) -> Dict[str, bool]:
        status = {}
        for name, engine in self.engines.items():
            try:
                status[name] = await engine.health_check()
            except Exception:
                status[name] = False
        return status

