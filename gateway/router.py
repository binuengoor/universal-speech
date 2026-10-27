import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from gateway.config import AppSettings
from gateway.engines.base import BaseTTSEngine
from gateway.engines.edge_engine import EdgeTTSEngine
from gateway.engines.google_engine import GoogleTTSEngine
from gateway.engines.kokoro_engine import KokoroEngine
from gateway.engines.piper_engine import PiperEngine
from gateway.errors import InvalidRequestError, is_client_error
from gateway.schemas import VoiceObject

logger = logging.getLogger(__name__)


@dataclass
class SynthesisPlan:
    """A validated, fully-resolved synthesis request."""

    engine: BaseTTSEngine
    voice: str
    speed: float
    response_format: str


@dataclass
class SynthesisResult:
    audio: bytes
    format: str
    engine: str
    voice: str
    fallback: bool = False


class TTSRouter:
    """
    Dynamic Router and Circuit Breaker for TTS synthesis.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.engines: Dict[str, BaseTTSEngine] = {}
        self._cooldowns: Dict[str, float] = {}  # engine_name -> cooldown_until_timestamp
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._init_engines()

    def _init_engines(self) -> None:
        engines_cfg = self.settings.engines
        if engines_cfg.edge_tts.enabled:
            self.engines["edge-tts"] = EdgeTTSEngine(
                default_voice=engines_cfg.edge_tts.default_voice,
                timeout_seconds=engines_cfg.edge_tts.timeout_seconds or 10.0,
            )

        if engines_cfg.piper.enabled:
            self.engines["piper"] = PiperEngine(
                models_dir=self.settings.paths.piper_models_dir,
                default_voice=engines_cfg.piper.default_voice,
            )

        if engines_cfg.kokoro.enabled:
            self.engines["kokoro"] = KokoroEngine(
                model_path=self.settings.paths.kokoro_model_path,
                voices_path=self.settings.paths.kokoro_voices_path,
                default_voice=engines_cfg.kokoro.default_voice,
            )

        if getattr(engines_cfg, "google_cloud", None) and engines_cfg.google_cloud.enabled:
            self.engines["google-cloud"] = GoogleTTSEngine(
                credentials_path=engines_cfg.google_cloud.credentials_path,
                default_voice=engines_cfg.google_cloud.default_voice,
                timeout_seconds=engines_cfg.google_cloud.timeout_seconds or 8.0,
            )

        limits = {
            "edge-tts": engines_cfg.edge_tts.max_concurrency,
            "piper": engines_cfg.piper.max_concurrency,
            "kokoro": engines_cfg.kokoro.max_concurrency,
            "google-cloud": engines_cfg.google_cloud.max_concurrency,
        }
        for name, limit in limits.items():
            if name in self.engines and limit and limit > 0:
                self._semaphores[name] = asyncio.Semaphore(limit)

    def is_engine_available(self, name: str) -> bool:
        if name not in self.engines:
            return False
        if not self.settings.circuit_breaker.enabled:
            return True
        cooldown_until = self._cooldowns.get(name, 0.0)
        return time.time() >= cooldown_until

    def set_engine_cooldown(self, name: str) -> None:
        if not self.settings.circuit_breaker.enabled:
            return
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

    def _map_openai_voice(self, voice: str, engine_name: str) -> str:
        key = voice.lower()
        engine_map = self.settings.engine_voice_maps.get(engine_name, {})
        if key in engine_map:
            return engine_map[key]
        if key in self.settings.openai_voice_map:
            return self.settings.openai_voice_map[key]
        return voice

    def resolve_engine_and_voice(
        self,
        model: Optional[str],
        voice: Optional[str],
    ) -> Tuple[BaseTTSEngine, str]:
        req_model = (model or "").lower().strip()
        req_voice = (voice or "").strip()
        if not req_model and not req_voice:
            req_voice = self.settings.defaults.voice

        # Step 1: Map short aliases (ryan, heart, aria, etc.)
        if req_voice.lower() in self.settings.short_aliases:
            req_voice = self.settings.short_aliases[req_voice.lower()]

        # Step 2: Determine engine based on explicit model or voice heuristics
        target_engine_name = None
        is_openai_voice = req_voice.lower() in self.settings.openai_voice_map or any(
            req_voice.lower() in m for m in self.settings.engine_voice_maps.values()
        )

        if req_model in ("edge-tts", "edge_tts"):
            target_engine_name = "edge-tts"
        elif req_model in ("piper", "piper-tts"):
            target_engine_name = "piper"
        elif req_model in ("kokoro", "kokoro-tts"):
            target_engine_name = "kokoro"
        elif req_model in ("google", "google-tts", "google-cloud", "gcp-tts"):
            target_engine_name = "google-cloud"
        elif is_openai_voice:
            # Generic model (tts-1 etc.) with an OpenAI voice name: use the default engine.
            target_engine_name = self.settings.defaults.engine
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

        if target_engine_name not in self.engines:
            if req_model and req_model not in ("tts-1", "tts-1-hd", "auto"):
                raise InvalidRequestError(f"TTS engine for model '{model}' is not enabled on this gateway")
            target_engine_name = next(iter(self.engines)) if self.engines else None

        engine = self.engines.get(target_engine_name) if target_engine_name else None
        if not engine:
            raise RuntimeError("No TTS engines are configured")

        # Step 3: OpenAI voice names are mapped per engine so they are valid for it.
        if req_voice and is_openai_voice:
            req_voice = self._map_openai_voice(req_voice, engine.engine_name)

        resolved_voice = req_voice if req_voice else engine.default_voice
        return engine, resolved_voice

    def resolve_format(self, engine: BaseTTSEngine, requested_format: Optional[str]) -> str:
        if requested_format:
            return requested_format.lower()
        return (self.settings.defaults.response_format or engine.native_format).lower()

    def plan(
        self,
        model: Optional[str],
        voice: Optional[str],
        speed: Optional[float],
        response_format: Optional[str],
    ) -> SynthesisPlan:
        """Resolve and validate a request. Raises InvalidRequestError for caller mistakes."""
        engine, resolved_voice = self.resolve_engine_and_voice(model, voice)
        if not engine.is_valid_voice(resolved_voice):
            raise InvalidRequestError(
                f"Voice '{resolved_voice}' is not available for engine '{engine.engine_name}'"
            )
        effective_speed = engine.clamp_speed(speed if speed is not None else self.settings.defaults.speed)
        return SynthesisPlan(
            engine=engine,
            voice=resolved_voice,
            speed=effective_speed,
            response_format=self.resolve_format(engine, response_format),
        )

    async def _run(self, engine: BaseTTSEngine, text: str, voice: str, speed: float, fmt: str) -> Tuple[bytes, str]:
        sem = self._semaphores.get(engine.engine_name)
        if sem is None:
            return await engine.synthesize_bytes(text=text, voice=voice, speed=speed, response_format=fmt)
        async with sem:
            return await engine.synthesize_bytes(text=text, voice=voice, speed=speed, response_format=fmt)

    def _fallback_engine(self, primary: BaseTTSEngine) -> Optional[BaseTTSEngine]:
        cb = self.settings.circuit_breaker
        if not cb.enabled:
            return None
        fallback = self.engines.get(cb.fallback_engine)
        if fallback is None or fallback.engine_name == primary.engine_name:
            return None
        return fallback

    async def synthesize_plan(self, text: str, plan: SynthesisPlan, allow_fallback: bool = True) -> SynthesisResult:
        """
        Synthesize audio with automatic circuit breaker cooldown fallback.
        Results produced by the fallback engine are flagged so callers do not
        cache them under the requested voice.
        """
        primary = plan.engine
        fallback = self._fallback_engine(primary) if allow_fallback else None

        async def _use_fallback(reason: str) -> SynthesisResult:
            assert fallback is not None
            fb_voice = self.settings.circuit_breaker.fallback_voice
            logger.info(
                "⚡ Falling back from %s to %s (voice: %s): %s",
                primary.engine_name, fallback.engine_name, fb_voice, reason,
            )
            audio, fmt = await self._run(fallback, text, fb_voice, fallback.clamp_speed(plan.speed), plan.response_format)
            return SynthesisResult(audio, fmt, fallback.engine_name, fb_voice, fallback=True)

        if not self.is_engine_available(primary.engine_name) and fallback is not None:
            try:
                return await _use_fallback("primary engine in cooldown")
            except Exception as fb_err:
                # Fallback is unusable (e.g. models not downloaded): retry the primary instead.
                logger.error("Fallback engine '%s' failed: %s", fallback.engine_name, fb_err)

        try:
            audio, fmt = await self._run(primary, text, plan.voice, plan.speed, plan.response_format)
            self.clear_engine_cooldown(primary.engine_name)
            return SynthesisResult(audio, fmt, primary.engine_name, plan.voice)
        except Exception as e:
            if is_client_error(e):
                # The request was bad, not the engine: never trip the breaker for everyone.
                raise InvalidRequestError(str(e)) from e
            logger.warning("Primary engine '%s' failed: %s", primary.engine_name, e)
            self.set_engine_cooldown(primary.engine_name)
            if fallback is not None:
                try:
                    return await _use_fallback(str(e))
                except Exception as fb_err:
                    logger.error("Fallback engine '%s' also failed: %s", fallback.engine_name, fb_err)
            raise

    async def synthesize(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Convenience wrapper returning (audio_bytes, format_name, used_engine_name).
        """
        plan = self.plan(model, voice, speed, response_format)
        result = await self.synthesize_plan(text, plan)
        return result.audio, result.format, result.engine

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
                status[name] = bool(await engine.health_check()) and self.is_engine_available(name)
            except Exception:
                status[name] = False
        return status

