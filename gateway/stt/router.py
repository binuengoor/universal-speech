import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from gateway.config import AppSettings
from gateway.errors import InvalidRequestError, is_client_error
from gateway.stt.base import BaseSTTEngine
from gateway.stt.google_stt_engine import GoogleSTTEngine
from gateway.stt.groq_engine import GroqSTTEngine
from gateway.stt.local_whisper_engine import LocalWhisperEngine

logger = logging.getLogger(__name__)


class STTRouter:
    """
    Dynamic Speech-to-Text Router with resilient in-memory cooldown fallback.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.engines: Dict[str, BaseSTTEngine] = {}
        self._cooldowns: Dict[str, float] = {}  # engine_name -> cooldown_until_timestamp
        # Local Whisper is CPU-bound and already multi-threaded; run one job at a time.
        self._local_semaphore = asyncio.Semaphore(1)
        self._init_engines()

    def _init_engines(self) -> None:
        stt_cfg = self.settings.stt

        # 1. Groq Cloud STT
        if stt_cfg.engines.groq.enabled:
            self.engines["groq"] = GroqSTTEngine(
                api_key=stt_cfg.engines.groq.api_key,
                default_model=stt_cfg.engines.groq.default_model,
                timeout_seconds=stt_cfg.engines.groq.timeout_seconds,
                translation_model=stt_cfg.engines.groq.translation_model,
            )

        # 2. Google Cloud STT
        if stt_cfg.engines.google_cloud.enabled:
            self.engines["google-cloud"] = GoogleSTTEngine(
                credentials_path=stt_cfg.engines.google_cloud.credentials_path,
                language_code=stt_cfg.engines.google_cloud.language_code,
                timeout_seconds=stt_cfg.engines.google_cloud.timeout_seconds,
            )

        # 3. Local faster-whisper (INT8 CPU)
        if stt_cfg.engines.local_whisper.enabled:
            self.engines["local-whisper"] = LocalWhisperEngine(
                model_size=stt_cfg.engines.local_whisper.model_size,
                device=stt_cfg.engines.local_whisper.device,
                compute_type=stt_cfg.engines.local_whisper.compute_type,
                cpu_threads=stt_cfg.engines.local_whisper.cpu_threads,
                models_dir=stt_cfg.engines.local_whisper.models_dir,
            )

    def is_engine_available(self, name: str) -> bool:
        if name not in self.engines:
            return False
        cooldown_until = self._cooldowns.get(name, 0.0)
        return time.time() >= cooldown_until

    def set_engine_cooldown(self, name: str) -> None:
        cooldown_duration = self.settings.stt.cooldown_seconds
        self._cooldowns[name] = time.time() + cooldown_duration
        logger.warning(
            "⚡ STT Engine '%s' placed in cooldown for %.0f seconds.",
            name,
            cooldown_duration,
        )

    def clear_engine_cooldown(self, name: str) -> None:
        if name in self._cooldowns:
            del self._cooldowns[name]

    def resolve_candidate_engines(
        self,
        requested_model: Optional[str],
        filename: str = "audio.wav",
        response_format: str = "json",
        task: str = "transcribe",
    ) -> List[BaseSTTEngine]:
        candidates = self._ordered_engines(requested_model)
        capable = [e for e in candidates if e.can_handle(filename, response_format, task)]
        if candidates and not capable:
            raise InvalidRequestError(
                f"No configured STT engine supports response_format='{response_format}' for {task}"
            )
        return capable

    def _ordered_engines(self, requested_model: Optional[str]) -> List[BaseSTTEngine]:
        req = (requested_model or "").lower().strip()
        stt_cfg = self.settings.stt

        # Pinned engine requests
        if req in ("groq", "groq-whisper"):
            if "groq" in self.engines:
                return [self.engines["groq"]]
        elif req in ("local-whisper", "whisper-local"):
            if "local-whisper" in self.engines:
                return [self.engines["local-whisper"]]
        elif req in ("google", "google-stt", "google-cloud"):
            if "google-cloud" in self.engines:
                return [self.engines["google-cloud"]]

        # Fallback hierarchy for generic models (whisper-1, auto, whisper-large-v3, etc.)
        candidates: List[BaseSTTEngine] = []
        priority_order = ["groq", "google-cloud", "local-whisper"]

        # If custom default is set, prioritize it
        if stt_cfg.default_engine in priority_order:
            priority_order.remove(stt_cfg.default_engine)
            priority_order.insert(0, stt_cfg.default_engine)

        for eng_name in priority_order:
            if eng_name in self.engines:
                # Add available (not in cooldown) engines first
                if self.is_engine_available(eng_name):
                    candidates.append(self.engines[eng_name])

        # If all candidates are in cooldown, append them anyway as last-ditch effort
        if not candidates:
            for eng_name in priority_order:
                if eng_name in self.engines:
                    candidates.append(self.engines[eng_name])

        return candidates

    async def _call(self, engine: BaseSTTEngine, method: str, **kwargs) -> Any:
        fn = getattr(engine, method)
        if engine.engine_name == "local-whisper":
            async with self._local_semaphore:
                return await fn(**kwargs)
        return await fn(**kwargs)

    async def _run_with_fallback(self, task: str, filename: str, model: Optional[str], response_format: str, **kwargs) -> Tuple[Any, str]:
        candidates = self.resolve_candidate_engines(model, filename, response_format, task)
        if not candidates:
            raise RuntimeError("No STT engines are configured or available.")

        method = "transcribe" if task == "transcribe" else "translate"
        last_err: Optional[BaseException] = None
        for engine in candidates:
            try:
                result = await self._call(
                    engine, method, filename=filename, model=model, response_format=response_format, **kwargs
                )
                self.clear_engine_cooldown(engine.engine_name)
                return result, engine.engine_name
            except Exception as e:
                if is_client_error(e):
                    # Corrupt/unsupported audio or bad parameters: the next engine would
                    # fail the same way, and no engine should be put in cooldown for it.
                    raise InvalidRequestError(f"STT request rejected by '{engine.engine_name}': {e}") from e
                logger.warning("STT engine '%s' %s failed: %s", engine.engine_name, task, str(e))
                self.set_engine_cooldown(engine.engine_name)
                last_err = e

        raise RuntimeError(f"All STT engines failed for {task}. Last error: {last_err}")

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
    ) -> Tuple[Any, str]:
        return await self._run_with_fallback(
            "transcribe",
            filename,
            model,
            response_format,
            audio_bytes=audio_bytes,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )

    async def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Tuple[Any, str]:
        return await self._run_with_fallback(
            "translate",
            filename,
            model,
            response_format,
            audio_bytes=audio_bytes,
            prompt=prompt,
            temperature=temperature,
        )

    async def get_engine_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        for name, engine in self.engines.items():
            in_cooldown = not self.is_engine_available(name)
            try:
                is_ok = await engine.health_check()
                status = "ok" if is_ok and not in_cooldown else ("cooldown" if in_cooldown else "degraded")
                health[name] = {
                    "status": status,
                    "in_cooldown": in_cooldown,
                }
            except Exception:
                health[name] = {"status": "error", "in_cooldown": in_cooldown}
        return health
