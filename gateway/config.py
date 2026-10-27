import os
from pathlib import Path
import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type
import yaml
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_VOICE_MAP = {
    "alloy": "en-US-AriaNeural",
    "echo": "en-US-GuyNeural",
    "fable": "en-GB-SoniaNeural",
    "onyx": "en-US-ChristopherNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-AnaNeural",
    "ash": "en-US-AndrewNeural",
    "ballad": "en-GB-RyanNeural",
    "coral": "en-US-EmmaNeural",
    "sage": "en-US-AvaNeural",
    "verse": "en-US-BrianNeural",
}

# OpenAI voice names mapped per engine, used when the request pins an engine
# (e.g. model="kokoro", voice="alloy") so the voice is valid for that engine.
DEFAULT_ENGINE_VOICE_MAPS: Dict[str, Dict[str, str]] = {
    "kokoro": {
        "alloy": "af_alloy",
        "ash": "am_adam",
        "ballad": "bm_george",
        "coral": "af_bella",
        "echo": "am_echo",
        "fable": "bm_fable",
        "nova": "af_nova",
        "onyx": "am_onyx",
        "sage": "af_sarah",
        "shimmer": "af_sky",
        "verse": "am_michael",
    },
    "google-cloud": {
        "alloy": "en-US-Neural2-F",
        "ash": "en-US-Neural2-D",
        "ballad": "en-GB-Neural2-B",
        "coral": "en-US-Neural2-C",
        "echo": "en-US-Neural2-J",
        "fable": "en-GB-Neural2-A",
        "nova": "en-US-Neural2-H",
        "onyx": "en-US-Neural2-D",
        "sage": "en-US-Neural2-E",
        "shimmer": "en-US-Neural2-G",
        "verse": "en-US-Neural2-I",
    },
}

DEFAULT_SHORT_ALIASES = {
    "aria": "en-US-AriaNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "sonia": "en-GB-SoniaNeural",
    "ryan": "en_US-ryan-medium",
    "amy": "en_US-amy-medium",
    "lessac": "en_US-lessac-medium",
    "heart": "af_heart",
    "bella": "af_bella",
    "sarah": "af_sarah",
    "nicole": "af_nicole",
    "adam": "am_adam",
    "michael": "am_michael",
    "neural2-f": "en-US-Neural2-F",
    "neural2-d": "en-US-Neural2-D",
    "journey-f": "en-US-Journey-F",
    "journey-d": "en-US-Journey-D",
}

ENV_PREFIXES = ("SPEECH_", "TTS_")


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    cors_origins: List[str] = ["*"]
    max_upload_mb: int = 25


class DefaultsConfig(BaseModel):
    engine: str = "edge-tts"
    voice: str = "en-US-AriaNeural"
    speed: float = 1.0
    response_format: str = "mp3"


class CacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 500
    max_memory_mb: int = 50
    ttl_seconds: int = 86400


class CircuitBreakerConfig(BaseModel):
    enabled: bool = True
    fallback_engine: str = "kokoro"
    fallback_voice: str = "af_heart"
    cooldown_seconds: float = 60.0


class PathsConfig(BaseModel):
    piper_models_dir: str = "models/piper"
    kokoro_model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/kokoro/voices-v1.0.bin"
    whisper_models_dir: str = "models/whisper"


class EngineDetailConfig(BaseModel):
    enabled: bool = True
    default_voice: str
    native_format: str = "wav"
    timeout_seconds: Optional[float] = None
    max_concurrency: int = 0  # 0 = unlimited


class GoogleEngineConfig(BaseModel):
    enabled: bool = True
    credentials_path: Optional[str] = "credentials/google-service-account.json"
    default_voice: str = "en-US-Neural2-F"
    native_format: str = "mp3"
    timeout_seconds: Optional[float] = 8.0
    max_concurrency: int = 0


class EnginesConfig(BaseModel):
    edge_tts: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="en-US-AriaNeural", native_format="mp3", timeout_seconds=10.0
    )
    piper: EngineDetailConfig = EngineDetailConfig(
        enabled=False, default_voice="en_US-ryan-medium", native_format="wav", max_concurrency=1
    )
    kokoro: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="af_heart", native_format="wav", max_concurrency=1
    )
    google_cloud: GoogleEngineConfig = Field(default_factory=GoogleEngineConfig)


class STTGroqConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    default_model: str = "whisper-large-v3-turbo"
    timeout_seconds: float = 30.0
    translation_model: str = "whisper-large-v3"


class STTLocalWhisperConfig(BaseModel):
    enabled: bool = True
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    models_dir: str = "models/whisper"


class STTGoogleConfig(BaseModel):
    enabled: bool = True
    credentials_path: Optional[str] = "credentials/google-service-account.json"
    language_code: str = "en-US"
    timeout_seconds: float = 10.0


class STTEnginesConfig(BaseModel):
    groq: STTGroqConfig = Field(default_factory=STTGroqConfig)
    local_whisper: STTLocalWhisperConfig = Field(default_factory=STTLocalWhisperConfig)
    google_cloud: STTGoogleConfig = Field(default_factory=STTGoogleConfig)


class STTConfig(BaseModel):
    enabled: bool = True
    default_engine: str = "groq"
    default_model: str = "whisper-1"
    fallback_engine: str = "local-whisper"
    cooldown_seconds: float = 60.0
    engines: STTEnginesConfig = Field(default_factory=STTEnginesConfig)


class CuratedVoiceItem(BaseModel):
    id: str
    engine: str
    name: Optional[str] = None
    language: str
    gender: Optional[str] = None
    sample_rate: Optional[int] = None


class CuratedVoicesConfig(BaseModel):
    enabled: bool = True
    voices: List[CuratedVoiceItem] = Field(default_factory=list)


class _DictSource(PydanticBaseSettingsSource):
    """Settings source backed by a pre-loaded dict (YAML file or env shortcuts)."""

    def __init__(self, settings_cls: Type[BaseSettings], data: Dict[str, Any]):
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items() if k in self.settings_cls.model_fields}


def _env_shortcuts() -> Dict[str, Any]:
    """Flat convenience variables that map onto nested settings."""
    data: Dict[str, Any] = {}
    api_key = os.getenv("SPEECH_API_KEY")
    if api_key:
        data.setdefault("server", {})["api_key"] = api_key
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        data.setdefault("stt", {}).setdefault("engines", {}).setdefault("groq", {})["api_key"] = groq_key
    return data


class AppSettings(BaseSettings):
    """
    Precedence (highest first): explicit init kwargs > SPEECH_* env (nested with "__")
    > legacy TTS_* env > flat env shortcuts (SPEECH_API_KEY, GROQ_API_KEY) > config.yaml > defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPEECH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    _yaml_data: ClassVar[Dict[str, Any]] = {}

    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    engines: EnginesConfig = Field(default_factory=EnginesConfig)
    curated_voices: CuratedVoicesConfig = Field(default_factory=CuratedVoicesConfig)
    openai_voice_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_OPENAI_VOICE_MAP))
    engine_voice_maps: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_ENGINE_VOICE_MAPS.items()}
    )
    short_aliases: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SHORT_ALIASES))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        from pydantic_settings import EnvSettingsSource

        legacy_env = EnvSettingsSource(settings_cls, env_prefix="TTS_", env_nested_delimiter="__")
        return (
            init_settings,
            env_settings,
            legacy_env,
            _DictSource(settings_cls, _env_shortcuts()),
            dotenv_settings,
            _DictSource(settings_cls, cls._yaml_data),
            file_secret_settings,
        )

    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None) -> "AppSettings":
        path_to_try = config_path or os.getenv("SPEECH_CONFIG_PATH") or os.getenv("TTS_CONFIG_PATH", "config.yaml")
        yaml_data: Dict[str, Any] = {}
        if Path(path_to_try).exists():
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded

        cls._yaml_data = yaml_data
        try:
            return cls()
        finally:
            cls._yaml_data = {}


settings = AppSettings.load_from_yaml()
