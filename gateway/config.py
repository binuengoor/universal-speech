import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_VOICE_MAP = {
    "alloy": "en-US-AriaNeural",
    "echo": "en-US-GuyNeural",
    "fable": "en-GB-SoniaNeural",
    "onyx": "en-US-ChristopherNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-AnaNeural",
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


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    cors_origins: List[str] = ["*"]


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
    timeout_seconds: float = 5.0
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


class GoogleEngineConfig(BaseModel):
    enabled: bool = True
    credentials_path: Optional[str] = "credentials/google-service-account.json"
    default_voice: str = "en-US-Neural2-F"
    native_format: str = "mp3"
    timeout_seconds: Optional[float] = 8.0


class EnginesConfig(BaseModel):
    edge_tts: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="en-US-AriaNeural", native_format="mp3", timeout_seconds=5.0
    )
    piper: EngineDetailConfig = EngineDetailConfig(
        enabled=False, default_voice="en_US-ryan-medium", native_format="wav"
    )
    kokoro: EngineDetailConfig = EngineDetailConfig(
        enabled=True, default_voice="af_heart", native_format="wav"
    )
    google_cloud: GoogleEngineConfig = Field(default_factory=GoogleEngineConfig)


class STTGroqConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    default_model: str = "whisper-large-v3-turbo"
    timeout_seconds: float = 10.0


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


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPEECH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    engines: EnginesConfig = Field(default_factory=EnginesConfig)
    curated_voices: CuratedVoicesConfig = Field(default_factory=CuratedVoicesConfig)
    openai_voice_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_OPENAI_VOICE_MAP))
    short_aliases: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SHORT_ALIASES))

    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None) -> "AppSettings":
        path_to_try = config_path or os.getenv("SPEECH_CONFIG_PATH") or os.getenv("TTS_CONFIG_PATH", "config.yaml")
        yaml_data: Dict[str, Any] = {}
        if Path(path_to_try).exists():
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded

        # Overlay GROQ_API_KEY if present in environment
        groq_env_key = os.getenv("GROQ_API_KEY", "")
        if groq_env_key:
            stt_dict = yaml_data.setdefault("stt", {})
            stt_engines = stt_dict.setdefault("engines", {})
            groq_dict = stt_engines.setdefault("groq", {})
            if not groq_dict.get("api_key"):
                groq_dict["api_key"] = groq_env_key

        return cls(**yaml_data)


settings = AppSettings.load_from_yaml()
