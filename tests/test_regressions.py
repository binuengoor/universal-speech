"""Regression tests for the audit fixes (fallback caching, breaker, auth, STT limits)."""
import io
from unittest.mock import AsyncMock

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from gateway.config import AppSettings
from gateway.engines.google_engine import split_text_for_google
from gateway.errors import InvalidRequestError
from gateway.main import app, cache, settings, stt_router, tts_router
from gateway.router import TTSRouter


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------- TTS fallback


@pytest.mark.asyncio
async def test_fallback_audio_is_flagged_and_not_cached():
    tts_router.engines["edge-tts"].synthesize_bytes = AsyncMock(side_effect=TimeoutError("edge down"))
    tts_router.engines["kokoro"].synthesize_bytes = AsyncMock(return_value=(b"KOKORO", "mp3"))
    payload = {"model": "edge-tts", "voice": "en-US-AriaNeural", "input": "hello", "response_format": "mp3"}

    async with _client() as ac:
        r1 = await ac.post("/v1/audio/speech", json=payload)
        assert r1.status_code == 200
        assert r1.headers["X-Fallback"] == "true"
        assert r1.headers["X-Engine"] == "kokoro"
        assert r1.headers["X-Voice"] == settings.circuit_breaker.fallback_voice
        assert r1.headers["Cache-Control"] == "no-store"
        assert cache.get_stats()["entries"] == 0

        # Edge recovers: the next request must get Edge audio, not a poisoned cache hit.
        tts_router._cooldowns.clear()
        tts_router.engines["edge-tts"].synthesize_bytes = AsyncMock(return_value=(b"EDGE", "mp3"))
        r2 = await ac.post("/v1/audio/speech", json=payload)
        assert r2.content == b"EDGE"
        assert r2.headers["X-Cache"] == "MISS"
        assert "X-Fallback" not in r2.headers


@pytest.mark.asyncio
async def test_no_fallback_header_returns_error_instead_of_substitute_voice():
    tts_router.engines["edge-tts"].synthesize_bytes = AsyncMock(side_effect=TimeoutError("edge down"))
    kokoro = AsyncMock(return_value=(b"KOKORO", "mp3"))
    tts_router.engines["kokoro"].synthesize_bytes = kokoro
    async with _client() as ac:
        r = await ac.post(
            "/v1/audio/speech",
            json={"model": "edge-tts", "voice": "en-US-AriaNeural", "input": "hello"},
            headers={"X-No-Fallback": "1"},
        )
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "server_error"
    kokoro.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_voice_is_400_and_does_not_trip_breaker():
    edge = AsyncMock(return_value=(b"EDGE", "mp3"))
    tts_router.engines["edge-tts"].synthesize_bytes = edge
    async with _client() as ac:
        r = await ac.post("/v1/audio/speech", json={"model": "edge-tts", "voice": "not-a-voice", "input": "hi"})
        assert r.status_code == 400
        assert "not available" in r.json()["error"]["message"]
        assert tts_router.is_engine_available("edge-tts")

        ok = await ac.post("/v1/audio/speech", json={"model": "edge-tts", "voice": "en-US-AriaNeural", "input": "hi"})
        assert ok.status_code == 200
        assert ok.headers["X-Engine"] == "edge-tts"
    edge.assert_called_once()


@pytest.mark.asyncio
async def test_upstream_client_error_does_not_trip_breaker():
    class Upstream400(Exception):
        code = 400

    tts_router.engines["edge-tts"].synthesize_bytes = AsyncMock(side_effect=Upstream400("bad input"))
    async with _client() as ac:
        r = await ac.post("/v1/audio/speech", json={"model": "edge-tts", "voice": "en-US-AriaNeural", "input": "hi"})
    assert r.status_code == 400
    assert tts_router.is_engine_available("edge-tts")


def test_openai_voices_map_per_engine():
    router = TTSRouter(settings=AppSettings())
    engine, voice = router.resolve_engine_and_voice("kokoro", "alloy")
    assert (engine.engine_name, voice) == ("kokoro", "af_alloy")
    engine, voice = router.resolve_engine_and_voice("google-cloud", "nova")
    assert engine.engine_name == "google-cloud" and "Neural2" in voice
    engine, voice = router.resolve_engine_and_voice("tts-1", "coral")
    assert (engine.engine_name, voice) == ("edge-tts", "en-US-EmmaNeural")


def test_speed_clamped_per_engine():
    router = TTSRouter(settings=AppSettings())
    plan = router.plan("kokoro", "af_heart", 4.0, None)
    assert plan.speed == 2.0
    plan = router.plan("kokoro", "af_heart", 0.25, None)
    assert plan.speed == 0.5


def test_disabled_engine_model_is_rejected():
    router = TTSRouter(settings=AppSettings())  # piper disabled by default
    with pytest.raises(InvalidRequestError):
        router.plan("piper", "en_US-ryan-medium", 1.0, None)


@pytest.mark.asyncio
async def test_cooldown_ignored_when_breaker_disabled():
    s = AppSettings()
    s.circuit_breaker.enabled = False
    router = TTSRouter(settings=s)
    router.engines["edge-tts"].synthesize_bytes = AsyncMock(return_value=(b"EDGE", "mp3"))
    router._cooldowns["edge-tts"] = 10**12
    audio, _, engine = await router.synthesize("hi", "edge-tts", "en-US-AriaNeural")
    assert (audio, engine) == (b"EDGE", "edge-tts")


@pytest.mark.asyncio
async def test_cooldown_retries_primary_when_fallback_unusable():
    router = TTSRouter(settings=AppSettings())
    router._cooldowns["edge-tts"] = 10**12
    router.engines["kokoro"].synthesize_bytes = AsyncMock(side_effect=FileNotFoundError("no models"))
    router.engines["edge-tts"].synthesize_bytes = AsyncMock(return_value=(b"EDGE", "mp3"))
    audio, _, engine = await router.synthesize("hi", "edge-tts", "en-US-AriaNeural")
    assert (audio, engine) == (b"EDGE", "edge-tts")


def test_google_text_split_respects_byte_limit():
    text = ("This is a sentence that is reasonably long. " * 300).strip()
    chunks = split_text_for_google(text, max_bytes=500)
    assert len(chunks) > 1
    assert all(len(c.encode("utf-8")) <= 500 for c in chunks)
    assert " ".join(chunks).split() == text.split()


# ---------------------------------------------------------------- config / auth


def test_env_overrides_yaml(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('server:\n  api_key: ""\ncache:\n  max_entries: 5\n  max_memory_mb: 9\n')
    monkeypatch.setenv("SPEECH_SERVER__API_KEY", "from-env")
    monkeypatch.setenv("SPEECH_CACHE__MAX_ENTRIES", "42")
    loaded = AppSettings.load_from_yaml(str(cfg))
    assert loaded.server.api_key == "from-env"
    assert loaded.cache.max_entries == 42
    assert loaded.cache.max_memory_mb == 9  # untouched YAML value survives the nested merge


def test_env_shortcuts(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text('stt:\n  engines:\n    groq:\n      api_key: "yaml-key"\n')
    monkeypatch.setenv("SPEECH_API_KEY", "short")
    monkeypatch.setenv("GROQ_API_KEY", "env-groq")
    loaded = AppSettings.load_from_yaml(str(cfg))
    assert loaded.server.api_key == "short"
    assert loaded.stt.engines.groq.api_key == "env-groq"


@pytest.mark.asyncio
async def test_api_key_enforced(monkeypatch):
    monkeypatch.setattr(settings.server, "api_key", "s3cret")
    async with _client() as ac:
        assert (await ac.get("/v1/models")).status_code == 401
        assert (await ac.get("/v1/models", headers={"Authorization": "Bearer nope"})).status_code == 401
        assert (await ac.get("/v1/models", headers={"Authorization": "Bearer s3cret"})).status_code == 200


@pytest.mark.asyncio
async def test_models_filtered_by_type_and_enabled_engines():
    async with _client() as ac:
        tts_ids = [m["id"] for m in (await ac.get("/v1/models?type=tts")).json()["data"]]
    assert "edge-tts" in tts_ids and "whisper-1" not in tts_ids
    assert "piper" not in tts_ids  # disabled in config


# ---------------------------------------------------------------- STT


@pytest.mark.asyncio
async def test_stt_upload_limit(monkeypatch):
    monkeypatch.setattr(settings.server, "max_upload_mb", 1)
    groq = AsyncMock(return_value={"text": "x"})
    stt_router.engines["groq"].transcribe = groq
    big = b"\0" * (1024 * 1024 + 1)
    async with _client() as ac:
        r = await ac.post("/v1/audio/transcriptions", files={"file": ("a.wav", big, "audio/wav")}, data={"model": "whisper-1"})
    assert r.status_code == 413
    groq.assert_not_called()


@pytest.mark.asyncio
async def test_stt_client_error_does_not_cool_down_or_fan_out():
    class Groq400(Exception):
        code = 400

    stt_router.engines["groq"].transcribe = AsyncMock(side_effect=Groq400("corrupt audio"))
    local = AsyncMock(return_value={"text": "x"})
    stt_router.engines["local-whisper"].transcribe = local
    async with _client() as ac:
        r = await ac.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav(), "audio/wav")}, data={"model": "whisper-1"})
    assert r.status_code == 400
    assert stt_router.is_engine_available("groq")
    local.assert_not_called()


@pytest.mark.asyncio
async def test_stt_rejects_unknown_response_format():
    async with _client() as ac:
        r = await ac.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", _wav(), "audio/wav")},
            data={"model": "whisper-1", "response_format": "bogus"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_google_stt_skipped_for_unsupported_format_and_translation():
    google = stt_router.engines["google-cloud"]
    assert not google.can_handle("a.wav", "srt", "transcribe")
    assert not google.can_handle("a.mp3", "json", "transcribe")
    assert not google.can_handle("a.wav", "json", "translate")
    assert google.can_handle("a.wav", "json", "transcribe")

    async with _client() as ac:
        r = await ac.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", _wav(), "audio/wav")},
            data={"model": "google-cloud", "response_format": "srt"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_timestamp_granularities_sdk_form_key_is_forwarded():
    groq = AsyncMock(return_value={"text": "x", "words": []})
    stt_router.engines["groq"].transcribe = groq
    async with _client() as ac:
        r = await ac.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", _wav(), "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities[]": ["word"]},
        )
    assert r.status_code == 200
    assert groq.call_args.kwargs["timestamp_granularities"] == ["word"]


@pytest.mark.kokoro
@pytest.mark.asyncio
async def test_kokoro_openai_voice_end_to_end():
    async with _client() as ac:
        r = await ac.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "voice": "alloy", "input": "Mapped voice.", "response_format": "wav", "speed": 3.5},
        )
    assert r.status_code == 200
    assert r.headers["X-Voice"] == "af_alloy"
    assert r.content.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_identical_concurrent_requests_synthesize_once():
    import asyncio

    async def slow(**kwargs):
        await asyncio.sleep(0.05)
        return b"EDGE", "mp3"

    edge = AsyncMock(side_effect=slow)
    tts_router.engines["edge-tts"].synthesize_bytes = edge
    payload = {"model": "edge-tts", "voice": "en-US-AriaNeural", "input": "same text"}
    async with _client() as ac:
        r1, r2 = await asyncio.gather(
            ac.post("/v1/audio/speech", json=payload), ac.post("/v1/audio/speech", json=payload)
        )
    assert r1.content == r2.content == b"EDGE"
    edge.assert_called_once()
