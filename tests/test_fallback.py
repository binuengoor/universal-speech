import pytest
from unittest.mock import AsyncMock, patch
from gateway.config import AppSettings
from gateway.router import TTSRouter


@pytest.mark.asyncio
async def test_circuit_breaker_fallback_on_edge_failure():
    settings = AppSettings()
    settings.circuit_breaker.enabled = True
    settings.circuit_breaker.fallback_engine = "kokoro"
    settings.circuit_breaker.fallback_voice = "af_heart"

    router = TTSRouter(settings=settings)

    # Mock edge engine failure
    router.engines["edge-tts"].synthesize_bytes = AsyncMock(side_effect=TimeoutError("Edge-TTS timed out"))

    # Mock kokoro engine success
    router.engines["kokoro"].synthesize_bytes = AsyncMock(return_value=(b"RIFF_mock_kokoro_wav", "wav"))

    audio_bytes, fmt, engine_name = await router.synthesize(
        text="Testing circuit breaker fallback to Kokoro",
        model="edge-tts",
        voice="en-US-AriaNeural",
    )

    assert engine_name == "kokoro"
    assert audio_bytes == b"RIFF_mock_kokoro_wav"
    assert fmt == "wav"
    router.engines["edge-tts"].synthesize_bytes.assert_called_once()
    router.engines["kokoro"].synthesize_bytes.assert_called_once()


@pytest.mark.asyncio
async def test_voice_and_engine_resolution():
    settings = AppSettings()
    settings.engines.piper.enabled = True  # Enable for legacy voice resolution test
    router = TTSRouter(settings=settings)

    # 1. Standard OpenAI voice alias
    engine, voice = router.resolve_engine_and_voice(model="tts-1", voice="alloy")
    assert engine.engine_name == "edge-tts"
    assert voice == "en-US-AriaNeural"

    # 2. Kokoro voice prefix
    engine, voice = router.resolve_engine_and_voice(model=None, voice="af_heart")
    assert engine.engine_name == "kokoro"
    assert voice == "af_heart"

    # 3. Piper short alias
    engine, voice = router.resolve_engine_and_voice(model=None, voice="ryan")
    assert engine.engine_name == "piper"
    assert voice == "en_US-ryan-medium"

    # 4. Explicit model override
    engine, voice = router.resolve_engine_and_voice(model="piper", voice="custom-voice")
    assert engine.engine_name == "piper"
    assert voice == "custom-voice"

    # 5. Google Cloud explicit model
    engine, voice = router.resolve_engine_and_voice(model="google-cloud", voice=None)
    assert engine.engine_name == "google-cloud"
    assert voice == "en-US-Neural2-F"

    # 6. Google Cloud voice heuristic (Neural2)
    engine, voice = router.resolve_engine_and_voice(model=None, voice="en-US-Neural2-F")
    assert engine.engine_name == "google-cloud"
    assert voice == "en-US-Neural2-F"

    # 7. Google Cloud voice heuristic (Journey)
    engine, voice = router.resolve_engine_and_voice(model=None, voice="en-US-Journey-D")
    assert engine.engine_name == "google-cloud"
    assert voice == "en-US-Journey-D"

    # 8. Google Cloud short alias
    engine, voice = router.resolve_engine_and_voice(model=None, voice="neural2-f")
    assert engine.engine_name == "google-cloud"
    assert voice == "en-US-Neural2-F"


@pytest.mark.asyncio
async def test_circuit_breaker_fallback_on_google_failure():
    settings = AppSettings()
    settings.circuit_breaker.enabled = True
    settings.circuit_breaker.fallback_engine = "kokoro"
    settings.circuit_breaker.fallback_voice = "af_heart"

    router = TTSRouter(settings=settings)

    # Mock google engine failure
    router.engines["google-cloud"].synthesize_bytes = AsyncMock(side_effect=TimeoutError("Google Cloud TTS timed out"))

    # Mock kokoro engine success
    router.engines["kokoro"].synthesize_bytes = AsyncMock(return_value=(b"RIFF_mock_kokoro_wav", "wav"))

    audio_bytes, fmt, engine_name = await router.synthesize(
        text="Testing google fallback to Kokoro",
        model="google-cloud",
        voice="en-US-Neural2-F",
    )

    assert engine_name == "kokoro"
    assert audio_bytes == b"RIFF_mock_kokoro_wav"
    assert fmt == "wav"
    router.engines["google-cloud"].synthesize_bytes.assert_called_once()
    router.engines["kokoro"].synthesize_bytes.assert_called_once()

