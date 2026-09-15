import io
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch
import soundfile as sf
import numpy as np

from gateway.main import app, stt_router
from gateway.config import settings


def generate_dummy_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a clean synthetic sine wave in memory as valid WAV bytes."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_stt_router():
    stt_router._cooldowns.clear()


@pytest.mark.asyncio
async def test_stt_transcriptions_groq():
    wav_bytes = generate_dummy_wav()

    # Mock groq engine response
    stt_router.engines["groq"].transcribe = AsyncMock(
        return_value={"text": "Hello, this is a test transcription from Groq."}
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}

        resp = await ac.post("/v1/audio/transcriptions", files=files, data=data)
        assert resp.status_code == 200
        assert resp.headers.get("X-Engine") == "groq"
        json_data = resp.json()
        assert json_data["text"] == "Hello, this is a test transcription from Groq."
        stt_router.engines["groq"].transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_stt_transcriptions_formats():
    wav_bytes = generate_dummy_wav()

    # Mock groq engine response for plain text and srt
    stt_router.engines["groq"].transcribe = AsyncMock(
        return_value="Plain text transcription."
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "text"}

        resp = await ac.post("/v1/audio/transcriptions", files=files, data=data)
        assert resp.status_code == 200
        assert resp.text == "Plain text transcription."


@pytest.mark.asyncio
async def test_stt_translations():
    wav_bytes = generate_dummy_wav()

    stt_router.engines["groq"].translate = AsyncMock(
        return_value={"text": "Good morning and welcome."}
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("spanish.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}

        resp = await ac.post("/v1/audio/translations", files=files, data=data)
        assert resp.status_code == 200
        assert resp.headers.get("X-Engine") == "groq"
        assert resp.json()["text"] == "Good morning and welcome."
        stt_router.engines["groq"].translate.assert_called_once()


@pytest.mark.asyncio
async def test_stt_fallback_to_local_whisper_on_groq_failure():
    wav_bytes = generate_dummy_wav()

    # Mock Groq failure (e.g. rate limit 429 or timeout)
    stt_router.engines["groq"].transcribe = AsyncMock(
        side_effect=RuntimeError("Groq 429 Too Many Requests: Rate limit exceeded")
    )

    # Mock Local Whisper success
    stt_router.engines["local-whisper"].transcribe = AsyncMock(
        return_value={"text": "Local whisper fallback succeeded."}
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1"}

        resp = await ac.post("/v1/audio/transcriptions", files=files, data=data)
        assert resp.status_code == 200
        assert resp.headers.get("X-Engine") == "local-whisper"
        assert resp.json()["text"] == "Local whisper fallback succeeded."

        # Verify Groq was placed in cooldown
        assert not stt_router.is_engine_available("groq")

        # 2nd call should immediately skip Groq with zero delay
        resp2 = await ac.post("/v1/audio/transcriptions", files={"file": ("test2.wav", wav_bytes, "audio/wav")}, data=data)
        assert resp2.status_code == 200
        assert resp2.headers.get("X-Engine") == "local-whisper"
        # Groq transcribe should still only have been called once!
        stt_router.engines["groq"].transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_stt_empty_file_bad_request():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("empty.wav", b"", "audio/wav")}
        resp = await ac.post("/v1/audio/transcriptions", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_audio_embeddings_stub():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/audio/embeddings")
        assert resp.status_code == 501
        assert "reserved" in resp.json()["detail"].lower()
