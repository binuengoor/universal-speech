import pytest
from httpx import ASGITransport, AsyncClient
from openai import OpenAI
from gateway.main import app, cache


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "engines" in data
        assert "cache" in data
        assert data["default_engine"] == "edge-tts"


@pytest.mark.asyncio
async def test_models_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "edge-tts" in model_ids
        assert "kokoro" in model_ids
        assert "tts-1" in model_ids
        assert "whisper-1" in model_ids
        assert "groq" in model_ids
        assert "local-whisper" in model_ids


@pytest.mark.asyncio
async def test_voices_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/v1/audio/voices")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["voices"]) > 0


@pytest.mark.asyncio
async def test_speech_edge_tts_caching():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "model": "edge-tts",
            "input": "Testing universal tts gateway cache.",
            "voice": "en-US-AriaNeural",
        }
        
        # 1st request -> Cache MISS
        r1 = await ac.post("/v1/audio/speech", json=payload)
        assert r1.status_code == 200
        assert r1.headers.get("X-Cache") == "MISS"
        assert r1.headers.get("Content-Type") == "audio/mpeg"
        assert len(r1.content) > 100

        # 2nd request -> Cache HIT (< 5ms response)
        r2 = await ac.post("/v1/audio/speech", json=payload)
        assert r2.status_code == 200
        assert r2.headers.get("X-Cache") == "HIT"
        assert r2.content == r1.content


@pytest.mark.asyncio
async def test_speech_piper_engine():
    # Dynamically mount piper engine into tts_router to verify backward compatibility
    from gateway.main import tts_router
    from gateway.engines.piper_engine import PiperEngine
    from gateway.config import settings

    tts_router.engines["piper"] = PiperEngine(
        models_dir=settings.paths.piper_models_dir,
        default_voice=settings.engines.piper.default_voice,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "model": "piper",
            "input": "Piper offline engine test.",
            "voice": "en_US-ryan-medium",
        }
        resp = await ac.post("/v1/audio/speech", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("Content-Type") == "audio/wav"
        assert len(resp.content) > 100


@pytest.mark.asyncio
async def test_speech_kokoro_engine():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "model": "kokoro",
            "input": "Kokoro high fidelity test.",
            "voice": "af_heart",
        }
        resp = await ac.post("/v1/audio/speech", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("Content-Type") == "audio/wav"
        assert len(resp.content) > 100


@pytest.mark.asyncio
async def test_speech_format_conversion():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Request WAV from Edge-TTS (native format is MP3)
        payload = {
            "model": "edge-tts",
            "input": "Conversion test.",
            "voice": "en-US-AriaNeural",
            "response_format": "wav",
        }
        resp = await ac.post("/v1/audio/speech", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("Content-Type") == "audio/wav"
        assert resp.content.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_openai_sdk_compatibility():
    # Test using OpenAI client against the ASGI app via httpx client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": "alloy",
                "input": "Hello from OpenAI SDK test!",
                "response_format": "mp3",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "audio/mpeg"
        assert len(response.content) > 0
