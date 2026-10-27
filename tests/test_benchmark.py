import asyncio
import io
import time
import pytest
from httpx import ASGITransport, AsyncClient
import soundfile as sf
from gateway.main import app, cache


async def measure_request(ac: AsyncClient, payload: dict) -> dict:
    t0 = time.perf_counter()
    resp = await ac.post("/v1/audio/speech", json=payload)
    elapsed = time.perf_counter() - t0
    
    assert resp.status_code == 200
    cache_header = resp.headers.get("X-Cache", "UNKNOWN")
    engine_header = resp.headers.get("X-Engine", "UNKNOWN")
    size_kb = len(resp.content) / 1024.0

    duration_sec = 0.0
    try:
        if resp.headers.get("Content-Type") == "audio/wav":
            with io.BytesIO(resp.content) as buf:
                info = sf.info(buf)
                duration_sec = info.duration
    except Exception:
        pass

    rtf = (elapsed / duration_sec) if duration_sec > 0 else 0.0

    return {
        "engine": engine_header,
        "model": payload.get("model"),
        "voice": payload.get("voice"),
        "latency_ms": round(elapsed * 1000, 2),
        "cache": cache_header,
        "size_kb": round(size_kb, 2),
        "duration_sec": round(duration_sec, 2),
        "rtf": round(rtf, 4),
    }


@pytest.mark.network
@pytest.mark.kokoro
@pytest.mark.asyncio
async def test_benchmark_suite():
    cache.clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        text = "Universal TTS Gateway delivers ultra-low latency text to speech for smart assistants and AI applications."
        
        benchmarks = []

        # 1. Edge-TTS Cold
        res_edge_cold = await measure_request(ac, {
            "model": "edge-tts",
            "input": text,
            "voice": "en-US-AriaNeural",
        })
        benchmarks.append(res_edge_cold)

        # 2. Edge-TTS Warm (LRU Cache Hit)
        res_edge_warm = await measure_request(ac, {
            "model": "edge-tts",
            "input": text,
            "voice": "en-US-AriaNeural",
        })
        benchmarks.append(res_edge_warm)
        assert res_edge_warm["cache"] == "HIT"
        assert res_edge_warm["latency_ms"] < 20.0  # In-memory LRU < 20ms

        # 3. Piper Offline Engine
        res_piper = await measure_request(ac, {
            "model": "piper",
            "input": text,
            "voice": "en_US-ryan-medium",
        })
        benchmarks.append(res_piper)

        # 4. Kokoro High-Fidelity Engine
        res_kokoro = await measure_request(ac, {
            "model": "kokoro",
            "input": text,
            "voice": "af_heart",
        })
        benchmarks.append(res_kokoro)

        print("\n" + "=" * 80)
        print(f"{'ENGINE':<12} | {'VOICE':<18} | {'LATENCY (ms)':<14} | {'CACHE':<6} | {'SIZE (KB)':<10} | {'RTF'}")
        print("-" * 80)
        for b in benchmarks:
            print(f"{b['engine']:<12} | {b['voice']:<18} | {b['latency_ms']:<14} | {b['cache']:<6} | {b['size_kb']:<10} | {b['rtf']}")
        print("=" * 80 + "\n")
