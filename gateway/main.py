from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.audio import get_mime_type
from gateway.cache import AudioCache
from gateway.config import AppSettings, settings
from gateway.router import TTSRouter
from gateway.schemas import (
    HealthResponse,
    ModelListResponse,
    ModelObject,
    SpeechRequest,
    VoiceListResponse,
)
from gateway.stt.router import STTRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("universal-speech")

# Global instances
cache = AudioCache(
    enabled=settings.cache.enabled,
    max_entries=settings.cache.max_entries,
    max_memory_mb=settings.cache.max_memory_mb,
    ttl_seconds=settings.cache.ttl_seconds,
)
tts_router = TTSRouter(settings=settings)
stt_router = STTRouter(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Universal Speech Gateway...")
    logger.info("Default TTS Engine: %s | Default Voice: %s", settings.defaults.engine, settings.defaults.voice)
    logger.info("Default STT Engine: %s | Default STT Model: %s", settings.stt.default_engine, settings.stt.default_model)
    logger.info("Cache Enabled: %s (Max: %s entries, %s MB)", settings.cache.enabled, settings.cache.max_entries, settings.cache.max_memory_mb)
    yield
    logger.info("Shutting down Universal Speech Gateway...")


app = FastAPI(
    title="Universal Speech Gateway",
    description="OpenAI-compatible drop-in Speech Gateway supporting Text-to-Speech (Edge-TTS, Kokoro, Google Cloud) and Speech-to-Text (Groq, faster-whisper, Google Cloud).",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    required_key = settings.server.api_key
    if not required_key:
        return  # No auth required

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/")
async def root():
    return {
        "service": "Universal Speech Gateway",
        "version": "0.2.0",
        "description": "1:1 drop-in replacement for OpenAI Speech & Audio API endpoints",
        "endpoints": {
            "speech": "POST /v1/audio/speech",
            "transcriptions": "POST /v1/audio/transcriptions",
            "translations": "POST /v1/audio/translations",
            "models": "GET /v1/models",
            "voices": "GET /v1/audio/voices (or /v1/voices)",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    tts_health = await tts_router.get_engine_health()
    stt_health = await stt_router.get_engine_health()
    all_ok = any(tts_health.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        version="0.2.0",
        service="universal-speech",
        default_engine=settings.defaults.engine,
        default_voice=settings.defaults.voice,
        engines=tts_health,
        stt=stt_health,
        cache=cache.get_stats(),
    )


@app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(verify_api_key)])
async def list_models():
    model_ids = [
        # TTS models
        "edge-tts",
        "kokoro",
        "google-cloud",
        "tts-1",
        "tts-1-hd",
        # STT models
        "whisper-1",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "groq",
        "local-whisper",
    ]
    if settings.engines.piper.enabled:
        model_ids.append("piper")

    models = [ModelObject(id=m) for m in model_ids]
    return ModelListResponse(data=models)


@app.get("/v1/audio/voices", response_model=VoiceListResponse, dependencies=[Depends(verify_api_key)])
@app.get("/v1/voices", response_model=VoiceListResponse, dependencies=[Depends(verify_api_key)])
async def list_voices(
    all: bool = Query(False, description="Return all upstream voices instead of curated subset"),
    locale: Optional[str] = Query(None, description="Filter voices by locale (e.g. en-US, ml-IN)"),
    engine: Optional[str] = Query(None, description="Filter voices by engine (e.g. google-cloud, edge-tts)"),
):
    voices = await tts_router.get_curated_voices(all=all, locale=locale, engine=engine)
    return VoiceListResponse(voices=voices)


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(request: SpeechRequest):
    if not request.input.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")

    engine, resolved_voice = tts_router.resolve_engine_and_voice(request.model, request.voice)
    target_format = tts_router.resolve_format(engine, request.response_format)
    speed = request.speed or 1.0

    # Cache Lookup
    cache_key = cache.generate_key(
        engine=engine.engine_name,
        voice=resolved_voice,
        text=request.input,
        speed=speed,
        response_format=target_format,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        audio_bytes, fmt = cached
        mime = get_mime_type(fmt)
        return Response(
            content=audio_bytes,
            media_type=mime,
            headers={
                "X-Cache": "HIT",
                "X-Engine": engine.engine_name,
                "Content-Disposition": f'attachment; filename="speech.{fmt}"',
            },
        )

    # Cache Miss -> Synthesize
    try:
        audio_bytes, out_fmt, used_engine = await tts_router.synthesize(
            text=request.input,
            model=request.model,
            voice=request.voice,
            speed=speed,
            response_format=request.response_format,
        )

        # Store in cache
        cache.put(cache_key, audio_bytes, out_fmt)

        mime = get_mime_type(out_fmt)
        return Response(
            content=audio_bytes,
            media_type=mime,
            headers={
                "X-Cache": "MISS",
                "X-Engine": used_engine,
                "Content-Disposition": f'attachment; filename="speech.{out_fmt}"',
            },
        )
    except Exception as e:
        logger.error("TTS Synthesis error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {str(e)}",
        )


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form("whisper-1", description="Model name or alias"),
    language: Optional[str] = Form(None, description="Audio language code (e.g. en, fr, es)"),
    prompt: Optional[str] = Form(None, description="Optional prompt guide"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    timestamp_granularities: Optional[List[str]] = Form(None),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    filename = file.filename or "audio.wav"

    try:
        result, used_engine = await stt_router.transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )

        headers = {
            "X-Engine": used_engine,
        }

        if response_format in ("json", "verbose_json"):
            return JSONResponse(content=result, headers=headers)
        return Response(content=str(result), media_type="text/plain", headers=headers)

    except Exception as e:
        logger.error("STT Transcription error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"STT transcription failed: {str(e)}",
        )


@app.post("/v1/audio/translations", dependencies=[Depends(verify_api_key)])
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: str = Form("whisper-1", description="Model name or alias"),
    prompt: Optional[str] = Form(None, description="Optional prompt guide"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    filename = file.filename or "audio.wav"

    try:
        result, used_engine = await stt_router.translate(
            audio_bytes=audio_bytes,
            filename=filename,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        headers = {
            "X-Engine": used_engine,
        }

        if response_format in ("json", "verbose_json"):
            return JSONResponse(content=result, headers=headers)
        return Response(content=str(result), media_type="text/plain", headers=headers)

    except Exception as e:
        logger.error("STT Translation error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"STT translation failed: {str(e)}",
        )


@app.post("/v1/audio/embeddings", dependencies=[Depends(verify_api_key)])
async def create_audio_embeddings():
    """Future audio search & embeddings stub."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Audio embeddings endpoint is reserved for future audio search and vector indexing.",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(settings.server.port)))
    uvicorn.run(
        "gateway.main:app",
        host=settings.server.host,
        port=port,
        reload=False,
    )
