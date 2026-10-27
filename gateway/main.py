from contextlib import asynccontextmanager
import asyncio
import hmac
import logging
import os
from typing import Dict, List, Optional
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from gateway.audio import get_mime_type
from gateway.cache import AudioCache
from gateway.config import AppSettings, settings
from gateway.errors import InvalidRequestError, PayloadTooLargeError
from gateway.router import SynthesisResult, TTSRouter
from gateway.schemas import (
    HealthResponse,
    ModelListResponse,
    ModelObject,
    SpeechRequest,
    VoiceListResponse,
)
from gateway.stt.base import ALL_RESPONSE_FORMATS
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

# Single-flight: identical concurrent cache misses share one synthesis.
_inflight: Dict[str, "asyncio.Future[SynthesisResult]"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Universal Speech Gateway...")
    logger.info("Default TTS Engine: %s | Default Voice: %s", settings.defaults.engine, settings.defaults.voice)
    logger.info("Default STT Engine: %s | Default STT Model: %s", settings.stt.default_engine, settings.stt.default_model)
    logger.info("Cache Enabled: %s (Max: %s entries, %s MB)", settings.cache.enabled, settings.cache.max_entries, settings.cache.max_memory_mb)
    if not settings.server.api_key:
        logger.warning(
            "⚠ No API key configured (set SPEECH_API_KEY). The gateway accepts unauthenticated requests; "
            "only expose it on a trusted network."
        )
    yield
    logger.info("Shutting down Universal Speech Gateway...")


app = FastAPI(
    title="Universal Speech Gateway",
    description="OpenAI-compatible drop-in Speech Gateway supporting Text-to-Speech (Edge-TTS, Kokoro, Google Cloud) and Speech-to-Text (Groq, faster-whisper, Google Cloud).",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS. Auth is a bearer token (not cookies), so credentials are never needed;
# wildcard origins combined with credentials would let any site act as the user.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials="*" not in settings.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cache", "X-Engine", "X-Voice", "X-Fallback"],
)


def _error_body(message: str, status_code: int) -> dict:
    """OpenAI-style error envelope; `detail` is kept for FastAPI-style clients."""
    err_type = "invalid_request_error" if 400 <= status_code < 500 else "server_error"
    return {"error": {"message": message, "type": err_type, "code": status_code}, "detail": message}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(str(exc.detail), exc.status_code),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = _error_body("Invalid request parameters", 422)
    body["detail"] = exc.errors()
    return JSONResponse(status_code=422, content=jsonable(body))


def jsonable(obj):
    from fastapi.encoders import jsonable_encoder

    return jsonable_encoder(obj)


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
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), required_key.encode()):
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
    """Liveness: always 200 while the process serves requests; `status` reports engine state."""
    tts_health = await tts_router.get_engine_health()
    stt_health = await stt_router.get_engine_health()
    default_ok = tts_health.get(settings.defaults.engine, False)
    return HealthResponse(
        status="ok" if default_ok else "degraded",
        version="0.2.0",
        service="universal-speech",
        default_engine=settings.defaults.engine,
        default_voice=settings.defaults.voice,
        engines=tts_health,
        stt=stt_health,
        cache=cache.get_stats(),
    )


@app.get("/ready")
async def readiness_check():
    """Readiness: 503 unless at least one TTS engine is usable right now."""
    tts_health = await tts_router.get_engine_health()
    ready = any(tts_health.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"ready": ready, "engines": tts_health},
    )


TTS_ENGINE_MODELS = ("edge-tts", "kokoro", "google-cloud", "piper")
STT_MODELS = {
    "groq": ("groq", "whisper-large-v3-turbo", "whisper-large-v3"),
    "local-whisper": ("local-whisper",),
    "google-cloud": (),
}


@app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(verify_api_key)])
async def list_models(type: Optional[str] = Query(None, description="Filter by 'tts' or 'stt'")):
    """Only advertise models backed by an engine that is enabled in this deployment."""
    tts_ids: List[str] = []
    if tts_router.engines:
        tts_ids = [m for m in TTS_ENGINE_MODELS if m in tts_router.engines] + ["tts-1", "tts-1-hd"]

    stt_ids: List[str] = []
    if stt_router.engines:
        stt_ids.append("whisper-1")
        for engine_name, ids in STT_MODELS.items():
            if engine_name in stt_router.engines:
                stt_ids.extend(i for i in ids if i not in stt_ids)

    if type == "tts":
        model_ids = tts_ids
    elif type == "stt":
        model_ids = stt_ids
    else:
        model_ids = tts_ids + stt_ids

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


def _audio_response(result: SynthesisResult, cache_status: str) -> Response:
    headers = {
        "X-Cache": cache_status,
        "X-Engine": result.engine,
        "X-Voice": result.voice,
        "Content-Disposition": f'attachment; filename="speech.{result.format}"',
    }
    if result.fallback:
        # Tell clients this is substitute audio so they don't persist it as the requested voice.
        headers["X-Fallback"] = "true"
        headers["Cache-Control"] = "no-store"
    return Response(content=result.audio, media_type=get_mime_type(result.format), headers=headers)


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    request: SpeechRequest,
    x_no_fallback: Optional[str] = Header(None, description="Set to 1 to fail instead of using the fallback voice"),
):
    if not request.input.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")

    try:
        plan = tts_router.plan(request.model, request.voice, request.speed, request.response_format)
    except InvalidRequestError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    allow_fallback = (x_no_fallback or "").lower() not in ("1", "true", "yes")

    cache_key = cache.generate_key(
        engine=plan.engine.engine_name,
        voice=plan.voice,
        text=request.input,
        speed=plan.speed,
        response_format=plan.response_format,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        audio_bytes, fmt = cached
        return _audio_response(
            SynthesisResult(audio_bytes, fmt, plan.engine.engine_name, plan.voice), "HIT"
        )

    flight_key = f"{cache_key}:{int(allow_fallback)}"
    pending = _inflight.get(flight_key)
    if pending is not None:
        try:
            return _audio_response(await asyncio.shield(pending), "MISS")
        except Exception:
            pass  # The leader failed; retry independently below so errors map cleanly.

    loop = asyncio.get_running_loop()
    future: "asyncio.Future[SynthesisResult]" = loop.create_future()
    _inflight[flight_key] = future
    try:
        result = await tts_router.synthesize_plan(request.input, plan, allow_fallback=allow_fallback)
        # Never cache fallback audio under the requested voice's key.
        if not result.fallback:
            cache.put(cache_key, result.audio, result.format)
        future.set_result(result)
        return _audio_response(result, "MISS")
    except InvalidRequestError as e:
        future.set_exception(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        future.set_exception(e)
        logger.error("TTS Synthesis error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS synthesis failed: engine '{plan.engine.engine_name}' is unavailable",
        )
    finally:
        _inflight.pop(flight_key, None)
        if future.done() and not future.cancelled():
            future.exception()  # Mark retrieved so asyncio doesn't warn when nobody awaited it.


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload while enforcing server.max_upload_mb without buffering beyond it."""
    limit = settings.server.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > limit:
        raise PayloadTooLargeError(f"Audio file exceeds {settings.server.max_upload_mb} MB limit")
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"Audio file exceeds {settings.server.max_upload_mb} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _stt_response(result, response_format: str, used_engine: str) -> Response:
    headers = {"X-Engine": used_engine}
    if response_format in ("json", "verbose_json"):
        if not isinstance(result, dict):
            result = {"text": str(result)}
        return JSONResponse(content=result, headers=headers)
    if isinstance(result, dict):
        result = result.get("text", "")
    media_type = "text/vtt" if response_format == "vtt" else "text/plain"
    return Response(content=str(result), media_type=media_type, headers=headers)


async def _run_stt(task: str, file: UploadFile, response_format: str, **kwargs) -> Response:
    if response_format not in ALL_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported response_format '{response_format}'. Use one of: {', '.join(sorted(ALL_RESPONSE_FORMATS))}",
        )
    try:
        audio_bytes = await _read_upload(file)
    except PayloadTooLargeError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    filename = os.path.basename(file.filename or "") or "audio.wav"
    fn = stt_router.transcribe if task == "transcribe" else stt_router.translate
    try:
        result, used_engine = await fn(
            audio_bytes=audio_bytes, filename=filename, response_format=response_format, **kwargs
        )
    except InvalidRequestError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("STT %s error: %s", task, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"STT {task} failed: no engine could process the audio",
        )
    return _stt_response(result, response_format, used_engine)


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form("whisper-1", description="Model name or alias"),
    language: Optional[str] = Form(None, description="Audio language code (e.g. en, fr, es)"),
    prompt: Optional[str] = Form(None, description="Optional prompt guide"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    timestamp_granularities: Optional[List[str]] = Form(None),
    timestamp_granularities_sdk: Optional[List[str]] = Form(None, alias="timestamp_granularities[]"),
):
    # The OpenAI SDK sends the array as `timestamp_granularities[]`.
    granularities = timestamp_granularities or timestamp_granularities_sdk
    return await _run_stt(
        "transcribe",
        file,
        response_format,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        timestamp_granularities=granularities,
    )


@app.post("/v1/audio/translations", dependencies=[Depends(verify_api_key)])
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: str = Form("whisper-1", description="Model name or alias"),
    prompt: Optional[str] = Form(None, description="Optional prompt guide"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
):
    return await _run_stt(
        "translate",
        file,
        response_format,
        model=model,
        prompt=prompt,
        temperature=temperature,
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
