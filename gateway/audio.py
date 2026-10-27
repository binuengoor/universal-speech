import asyncio
import io
import logging
from typing import Dict, Optional
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

MIME_TYPES: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


def get_mime_type(audio_format: str) -> str:
    return MIME_TYPES.get(audio_format.lower(), "application/octet-stream")


async def convert_audio_ffmpeg(
    audio_bytes: bytes,
    source_format: str,
    target_format: str,
    sample_rate: Optional[int] = None,
) -> bytes:
    """
    Convert audio bytes between formats using an async ffmpeg subprocess.
    """
    if source_format.lower() == target_format.lower():
        return audio_bytes

    # Map target formats to ffmpeg output format arguments
    format_map = {
        "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"],
        "wav": ["-f", "wav", "-c:a", "pcm_s16le"],
        "opus": ["-f", "opus", "-c:a", "libopus", "-b:a", "64k"],
        "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
        "flac": ["-f", "flac"],
        "pcm": ["-f", "s16le", "-c:a", "pcm_s16le"],
    }

    output_args = format_map.get(target_format.lower(), ["-f", target_format.lower()])
    if target_format.lower() == "pcm" and not sample_rate:
        # Raw PCM has no header: match OpenAI's contract (24kHz, 16-bit, mono).
        sample_rate = 24000
        output_args = ["-ac", "1", *output_args]
    rate_args = ["-ar", str(sample_rate)] if sample_rate else []

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        *rate_args,
        *output_args,
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=audio_bytes)
        if proc.returncode != 0:
            logger.error("FFmpeg conversion failed: %s", stderr.decode(errors="ignore"))
            raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode(errors='ignore')}")
        return stdout
    except FileNotFoundError:
        # Fallback to soundfile if ffmpeg is not installed and formats are supported (e.g., wav/flac)
        logger.warning("FFmpeg binary not found, attempting in-memory fallback")
        return fallback_convert_soundfile(audio_bytes, source_format, target_format)


def fallback_convert_soundfile(audio_bytes: bytes, source_format: str, target_format: str) -> bytes:
    if source_format.lower() == "wav" and target_format.lower() == "pcm":
        with io.BytesIO(audio_bytes) as in_f:
            data, _ = sf.read(in_f, dtype="int16")
            return data.tobytes()
    elif target_format.lower() in ("wav", "flac"):
        with io.BytesIO(audio_bytes) as in_f:
            data, samplerate = sf.read(in_f)
        out_f = io.BytesIO()
        sf.write(out_f, data, samplerate, format=target_format.upper())
        return out_f.getvalue()
    raise RuntimeError(f"Cannot convert from {source_format} to {target_format} without ffmpeg installed.")
