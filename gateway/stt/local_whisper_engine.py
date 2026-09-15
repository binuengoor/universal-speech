import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from gateway.stt.base import BaseSTTEngine

logger = logging.getLogger(__name__)


class LocalWhisperEngine(BaseSTTEngine):
    """
    Offline local STT using faster-whisper (CTranslate2) with INT8 CPU quantization.
    Features lazy model loading and zero PyTorch overhead.
    """

    engine_name = "local-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        models_dir: str = "models/whisper",
    ):
        self.model_size = model_size
        self.default_model = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.models_dir = models_dir
        self._model = None
        self._init_lock = asyncio.Lock()

    def _get_or_load_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        download_path = Path(self.models_dir)
        download_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading faster-whisper model '%s' (device=%s, compute_type=%s, threads=%d)...",
            self.model_size,
            self.device,
            self.compute_type,
            self.cpu_threads,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(download_path),
        )
        logger.info("faster-whisper model '%s' successfully loaded.", self.model_size)
        return self._model

    def _format_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"

    def _run_transcription_sync(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> Any:
        model = self._get_or_load_model()
        audio_file = io.BytesIO(audio_bytes)

        segments_generator, info = model.transcribe(
            audio_file,
            task=task,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=5,
            word_timestamps=True if response_format == "verbose_json" else False,
        )

        segments_list = list(segments_generator)
        full_text = " ".join(seg.text.strip() for seg in segments_list).strip()

        if response_format == "text":
            return full_text

        if response_format == "json":
            return {"text": full_text}

        if response_format == "verbose_json":
            segments_data = []
            words_data = []
            for seg in segments_list:
                segments_data.append({
                    "id": seg.id,
                    "seek": seg.seek,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "tokens": seg.tokens,
                    "temperature": seg.temperature,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                    "no_speech_prob": seg.no_speech_prob,
                })
                if seg.words:
                    for w in seg.words:
                        words_data.append({
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                        })

            return {
                "task": task,
                "language": info.language or (language or "en"),
                "duration": info.duration,
                "text": full_text,
                "words": words_data if words_data else None,
                "segments": segments_data,
            }

        if response_format == "srt":
            lines = []
            for i, seg in enumerate(segments_list, start=1):
                start_str = self._format_srt_time(seg.start)
                end_str = self._format_srt_time(seg.end)
                lines.append(f"{i}\n{start_str} --> {end_str}\n{seg.text.strip()}\n")
            return "\n".join(lines)

        if response_format == "vtt":
            lines = ["WEBVTT\n"]
            for seg in segments_list:
                start_str = self._format_vtt_time(seg.start)
                end_str = self._format_vtt_time(seg.end)
                lines.append(f"{start_str} --> {end_str}\n{seg.text.strip()}\n")
            return "\n".join(lines)

        return {"text": full_text}

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Any:
        async with self._init_lock:
            pass

        return await asyncio.to_thread(
            self._run_transcription_sync,
            audio_bytes=audio_bytes,
            task="transcribe",
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )

    async def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Any:
        async with self._init_lock:
            pass

        return await asyncio.to_thread(
            self._run_transcription_sync,
            audio_bytes=audio_bytes,
            task="translate",
            language=None,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
        )

    async def health_check(self) -> bool:
        # Ready if faster_whisper library is importable
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
