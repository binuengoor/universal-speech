from unittest.mock import AsyncMock, patch

import pytest

import io
import struct
import wave

from gateway.engines.phonon_engine import (
    PHONON_EN_VOICES,
    PhononEngine,
    normalize_wav_header,
)
from gateway.errors import InvalidRequestError


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text or (content.decode("utf-8", "ignore") if content else "")


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient used as an async context manager."""

    last_call = None

    def __init__(self, response, **_kwargs):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        _FakeClient.last_call = (url, json)
        return self.response

    async def get(self, url):
        _FakeClient.last_call = (url, None)
        return self.response


def _engine() -> PhononEngine:
    return PhononEngine(base_url="http://phonon-tts:8880/", default_voice="Easton", timeout_seconds=5.0)


def test_engine_identity():
    engine = _engine()
    assert engine.engine_name == "phonon"
    assert engine.native_format == "wav"
    assert engine.base_url == "http://phonon-tts:8880"
    assert engine.default_voice == "Easton"


@pytest.mark.parametrize(
    "voice,expected",
    [
        ("Easton", "Easton"),
        ("easton", "Easton"),
        ("  JUNIPER ", "Juniper"),
        ("Captain Silas Blackwater", "Captain Silas Blackwater"),
        ("nope", None),
        ("", None),
    ],
)
def test_voice_validation(voice, expected):
    engine = _engine()
    assert engine.canonical_voice(voice) == expected
    assert engine.is_valid_voice(voice) is (expected is not None)


@pytest.mark.asyncio
async def test_get_voices_covers_english_catalogue():
    voices = await _engine().get_voices()
    assert len(voices) == len(PHONON_EN_VOICES) == 37
    assert {v.engine for v in voices} == {"phonon"}
    assert all(v.language in ("en-US", "en-GB") for v in voices)
    assert all(v.sample_rate == 24000 for v in voices)
    assert any(v.id == "Easton" and v.gender == "male" for v in voices)


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["wav", "mp3", "opus"])
async def test_native_formats_pass_through_without_transcoding(fmt):
    engine = _engine()
    with patch.object(engine, "_post", AsyncMock(return_value=b"AUDIO")) as post, patch(
        "gateway.engines.phonon_engine.convert_audio_ffmpeg", AsyncMock()
    ) as convert:
        audio, out_fmt = await engine.synthesize_bytes(text="hi", voice="Easton", response_format=fmt)

    assert (audio, out_fmt) == (b"AUDIO", fmt)
    post.assert_awaited_once_with("hi", "Easton", fmt)
    convert.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_format_is_fetched_as_wav_then_converted():
    engine = _engine()
    with patch.object(engine, "_post", AsyncMock(return_value=b"WAV")) as post, patch(
        "gateway.engines.phonon_engine.convert_audio_ffmpeg", AsyncMock(return_value=b"FLAC")
    ) as convert:
        audio, out_fmt = await engine.synthesize_bytes(text="hi", voice="Easton", response_format="flac")

    assert (audio, out_fmt) == (b"FLAC", "flac")
    post.assert_awaited_once_with("hi", "Easton", "wav")
    convert.assert_awaited_once_with(b"WAV", "wav", "flac")


@pytest.mark.asyncio
async def test_speed_other_than_one_is_retimed_locally():
    """The model has no rate control, so the gateway must not forward speed."""
    engine = _engine()
    with patch.object(engine, "_post", AsyncMock(return_value=b"WAV")) as post, patch.object(
        engine, "_apply_speed", AsyncMock(return_value=b"FAST")
    ) as retime:
        audio, out_fmt = await engine.synthesize_bytes(
            text="hi", voice="Easton", speed=1.5, response_format="mp3"
        )

    assert (audio, out_fmt) == (b"FAST", "mp3")
    post.assert_awaited_once_with("hi", "Easton", "wav")
    retime.assert_awaited_once_with(b"WAV", 1.5, "mp3")


@pytest.mark.asyncio
async def test_speed_one_skips_retiming():
    engine = _engine()
    with patch.object(engine, "_post", AsyncMock(return_value=b"WAV")), patch.object(
        engine, "_apply_speed", AsyncMock()
    ) as retime:
        await engine.synthesize_bytes(text="hi", voice="Easton", speed=1.0, response_format="wav")
    retime.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_voice_is_rejected_before_any_network_call():
    engine = _engine()
    with patch.object(engine, "_post", AsyncMock()) as post:
        with pytest.raises(ValueError):
            await engine.synthesize_bytes(text="hi", voice="NotAVoice")
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_upstream_4xx_becomes_invalid_request():
    engine = _engine()
    fake = _FakeClient(_FakeResponse(400, text="unknown voice"))
    with patch("gateway.engines.phonon_engine.httpx.AsyncClient", lambda **kw: fake):
        with pytest.raises(InvalidRequestError):
            await engine._post("hi", "Easton", "wav")


@pytest.mark.asyncio
async def test_upstream_5xx_is_an_engine_failure():
    engine = _engine()
    fake = _FakeClient(_FakeResponse(503, text="loading"))
    with patch("gateway.engines.phonon_engine.httpx.AsyncClient", lambda **kw: fake):
        with pytest.raises(RuntimeError):
            await engine._post("hi", "Easton", "wav")


@pytest.mark.asyncio
async def test_empty_audio_is_an_error():
    engine = _engine()
    fake = _FakeClient(_FakeResponse(200, content=b""))
    with patch("gateway.engines.phonon_engine.httpx.AsyncClient", lambda **kw: fake):
        with pytest.raises(RuntimeError):
            await engine._post("hi", "Easton", "wav")


@pytest.mark.asyncio
async def test_health_check_hits_health_endpoint():
    engine = _engine()
    fake = _FakeClient(_FakeResponse(200, content=b'{"status":"ok"}'))
    with patch("gateway.engines.phonon_engine.httpx.AsyncClient", lambda **kw: fake):
        assert await engine.health_check() is True
    assert _FakeClient.last_call[0] == "http://phonon-tts:8880/health"

    from gateway.engines.phonon_engine import httpx as real_httpx

    with patch(
        "gateway.engines.phonon_engine.httpx.AsyncClient",
        side_effect=real_httpx.ConnectError("refused"),
    ):
        assert await engine.health_check() is False


@pytest.mark.asyncio
async def test_payload_never_sends_speed():
    engine = _engine()
    fake = _FakeClient(_FakeResponse(200, content=b"WAV"))
    with patch("gateway.engines.phonon_engine.httpx.AsyncClient", lambda **kw: fake):
        await engine._post("hi", "Easton", "wav")
    _, payload = _FakeClient.last_call
    assert "speed" not in payload
    assert payload == {"input": "hi", "voice": "Easton", "response_format": "wav"}


def _streaming_wav(frames: int, sample_rate: int = 24000) -> bytes:
    """A WAV the way ptts-openai-server emits it: real frames, sentinel sizes."""
    payload = b"\x00\x01" * frames
    header = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )
    return header + payload


def test_normalize_wav_header_fixes_streaming_sentinel():
    raw = _streaming_wav(frames=24000)  # one second at 24 kHz
    fixed = normalize_wav_header(raw)

    assert struct.unpack("<I", fixed[4:8])[0] == len(raw) - 8
    idx = fixed.find(b"data")
    assert struct.unpack("<I", fixed[idx + 4 : idx + 8])[0] == len(raw) - idx - 8
    # The frames are untouched, so a normal reader now reports the right duration.
    with wave.open(io.BytesIO(fixed)) as w:
        assert w.getnframes() == 24000
        assert w.getnframes() / w.getframerate() == pytest.approx(1.0)
    assert fixed[idx + 8 :] == raw[idx + 8 :]


@pytest.mark.parametrize("payload", [b"", b"ID3not a wav", b"RIFFtooshort"])
def test_normalize_wav_header_leaves_non_wav_alone(payload):
    assert normalize_wav_header(payload) == payload


def test_router_resolves_phonon_model_and_voice_map():
    from gateway.config import settings
    from gateway.router import TTSRouter

    original = settings.engines.phonon.enabled
    settings.engines.phonon.enabled = True
    try:
        router = TTSRouter(settings)
        assert "phonon" in router.engines

        plan = router.plan("phonon", "Easton", 1.0, None)
        assert plan.engine.engine_name == "phonon"
        assert plan.voice == "Easton"

        # OpenAI voice names map onto a real Phonon voice for this engine.
        mapped = router.plan("phonon", "onyx", 1.0, None)
        assert mapped.voice == "Easton"

        # A Phonon voice name alone is enough to select the engine.
        by_voice = router.plan(None, "Maeve", 1.0, None)
        assert by_voice.engine.engine_name == "phonon"
        assert by_voice.voice == "Maeve"
        by_lower = router.plan("tts-1", "easton", 1.0, None)
        assert by_lower.engine.engine_name == "phonon"
        assert by_lower.voice == "easton"

        with pytest.raises(InvalidRequestError):
            router.plan("phonon", "definitely-not-a-voice", 1.0, None)
    finally:
        settings.engines.phonon.enabled = original