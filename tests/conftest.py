import importlib.util
import os
from pathlib import Path

import pytest

from gateway.config import settings
from gateway.main import cache, stt_router, tts_router

RUN_NETWORK = os.getenv("RUN_NETWORK_TESTS") == "1"


def _kokoro_models_present() -> bool:
    return Path(settings.paths.kokoro_model_path).exists() and Path(settings.paths.kokoro_voices_path).exists()


def _piper_ready() -> bool:
    models = Path(settings.paths.piper_models_dir)
    return importlib.util.find_spec("piper") is not None and any(models.glob("*.onnx"))


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs live third-party services (set RUN_NETWORK_TESTS=1)")
    config.addinivalue_line("markers", "kokoro: needs downloaded Kokoro model files")
    config.addinivalue_line("markers", "piper: needs piper-tts and a downloaded Piper voice")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "network" in item.keywords and not RUN_NETWORK:
            item.add_marker(pytest.mark.skip(reason="network test (set RUN_NETWORK_TESTS=1)"))
        if "kokoro" in item.keywords and not _kokoro_models_present():
            item.add_marker(pytest.mark.skip(reason="Kokoro models not downloaded"))
        if "piper" in item.keywords and not _piper_ready():
            item.add_marker(pytest.mark.skip(reason="piper-tts or Piper voice not installed"))


@pytest.fixture(autouse=True)
def isolate_global_state():
    """Tests mutate the module-level routers; restore them so failures don't cascade."""
    tts_engines = dict(tts_router.engines)
    stt_engines = dict(stt_router.engines)
    tts_methods = {n: dict(vars(e)) for n, e in tts_engines.items()}
    stt_methods = {n: dict(vars(e)) for n, e in stt_engines.items()}
    tts_router._cooldowns.clear()
    stt_router._cooldowns.clear()
    cache.clear()
    yield
    for name, engine in tts_engines.items():
        vars(engine).clear()
        vars(engine).update(tts_methods[name])
    for name, engine in stt_engines.items():
        vars(engine).clear()
        vars(engine).update(stt_methods[name])
    tts_router.engines.clear()
    tts_router.engines.update(tts_engines)
    stt_router.engines.clear()
    stt_router.engines.update(stt_engines)
    tts_router._cooldowns.clear()
    stt_router._cooldowns.clear()
    cache.clear()
