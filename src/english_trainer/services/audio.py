"""Text-to-speech with Edge TTS and pygame playback (install the ``audio`` extra)."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

VOICE_US = "en-US-JennyNeural"
VOICE_UK = "en-GB-SoniaNeural"
RATE_NORMAL = "+0%"
RATE_SLOW = "-40%"


def audio_available() -> bool:
    """True when both optional audio dependencies are installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("edge_tts", "pygame"))


async def _synthesize(text: str, voice: str, rate: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))


def speak(text: str, *, voice: str = VOICE_US, rate: str = RATE_NORMAL) -> bool:
    """Play ``text`` aloud and block until playback ends. Returns ``False`` on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "speech.mp3"
        try:
            import pygame

            asyncio.run(_synthesize(text, voice, rate, path))
            pygame.mixer.init()
            try:
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.play()
                clock = pygame.time.Clock()
                while pygame.mixer.music.get_busy():
                    clock.tick(10)
                pygame.mixer.music.unload()
            finally:
                pygame.mixer.quit()
        except Exception as exc:  # noqa: BLE001 - audio stacks fail in many platform-specific ways
            logger.info("audio.playback_failed", extra={"error": type(exc).__name__})
            return False
    return True
