"""Text-to-speech: edge-tts (online) with pygame playback, falls back silently."""

import asyncio
import os
import tempfile
from typing import Optional


VOICE_US = "en-US-JennyNeural"
VOICE_UK = "en-GB-SoniaNeural"


RATE_NORMAL = "+0%"
RATE_SLOW   = "-40%"


async def _generate_mp3(text: str, voice: str, path: str, rate: str = RATE_NORMAL) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(path)


def speak(text: str, voice: Optional[str] = None, rate: str = RATE_NORMAL) -> bool:
    """Play text aloud. Returns True on success, False if audio is unavailable."""
    if not voice:
        voice = VOICE_US
    try:
        import pygame

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()

        asyncio.run(_generate_mp3(text, voice, tmp_path, rate=rate))

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        return True
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def audio_available() -> bool:
    """Quick check: returns True if edge-tts and pygame are importable."""
    try:
        import edge_tts  # noqa: F401
        import pygame    # noqa: F401
        return True
    except ImportError:
        return False
