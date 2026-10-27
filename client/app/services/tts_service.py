"""Low-latency local speech using pyttsx3."""

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import logging
import tempfile
import threading
import time
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Speak text locally without blocking the async client loop."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
        self._playback_lock = threading.Lock()
        self._edge_lock = asyncio.Lock()
        self._mixer_ready = False
        self._audio_path: Path | None = None
        try:
            import pygame

            pygame.mixer.init()
            self._mixer_ready = True
        except Exception:
            logger.exception("Could not initialize pygame audio mixer")
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file.close()
        self._audio_path = Path(temp_file.name)
        atexit.register(self._cleanup_audio_file)

    def _speak_sync(self, text: str) -> None:
        com_initialized = False
        engine = None
        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialized = True
            except ImportError:
                # pythoncom is only available on Windows; other pyttsx3 drivers
                # do not require explicit COM setup.
                pass

            import pyttsx3

            engine = pyttsx3.init()
            local_voice_ids = {voice.id for voice in engine.getProperty("voices")}
            if settings.tts_voice in local_voice_ids:
                engine.setProperty("voice", settings.tts_voice)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            logger.exception("Local text-to-speech failed")
            raise
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    logger.debug("TTS engine shutdown failed", exc_info=True)
            if com_initialized:
                import pythoncom

                pythoncom.CoUninitialize()

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        try:
            await self._speak_edge(text)
        except Exception:
            logger.exception("Edge TTS failed; falling back to local pyttsx3")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._speak_sync, text)

    async def _speak_edge(self, text: str) -> None:
        import edge_tts

        if not self._mixer_ready or self._audio_path is None:
            raise RuntimeError("pygame audio mixer is unavailable")
        voice = settings.tts_voice or "en-US-GuyNeural"
        async with self._edge_lock:
            await edge_tts.Communicate(text, voice=voice).save(str(self._audio_path))
            await asyncio.to_thread(self._play_mp3_sync, self._audio_path)

    def _play_mp3_sync(self, audio_path: Path) -> None:
        import pygame

        with self._playback_lock:
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.03)
            pygame.mixer.music.unload()

    def _cleanup_audio_file(self) -> None:
        if self._audio_path is not None:
            try:
                self._audio_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove temporary TTS audio file", exc_info=True)
