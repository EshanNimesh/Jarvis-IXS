"""Local speech synthesis with Edge TTS and pygame playback."""

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import threading
import time
from app.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Synthesize sentences asynchronously and play queued audio in order."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
        self._playback_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._playback_generation = 0
        self._mixer_ready = False
        self._current_audio_file: BytesIO | None = None
        try:
            import pygame

            pygame.mixer.init()
            self._mixer_ready = True
        except Exception:
            logger.exception("Could not initialize pygame audio mixer")
        atexit.register(self.close)

    async def synthesize(self, text: str) -> bytes:
        """Fetch sentence audio bytes without waiting for current playback."""
        if not self._mixer_ready:
            raise RuntimeError("pygame audio mixer is unavailable")
        import edge_tts

        voice = settings.tts_voice or "en-US-GuyNeural"
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice=voice).stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk["data"])
        if not audio:
            raise RuntimeError("Edge TTS returned an empty audio response")
        return bytes(audio)

    def play_prepared(self, audio_data: bytes) -> None:
        """Play an in-memory MP3 after the current sentence finishes."""
        import pygame

        with self._state_lock:
            generation = self._playback_generation
        with self._playback_lock:
            while pygame.mixer.music.get_busy():
                with self._state_lock:
                    if generation != self._playback_generation:
                        return
                time.sleep(0.01)
            with self._state_lock:
                if generation != self._playback_generation:
                    return
                audio_file = BytesIO(audio_data)
                pygame.mixer.music.load(audio_file, "mp3")
                self._current_audio_file = audio_file
                pygame.mixer.music.play()

    def is_playing(self) -> bool:
        """Return whether JARVIS audio is currently playing through pygame."""
        try:
            import pygame

            return bool(pygame.mixer.get_init() and pygame.mixer.music.get_busy())
        except Exception:
            return False

    def wait_until_done(self) -> None:
        with self._playback_lock:
            while self.is_playing():
                time.sleep(0.02)
            self._current_audio_file = None

    async def speak(self, text: str) -> None:
        """Synthesize and play one sentence, falling back to local pyttsx3."""
        if not text.strip():
            return
        try:
            audio_data = await self.synthesize(text)
            await asyncio.to_thread(self.play_prepared, audio_data)
            await asyncio.to_thread(self.wait_until_done)
        except Exception:
            logger.exception("Edge TTS failed; falling back to local pyttsx3")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._speak_sync, text)

    def _speak_sync(self, text: str) -> None:
        com_initialized = False
        engine = None
        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
                com_initialized = True
            except ImportError:
                pass
            import pyttsx3

            engine = pyttsx3.init()
            local_voice_ids = {voice.id for voice in engine.getProperty("voices")}
            if settings.tts_voice in local_voice_ids:
                engine.setProperty("voice", settings.tts_voice)
            engine.say(text)
            engine.runAndWait()
        finally:
            if engine is not None:
                engine.stop()
            if com_initialized:
                import pythoncom

                pythoncom.CoUninitialize()

    def stop(self) -> None:
        """Stop pygame playback immediately."""
        with self._state_lock:
            self._playback_generation += 1
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            with self._playback_lock:
                self._current_audio_file = None
        except Exception:
            logger.exception("Could not stop TTS playback")

    def close(self) -> None:
        self.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
