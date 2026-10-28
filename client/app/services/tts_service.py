"""Local Kokoro ONNX speech synthesis with pygame playback."""

import atexit
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)
DEFAULT_KOKORO_VOICE = "am_michael"
MIXER_FREQUENCY = 24_000
MIXER_FALLBACK_FREQUENCY = 44_100
MIXER_SIZE = -16
MIXER_CHANNELS = 1


def initialize_pygame_mixer() -> tuple[int, int, int]:
    """Try 24 kHz then 44.1 kHz, accepting the actual mono or stereo format."""
    import pygame

    def supported(config: tuple[int, int, int] | None) -> bool:
        return bool(
            config is not None
            and config[0] in (MIXER_FREQUENCY, MIXER_FALLBACK_FREQUENCY)
            and config[1] == MIXER_SIZE
            and config[2] in (1, 2)
        )

    current = pygame.mixer.get_init()
    if supported(current):
        return current

    for frequency in (MIXER_FREQUENCY, MIXER_FALLBACK_FREQUENCY):
        current = pygame.mixer.get_init()
        if current is not None:
            pygame.mixer.quit()
        try:
            pygame.mixer.init(
                frequency=frequency, size=MIXER_SIZE,
                channels=MIXER_CHANNELS, buffer=512,
            )
        except Exception:
            logger.warning("Could not initialize pygame mixer at %d Hz", frequency, exc_info=True)
            continue
        current = pygame.mixer.get_init()
        if supported(current):
            return current
        logger.warning("Pygame mixer opened unsupported format %s", current)
        pygame.mixer.quit()
    raise RuntimeError("Could not initialize pygame mixer at 24 kHz or 44.1 kHz")


class TTSService:
    """Load Kokoro once, synthesize speech locally, and queue PCM for pygame."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-tts")
        self._playback_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._playback_generation = 0
        self._mixer_ready = False
        self.actual_sample_rate = MIXER_FREQUENCY
        self.actual_channels = MIXER_CHANNELS
        self._channel = None
        self._sounds: list[object] = []
        self._playback_queue: queue.Queue[
            tuple[int, bytes | None, float | None, int | None, threading.Event | None] | None
        ] = queue.Queue()
        self._closed = False
        self._first_playback_timing_logged = False
        self._kokoro = None
        self._kokoro_load_error: Exception | None = None

        try:
            import pygame

            mixer_config = initialize_pygame_mixer()
            self.actual_sample_rate, _sample_size, self.actual_channels = mixer_config
            self._channel = pygame.mixer.find_channel(True)
            self._mixer_ready = True
            logger.info(
                "TTS audio format: %d Hz, %d channel, signed 16-bit PCM",
                self.actual_sample_rate, self.actual_channels,
            )
        except Exception:
            logger.exception("Could not initialize pygame audio mixer")

        try:
            from kokoro_onnx import Kokoro

            model_path = settings.kokoro_model_path
            voices_path = settings.kokoro_voices_path
            if not model_path.is_file() or not voices_path.is_file():
                raise FileNotFoundError(
                    "Kokoro model files are missing. Place kokoro-v1.0.onnx and "
                    "voices-v1.0.bin in client/models/ (or set KOKORO_MODEL_PATH "
                    "and KOKORO_VOICES_PATH)."
                )
            self._kokoro = Kokoro(str(model_path), str(voices_path))
            logger.info("Kokoro ONNX model loaded from %s", model_path)
        except Exception as exc:
            self._kokoro_load_error = exc
            logger.error("Could not load local Kokoro ONNX model: %s", exc)

        self._playback_thread = threading.Thread(
            target=self._playback_worker, name="jarvis-audio-playback", daemon=True
        )
        if self._mixer_ready:
            self._playback_thread.start()
        atexit.register(self.close)

    def begin_turn(self) -> None:
        with self._state_lock:
            self._first_playback_timing_logged = False

    def queue_pcm_chunk(
        self, pcm_chunk: bytes, *, timing_started_at: float | None = None,
        sentence_number: int | None = None,
    ) -> None:
        """Queue signed-16-bit PCM already matched to the active mixer format."""
        if not pcm_chunk or not self._mixer_ready or self._closed:
            return
        with self._state_lock:
            generation = self._playback_generation
        self._playback_queue.put((generation, pcm_chunk, timing_started_at, sentence_number, None))

    def _playback_worker(self) -> None:
        import pygame

        while True:
            item = self._playback_queue.get()
            if item is None:
                self._playback_queue.task_done()
                return
            generation, pcm_chunk, timing_started_at, sentence_number, finished = item
            try:
                if pcm_chunk is None:
                    while self._generation_is_current(generation) and self._channel.get_busy():
                        time.sleep(0.002)
                    with self._playback_lock:
                        if self._generation_is_current(generation):
                            self._sounds.clear()
                    if finished is not None:
                        finished.set()
                    continue
                if not self._generation_is_current(generation):
                    continue
                sound = pygame.mixer.Sound(buffer=pcm_chunk)
                with self._playback_lock:
                    while self._channel.get_busy() and self._channel.get_queue() is not None:
                        if not self._generation_is_current(generation):
                            break
                        time.sleep(0.001)
                    if not self._generation_is_current(generation):
                        continue
                    if self._channel.get_busy():
                        self._channel.queue(sound)
                    else:
                        self._channel.play(sound)
                    self._sounds.append(sound)
                    if sentence_number == 1 and timing_started_at is not None and not self._first_playback_timing_logged:
                        self._first_playback_timing_logged = True
                        logger.info(
                            "[TIMING] 5. Pygame Audio Playback Started: +%.3fms (TOTAL TIME)",
                            (time.perf_counter() - timing_started_at) * 1000,
                        )
            except Exception:
                logger.exception("Could not play Kokoro PCM chunk")
            finally:
                self._playback_queue.task_done()

    def _generation_is_current(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._playback_generation

    def _synthesize_pcm(self, text: str) -> bytes:
        """Run Kokoro and convert its float samples to the pygame mixer format."""
        if self._kokoro is None:
            raise RuntimeError(f"Kokoro is unavailable: {self._kokoro_load_error}")
        voice = settings.tts_voice or DEFAULT_KOKORO_VOICE
        samples, source_rate = self._kokoro.create(
            text, voice=voice, speed=1.0, lang="en-us"
        )
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return b""

        # Kokoro emits mono float audio; resample and duplicate channels as needed
        # so raw PCM matches the exact pygame mixer format (avoids pitch changes).
        if int(source_rate) != self.actual_sample_rate:
            output_length = max(1, round(samples.size * self.actual_sample_rate / int(source_rate)))
            source_positions = np.arange(samples.size, dtype=np.float64)
            target_positions = np.linspace(0, samples.size - 1, output_length)
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = np.rint(pcm * 32767.0).astype("<i2")
        if self.actual_channels == 2:
            pcm = np.repeat(pcm[:, None], 2, axis=1)
        return pcm.tobytes()

    async def synthesize_stream(
        self, text: str, *, timing_started_at: float | None = None,
        sentence_number: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield Kokoro-generated signed-16-bit PCM from memory."""
        synthesis_started = time.perf_counter()
        try:
            if not self._mixer_ready:
                raise RuntimeError("pygame audio mixer is unavailable")
            pcm = await asyncio.to_thread(self._synthesize_pcm, text)
            if pcm and sentence_number == 1 and timing_started_at is not None:
                logger.info(
                    "[TIMING] 4. TTS Sentence 1 First Audio Byte Ready: +%.3fms",
                    (time.perf_counter() - timing_started_at) * 1000,
                )
            # Push short PCM blocks to pygame as soon as local synthesis completes.
            bytes_per_frame = 2 * self.actual_channels
            block_size = 4096 - (4096 % bytes_per_frame)
            for offset in range(0, len(pcm), block_size):
                yield pcm[offset:offset + block_size]
        finally:
            logger.info("[LATENCY] kokoro_synthesis_ms=%.0f", (time.perf_counter() - synthesis_started) * 1000)

    def finish_playback(self) -> None:
        if not self._mixer_ready or self._channel is None:
            return
        with self._state_lock:
            generation = self._playback_generation
        finished = threading.Event()
        self._playback_queue.put((generation, None, None, None, finished))
        finished.wait()

    def is_playing(self) -> bool:
        try:
            return bool(self._channel is not None and self._channel.get_busy())
        except Exception:
            return False

    def wait_until_done(self) -> None:
        self.finish_playback()

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        try:
            async for pcm_chunk in self.synthesize_stream(text, sentence_number=1):
                self.queue_pcm_chunk(pcm_chunk, sentence_number=1)
            await asyncio.to_thread(self.wait_until_done)
        except Exception:
            logger.exception("Kokoro TTS failed; falling back to local pyttsx3")
            await asyncio.to_thread(self.stop)
            await self.speak_local(text)

    async def speak_local(self, text: str) -> None:
        """Use the local system speech engine only if Kokoro cannot synthesize."""
        started = time.perf_counter()
        await asyncio.get_running_loop().run_in_executor(self._executor, self._speak_sync, text)
        logger.info("[LATENCY] pyttsx3_fallback_ms=%.0f", (time.perf_counter() - started) * 1000)

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
            engine.say(text)
            engine.runAndWait()
        finally:
            if engine is not None:
                engine.stop()
            if com_initialized:
                import pythoncom
                pythoncom.CoUninitialize()

    def stop(self) -> None:
        with self._state_lock:
            self._playback_generation += 1
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                if self._channel is not None:
                    self._channel.stop()
            with self._playback_lock:
                self._sounds.clear()
        except Exception:
            logger.exception("Could not stop TTS playback")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._mixer_ready:
            self._playback_queue.put(None)
            self._playback_thread.join(timeout=1.0)
        if self._kokoro is not None:
            try:
                self._kokoro.voices.close()
            except Exception:
                logger.debug("Could not close Kokoro voice resources", exc_info=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
