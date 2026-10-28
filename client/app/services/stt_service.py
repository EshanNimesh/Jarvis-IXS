"""Low-latency local speech-to-text with streaming voice activity detection."""

import logging
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    """Stream microphone audio until trailing silence or the hard time limit."""

    sample_rate = 16_000

    def wait_for_voice_activity(self, stop_event: threading.Event) -> bool:
        """Monitor the microphone until speech is detected or monitoring is stopped."""
        chunk_seconds = 0.1
        chunk_frames = int(chunk_seconds * self.sample_rate)
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            ambient_chunks = []
            for _ in range(2):
                if stop_event.is_set():
                    return False
                block, _overflowed = stream.read(chunk_frames)
                ambient_chunks.append(np.asarray(block, dtype=np.float32).reshape(-1))
            ambient = np.concatenate(ambient_chunks)
            ambient_rms = float(np.sqrt(np.mean(np.square(ambient)))) if ambient.size else 0.0
            threshold = max(0.005, ambient_rms * 1.8)
            while not stop_event.is_set():
                block, _overflowed = stream.read(chunk_frames)
                samples = np.asarray(block, dtype=np.float32).reshape(-1)
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                if rms >= threshold:
                    logger.info("Voice activity detected during TTS playback")
                    return True
        return False

    def __init__(self) -> None:
        # cpu_threads is a WhisperModel setting in faster-whisper, not a
        # transcribe() option.
        self.model = WhisperModel(
            "base.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        self.last_turn_end_at: float | None = None

    def listen_and_transcribe(self, duration_seconds: int = 15) -> str:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero")

        self.last_turn_end_at = None
        capture_started = time.perf_counter()
        max_seconds = min(float(duration_seconds), 15.0)
        chunk_seconds = 0.1
        chunk_frames = int(chunk_seconds * self.sample_rate)
        silence_chunks_required = round(0.8 / chunk_seconds)
        ambient_chunks: list[np.ndarray] = []
        speech_chunks: list[np.ndarray] = []
        speech_started = False
        trailing_silence = 0
        started_at = time.monotonic()

        logger.info("Listening for speech (maximum %.1f seconds)", max_seconds)
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            for _ in range(2):
                block, _overflowed = stream.read(chunk_frames)
                ambient_chunks.append(np.asarray(block, dtype=np.float32).reshape(-1))
            ambient = np.concatenate(ambient_chunks)
            ambient_rms = float(np.sqrt(np.mean(np.square(ambient)))) if ambient.size else 0.0
            speech_threshold = max(0.005, ambient_rms * 1.8)

            while time.monotonic() - started_at < max_seconds:
                block, _overflowed = stream.read(chunk_frames)
                chunk = np.asarray(block, dtype=np.float32).reshape(-1)
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                if rms >= speech_threshold:
                    speech_started = True
                    trailing_silence = 0
                    speech_chunks.append(chunk)
                elif speech_started:
                    trailing_silence += 1
                    if trailing_silence >= silence_chunks_required:
                        self.last_turn_end_at = time.perf_counter()
                        logger.info("[TIMING] 1. Mic Silence / Turn End Detected: +0.000ms")
                        break
                    speech_chunks.append(chunk)

        if speech_started and self.last_turn_end_at is None:
            # The recording cap can end a turn before the silence window does.
            self.last_turn_end_at = time.perf_counter()
            logger.info("[TIMING] 1. Mic Silence / Turn End Detected: +0.000ms (recording cap)")

        if not speech_chunks:
            logger.info(
                "[LATENCY] stt_capture_ms=%.0f result=no_speech",
                (time.perf_counter() - capture_started) * 1000,
            )
            return ""
        if trailing_silence:
            speech_chunks = speech_chunks[:-trailing_silence]
        audio = np.concatenate(speech_chunks) if speech_chunks else np.array([], dtype=np.float32)
        if audio.size == 0:
            logger.info(
                "[LATENCY] stt_capture_ms=%.0f result=empty_audio",
                (time.perf_counter() - capture_started) * 1000,
            )
            return ""

        transcription_started = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 600},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt="JARVIS Sri Lankan English voice assistant technical conversation.",
        )
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
        transcription_completed_at = time.perf_counter()
        if self.last_turn_end_at is not None:
            logger.info(
                "[TIMING] 2. STT Transcription Complete: +%.3fms",
                (transcription_completed_at - self.last_turn_end_at) * 1000,
            )
        logger.info(
            "[LATENCY] stt_capture_ms=%.0f stt_transcribe_ms=%.0f audio_seconds=%.2f",
            (transcription_started - capture_started) * 1000,
            (time.perf_counter() - transcription_started) * 1000,
            audio.size / self.sample_rate,
        )
        return transcription
