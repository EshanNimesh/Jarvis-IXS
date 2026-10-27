"""Low-latency local speech-to-text with streaming voice activity detection."""

import logging
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    """Stream microphone audio until trailing silence or the hard time limit."""

    sample_rate = 16_000

    def __init__(self) -> None:
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    def listen_and_transcribe(self, duration_seconds: int = 8) -> str:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero")

        max_seconds = min(duration_seconds, 8)
        chunk_seconds = 0.1
        chunk_frames = int(chunk_seconds * self.sample_rate)
        silence_chunks_required = int(0.8 / chunk_seconds)
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
                        break
                    speech_chunks.append(chunk)

        if not speech_chunks:
            return ""
        if trailing_silence:
            speech_chunks = speech_chunks[:-trailing_silence]
        audio = np.concatenate(speech_chunks) if speech_chunks else np.array([], dtype=np.float32)
        if audio.size == 0:
            return ""

        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
