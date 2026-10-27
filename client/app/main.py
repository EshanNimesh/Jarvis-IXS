"""Interactive local JARVIS client loop."""

import asyncio
import logging
import re
import threading

from app.services.brain_client import BrainClient
from app.services.executor import ActionExecutor
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ABBREVIATIONS = {
    "mr.", "mrs.", "dr.", "ms.", "vs.", "e.", "e.g.", "i.", "i.e.", "etc."
}


def take_completed_sentence(text: str, *, final: bool = False) -> tuple[str | None, str]:
    """Extract a sentence as soon as its punctuation arrives, preserving abbreviations."""
    boundary: int | None = None
    for match in re.finditer(r"[.!?\n]", text):
        punctuation = match.group()
        if punctuation == ".":
            previous_is_digit = match.start() > 0 and text[match.start() - 1].isdigit()
            next_is_digit = match.end() < len(text) and text[match.end()].isdigit()
            if previous_is_digit and next_is_digit:
                continue
            prefix = text[: match.end()]
            token_match = re.search(r"([A-Za-z.]+)$", prefix)
            if token_match and token_match.group(1).lower() in ABBREVIATIONS:
                continue
        boundary = match.end()
        break

    word_matches = list(re.finditer(r"\S+", text))
    if len(word_matches) >= 12:
        twelfth_word_end = word_matches[11].end()
        has_boundary = twelfth_word_end < len(text) or final
        if has_boundary and (boundary is None or twelfth_word_end < boundary):
            boundary = twelfth_word_end

    if boundary is None:
        return None, text
    sentence = text[:boundary].strip()
    remainder = text[boundary:].lstrip()
    remainder = re.sub(r'^["\'”’»]+\s*', "", remainder)
    return (sentence, remainder) if sentence else (None, text)


async def stream_response_to_tts(
    brain: BrainClient, tts: TTSService, stt: STTService, prompt: str
) -> dict[str, object]:
    """Download, synthesize, and queue completed sentences concurrently."""
    audio_queue: asyncio.Queue[object] = asyncio.Queue()
    audio_end = object()
    barge_in = asyncio.Event()
    monitor_stop = threading.Event()

    async def play_audio_queue() -> None:
        started = False
        monitor_task: asyncio.Task[None] | None = None

        async def monitor_microphone() -> None:
            while not monitor_stop.is_set():
                try:
                    # The microphone monitor is idle during token generation and
                    # only treats voice as a barge-in while JARVIS audio is active.
                    if not await asyncio.to_thread(tts.is_playing):
                        await asyncio.sleep(0.05)
                        continue
                    detected = await asyncio.to_thread(stt.wait_for_voice_activity, monitor_stop)
                    if (
                        detected
                        and not monitor_stop.is_set()
                        and await asyncio.to_thread(tts.is_playing)
                    ):
                        barge_in.set()
                        await asyncio.to_thread(tts.stop)
                        return
                except Exception:
                    logger.exception("Barge-in microphone monitor failed")
                    return

        monitor_task = asyncio.create_task(monitor_microphone())

        while True:
            item = await audio_queue.get()
            if item is audio_end:
                break
            task, sentence = item
            if barge_in.is_set():
                task.cancel()
                continue
            try:
                audio_data = await task
                if barge_in.is_set():
                    continue
                await asyncio.to_thread(tts.play_prepared, audio_data)
                started = True
            except Exception:
                logger.exception("Sentence pre-synthesis failed")
                if barge_in.is_set():
                    continue
                await asyncio.to_thread(tts.wait_until_done)
                await tts.speak(sentence)
                started = False
        monitor_stop.set()
        if monitor_task is not None:
            await monitor_task
        if started and not barge_in.is_set():
            await asyncio.to_thread(tts.wait_until_done)

    playback_task = asyncio.create_task(play_audio_queue())
    buffer = ""
    stream_error: Exception | None = None
    try:
        async for chunk in brain.query_brain_stream(prompt):
            buffer += chunk
            while True:
                sentence, remainder = take_completed_sentence(buffer)
                if sentence is None:
                    break
                buffer = remainder
                if barge_in.is_set():
                    continue
                print(f"[TTS] Speaking response: {sentence}")
                audio_task = asyncio.create_task(tts.synthesize(sentence))
                await audio_queue.put((audio_task, sentence))
    except Exception as exc:
        stream_error = exc

    while True:
        sentence, remainder = take_completed_sentence(buffer, final=True)
        if sentence is None:
            break
        buffer = remainder
        print(f"[TTS] Speaking response: {sentence}")
        audio_task = asyncio.create_task(tts.synthesize(sentence))
        await audio_queue.put((audio_task, sentence))
    if buffer.strip():
        tail = buffer.strip()
        print(f"[TTS] Speaking response: {tail}")
        audio_task = asyncio.create_task(tts.synthesize(tail))
        await audio_queue.put((audio_task, tail))
    await audio_queue.put(audio_end)
    await playback_task
    if stream_error is not None:
        raise stream_error
    return brain.last_stream_result


async def run() -> None:
    brain = BrainClient()
    executor = ActionExecutor()
    tts = TTSService()
    stt = STTService()
    print("JARVIS client ready. Type /voice to speak, or /quit to exit.")

    try:
        while True:
            prompt = (await asyncio.to_thread(input, "You: ")).strip()
            if prompt.lower() in {"exit", "quit", "/quit", "/exit"}:
                print("Goodbye!")
                await asyncio.to_thread(tts.stop)
                break
            if prompt.lower() == "/voice" or not prompt:
                await asyncio.to_thread(tts.stop)
                print("[STT] Listening to microphone...")
                try:
                    prompt = await asyncio.to_thread(stt.listen_and_transcribe)
                except Exception as exc:
                    logger.exception("Voice capture or transcription failed")
                    message = f"I couldn't transcribe the audio: {exc}"
                    print(f"JARVIS: {message}")
                    await tts.speak(message)
                    continue
                if not prompt:
                    print("No speech recognized. Try again.")
                    continue
                print(f"[STT] Transcribed: {prompt}")
            if not prompt:
                continue
            try:
                try:
                    streamed_result = await stream_response_to_tts(brain, tts, stt, prompt)
                except Exception:
                    logger.exception("Streaming query failed; retrying with regular query")
                    speak_text, action = await asyncio.to_thread(brain.query, prompt)
                    streamed_result = {
                        "speak": speak_text,
                        "action": action.get("action", "none"),
                        "payload": action.get("payload", {}),
                    }
                    print(f"[TTS] Speaking response: {speak_text}")
                    await tts.speak(speak_text)
                print("[BRAIN] Received response")
                speak_text = streamed_result.get("speak", "")
                if isinstance(speak_text, str):
                    print(f"JARVIS: {speak_text}")
                action = {
                    "action": streamed_result.get("action", "none"),
                    "payload": streamed_result.get("payload", {}),
                }
                if not isinstance(action["payload"], dict):
                    action["payload"] = {}
                result = await asyncio.to_thread(executor.execute, action)
                logger.info("Action result: %s", result)
            except Exception as exc:
                logger.exception("Query failed")
                message = f"I couldn't complete that request: {exc}"
                print(f"JARVIS: {message}")
                await tts.speak(message)
    finally:
        await brain.close()
        await asyncio.to_thread(tts.close)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
