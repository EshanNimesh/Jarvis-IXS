"""Interactive local JARVIS client loop."""

import asyncio
import logging
import re

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
    """Extract the first completed sentence without splitting known abbreviations."""
    for match in re.finditer(r"[.!?\n]", text):
        punctuation = match.group()
        if punctuation != "\n":
            if match.end() == len(text):
                if not final:
                    continue
            elif not text[match.end()].isspace():
                continue
        if match.group() == ".":
            prefix = text[: match.end()]
            token_match = re.search(r"([A-Za-z.]+)$", prefix)
            if token_match and token_match.group(1).lower() in ABBREVIATIONS:
                continue
        sentence = text[: match.end()].strip()
        if sentence:
            remainder = text[match.end() :].lstrip()
            remainder = re.sub(r'^["\'”’»]+\s*', "", remainder)
            return sentence, remainder
    return None, text


async def stream_response_to_tts(brain: BrainClient, tts: TTSService, prompt: str) -> dict[str, object]:
    """Download, synthesize, and queue completed sentences concurrently."""
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue[object] = asyncio.Queue()
    audio_queue: asyncio.Queue[object] = asyncio.Queue()
    token_end, audio_end = object(), object()

    def download() -> None:
        try:
            for chunk in brain.query_brain_stream(prompt):
                asyncio.run_coroutine_threadsafe(token_queue.put(chunk), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(token_queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(token_queue.put(token_end), loop).result()

    download_task = asyncio.create_task(asyncio.to_thread(download))

    async def play_audio_queue() -> None:
        started = False
        while True:
            item = await audio_queue.get()
            if item is audio_end:
                break
            task, sentence = item
            try:
                audio_data = await task
                await asyncio.to_thread(tts.play_prepared, audio_data)
                started = True
            except Exception:
                logger.exception("Sentence pre-synthesis failed")
                await asyncio.to_thread(tts.wait_until_done)
                await tts.speak(sentence)
                started = False
        if started:
            await asyncio.to_thread(tts.wait_until_done)

    playback_task = asyncio.create_task(play_audio_queue())
    buffer = ""
    stream_error: Exception | None = None
    while True:
        item = await token_queue.get()
        if item is token_end:
            break
        if isinstance(item, Exception):
            stream_error = item
            continue
        chunk = str(item)
        buffer += chunk
        while True:
            sentence, remainder = take_completed_sentence(buffer)
            if sentence is None:
                break
            buffer = remainder
            print(f"[TTS] Speaking response: {sentence}")
            audio_task = asyncio.create_task(tts.synthesize(sentence))
            await audio_queue.put((audio_task, sentence))

    await download_task
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
                    streamed_result = await stream_response_to_tts(brain, tts, prompt)
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
        await asyncio.to_thread(tts.close)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
