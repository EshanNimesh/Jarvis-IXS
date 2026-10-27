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


async def stream_response_to_tts(brain: BrainClient, tts: TTSService, prompt: str) -> dict[str, object]:
    """Download Brain chunks in the background and speak completed sentences."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    def enqueue(item: object) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

    def download() -> None:
        try:
            for chunk in brain.query_brain_stream(prompt):
                enqueue(chunk)
        except Exception as exc:
            enqueue(exc)
        finally:
            enqueue(sentinel)

    download_task = asyncio.create_task(asyncio.to_thread(download))
    buffer = ""
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            await download_task
            raise item
        chunk = str(item)
        buffer += chunk
        while match := re.search(r"[.!?]", buffer):
            sentence = buffer[: match.end()].strip()
            buffer = buffer[match.end() :]
            if sentence:
                print(f"[TTS] Speaking response: {sentence}")
                await tts.speak(sentence)

    await download_task
    if buffer.strip():
        print(f"[TTS] Speaking response: {buffer.strip()}")
        await tts.speak(buffer.strip())
    return brain.last_stream_result


async def run() -> None:
    brain = BrainClient()
    executor = ActionExecutor()
    tts = TTSService()
    stt = STTService()
    print("JARVIS client ready. Type /voice to speak, or /quit to exit.")

    while True:
        prompt = (await asyncio.to_thread(input, "You: ")).strip()
        if prompt.lower() in {"exit", "quit", "/quit", "/exit"}:
            print("Goodbye!")
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


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
