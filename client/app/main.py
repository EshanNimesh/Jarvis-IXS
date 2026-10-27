"""Interactive local JARVIS client loop."""

import asyncio
import logging

from app.services.brain_client import BrainClient
from app.services.executor import ActionExecutor
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            speak_text, action = await asyncio.to_thread(brain.query, prompt)
            print("[BRAIN] Received response")
            print(f"JARVIS: {speak_text}")
            print("[TTS] Speaking response...")
            await tts.speak(speak_text)
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
