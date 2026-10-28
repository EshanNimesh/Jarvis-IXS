"""Modal-hosted SSE endpoint for the Muse Glimmer 30B model."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import uuid
from collections.abc import Iterator

import modal

APP_NAME = "jarvis-glimmer"
MODEL_ID = "meta-models/Muse-Glimmer-30B"
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub", "fastapi", "uvicorn")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    min_containers=1,
    volumes={HF_CACHE_PATH: volume},
)
class GlimmerModel:
    """Keep vLLM, tokenizer, and their model weights warm in a GPU container."""

    @modal.enter()
    def load_model(self) -> None:
        import vllm
        from transformers import AutoTokenizer

        engine_args = vllm.AsyncEngineArgs(
            model=MODEL_ID,
            download_dir=HF_CACHE_PATH,
            trust_remote_code=True,
        )
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = vllm.SamplingParams(temperature=0.7, max_tokens=1024)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE_PATH,
            trust_remote_code=True,
        )
        volume.commit()

        # vLLM's async engine must be used from one persistent event loop.
        self._engine_loop = asyncio.new_event_loop()
        self._engine_thread = threading.Thread(
            target=self._engine_loop.run_forever,
            name="glimmer-vllm-loop",
            daemon=True,
        )
        self._engine_thread.start()

    @modal.method()
    def stream_tokens(self, prompt: str, system: str) -> Iterator[str]:
        """Yield incremental generated text through Modal's remote generator."""
        request_id = uuid.uuid4().hex
        token_queue: queue.Queue[tuple[str | None, BaseException | None]] = queue.Queue()

        async def generate() -> None:
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                previous_text = ""
                async for output in self.engine.generate(
                    formatted_prompt,
                    self.sampling_params,
                    request_id,
                ):
                    current_text = output.outputs[0].text
                    delta = current_text[len(previous_text):]
                    previous_text = current_text
                    if delta:
                        token_queue.put((delta, None))
            except asyncio.CancelledError:
                await self.engine.abort(request_id)
                raise
            except Exception as exc:
                token_queue.put((None, exc))
            finally:
                token_queue.put((None, None))

        generation = asyncio.run_coroutine_threadsafe(generate(), self._engine_loop)
        try:
            while True:
                token, error = token_queue.get()
                if error is not None:
                    raise RuntimeError(f"vLLM generation failed: {error}") from error
                if token is None:
                    break
                yield token
        finally:
            if not generation.done():
                generation.cancel()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_sse(body: dict[str, str]):
    """Accept {prompt, system} JSON and stream generated text as SSE."""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    prompt = body.get("prompt")
    system = body.get("system", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=422, detail="'prompt' must be a non-empty string")
    if not isinstance(system, str):
        raise HTTPException(status_code=422, detail="'system' must be a string")

    def events():
        try:
            for token in GlimmerModel().stream_tokens.remote_gen(prompt, system):
                yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logging.exception("Muse Glimmer generation failed")
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
