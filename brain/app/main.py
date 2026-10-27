"""FastAPI application entry point."""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.routers import agent, health

app = FastAPI(title="JARVIS Central Brain")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(agent.router)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            prompt = await websocket.receive_text()
            try:
                async for token in agent.llm_service.query_llm_tokens(prompt):
                    await websocket.send_text(token)
                await websocket.send_text(json.dumps({"event": "done"}))
            except Exception as exc:
                logger.exception("WebSocket LLM stream failed")
                await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        logger.info("Brain WebSocket client disconnected")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "online", "core": "JARVIS Brain Server v1.0"}
