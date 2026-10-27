"""Agent query endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/v1", tags=["agent"])


class QueryRequest(BaseModel):
    prompt: str
    context: dict[str, object]


@router.post("/query")
def query_agent(payload: QueryRequest) -> dict[str, object]:
    """Pass the incoming prompt to the configured LLM provider."""
    llm_response = llm_service.generate_response(payload.prompt)
    return {
        "status": "success",
        "prompt": payload.prompt,
        "response": llm_response,
    }


@router.post("/query_stream")
async def query_agent_stream(payload: QueryRequest) -> StreamingResponse:
    """Stream provider SSE events directly to the client."""
    return StreamingResponse(
        llm_service.query_llm_stream(payload.prompt),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
