"""Agent query endpoints."""

import json

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


async def _stream_query(prompt: str):
    try:
        async for text_chunk in llm_service.query_llm_stream(prompt):
            yield json.dumps({"text": text_chunk}, ensure_ascii=False) + "\n"
    except Exception as exc:
        yield json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n"


@router.post("/query_stream")
async def query_agent_stream(payload: QueryRequest) -> StreamingResponse:
    """Stream provider-generated content as newline-delimited JSON events."""
    return StreamingResponse(
        _stream_query(payload.prompt),
        media_type="application/x-ndjson",
    )
