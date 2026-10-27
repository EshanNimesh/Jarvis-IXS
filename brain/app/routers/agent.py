"""Agent query endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

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
