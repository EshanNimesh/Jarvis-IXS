"""Agent query endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["agent"])


class QueryRequest(BaseModel):
    prompt: str
    context: dict[str, object]


@router.post("/query")
def query_agent(payload: QueryRequest) -> dict[str, str]:
    """Return a placeholder response while agent processing is developed."""
    return {
        "status": "success",
        "action": "think",
        "response": "Processing query...",
    }
