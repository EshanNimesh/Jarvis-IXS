"""Health and service metadata endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])
_started_at = datetime.now(timezone.utc)


@router.get("/health")
def health_check() -> dict[str, object]:
    """Report service health, core metadata, and process uptime."""
    now = datetime.now(timezone.utc)
    return {
        "status": "ok",
        "core": "JARVIS Brain Server v1.0",
        "environment": settings.env,
        "uptime": {
            "status": "running",
            "started_at": _started_at.isoformat(),
            "seconds": round((now - _started_at).total_seconds(), 2),
        },
    }
