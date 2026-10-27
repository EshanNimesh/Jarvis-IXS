"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import agent, health

app = FastAPI(title="JARVIS Central Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(agent.router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "online", "core": "JARVIS Brain Server v1.0"}
