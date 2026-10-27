from fastapi import FastAPI

app = FastAPI(title="JARVIS Central Brain")

@app.get("/")
def read_root():
    return {"status": "online", "core": "JARVIS Brain Server v1.0"}

@app.get("/health")
def health_check():
    return {"health": "ok"}