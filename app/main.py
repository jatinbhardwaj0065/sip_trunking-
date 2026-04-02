from fastapi import FastAPI, WebSocket
from app.websocket.ws_handler import handle_websocket

app = FastAPI(title="Voice Bot")


@app.get("/")
async def root():
    return {"message": "Voice bot app is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)
