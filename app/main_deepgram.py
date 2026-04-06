from fastapi import FastAPI, WebSocket

from app.websocket.ws_handler_deepgram import handle_websocket

app = FastAPI(title="Voice Bot (Deepgram)")


@app.get("/")
async def root():
    return {"message": "Voice bot app is running with Deepgram STT/TTS"}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "deepgram"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)
