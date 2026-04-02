from pathlib import Path
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import traceback
import wave
import base64
import json
import asyncio
import shutil


def save_wav_file(
    audio_bytes: bytes,
    output_dir: str = "recordings",
    sample_rate: int = 8000,
    channels: int = 1,
    sample_width: int = 2,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = Path(output_dir) / filename

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return str(file_path)


async def send_sample_audio(websocket: WebSocket, wav_path: str = "sample_reply.wav") -> None:
    path = Path(wav_path)
    if not path.exists():
        print(f"Sample reply file not found: {wav_path}")
        return

    fixed_reply = Path("/tmp/python_reply.wav")
    shutil.copyfile(path, fixed_reply)
    print(f"Copied reply audio to {fixed_reply}")

    audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

    payload = {
        "type": "streamAudio",
        "data": {
            "audioDataType": "wav",
            "sampleRate": 8000,
            "audioData": audio_b64,
        },
    }

    await websocket.send_text(json.dumps(payload))
    print("Sent streamAudio response to FreeSWITCH")


async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")

    audio_buffer = bytearray()
    playback_sent = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                print("WebSocket disconnect frame received")
                break

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)
                print(f"Received chunk: {len(chunk)} bytes, total={len(audio_buffer)}")

                # Send playback once, shortly after audio starts arriving
                if not playback_sent and len(audio_buffer) >= 3200:
                    await asyncio.sleep(0.2)
                    await send_sample_audio(websocket, "sample_reply.wav")
                    playback_sent = True

            elif message.get("text") is not None:
                print("Received text:", message["text"])

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", repr(e))
        traceback.print_exc()

    finally:
        print("Final total bytes:", len(audio_buffer))
        if audio_buffer:
            try:
                path = save_wav_file(bytes(audio_buffer), sample_rate=8000)
                print("Saved WAV:", path)
            except Exception as e:
                print("Save error:", repr(e))
                traceback.print_exc()

        try:
            await websocket.close()
        except Exception:
            pass
