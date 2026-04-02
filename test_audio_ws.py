import asyncio
import json
import wave

import websockets


AUDIO_FILE = "sample.wav"
WS_URL = "ws://127.0.0.1:8000/ws"
CHUNK_SIZE = 3200


async def send_audio():
    async with websockets.connect(WS_URL) as websocket:
        with wave.open(AUDIO_FILE, "rb") as wav_file:
            print("Channels:", wav_file.getnchannels())
            print("Sample width:", wav_file.getsampwidth())
            print("Frame rate:", wav_file.getframerate())

            while True:
                chunk = wav_file.readframes(CHUNK_SIZE // 2)
                if not chunk:
                    break
                await websocket.send(chunk)

        await websocket.send("stop")

        response = await websocket.recv()
        try:
            print(json.loads(response))
        except Exception:
            print(response)


if __name__ == "__main__":
    asyncio.run(send_audio())
