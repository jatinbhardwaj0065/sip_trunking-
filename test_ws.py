import asyncio
import json
import websockets


async def test():
    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:
        await websocket.send("hello bot")
        response = await websocket.recv()

        try:
            print(json.loads(response))
        except Exception:
            print(response)


if __name__ == "__main__":
    asyncio.run(test())
