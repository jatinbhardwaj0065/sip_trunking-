import asyncio
import base64
import json
import wave
from pathlib import Path

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK


AUDIO_FILE = "myvoice.wav"
WS_URL = "ws://127.0.0.1:8000/ws"
TARGET_SAMPLE_RATE = 8000
CHUNK_MS = 100
REPLY_OUTPUT_DIR = Path("recordings")
RESPONSE_AUDIO_MODE = "stream"


def load_pcm_audio(path: str) -> tuple[bytes, int]:
    """Load a WAV file and normalize it to mono 16-bit PCM."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    print("Channels:", channels)
    print("Sample width:", sample_width)
    print("Frame rate:", sample_rate)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    return samples.tobytes(), sample_rate


def resample_pcm(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample mono 16-bit PCM to any target sample rate using linear interpolation."""
    if from_rate == to_rate:
        return pcm_data

    samples = np.frombuffer(pcm_data, dtype=np.int16)
    if samples.size == 0:
        return b""

    # Fast paths for common conversions
    if from_rate == 16000 and to_rate == 8000:
        return samples[::2].astype(np.int16).tobytes()

    if from_rate == 8000 and to_rate == 16000:
        return np.repeat(samples, 2).astype(np.int16).tobytes()

    # Generic resampling using linear interpolation
    # Calculate number of samples in output
    num_output_samples = int(len(samples) * to_rate / from_rate)

    # Create input and output time grids
    input_indices = np.arange(len(samples))
    output_indices = np.linspace(0, len(samples) - 1, num_output_samples)

    # Linear interpolation
    resampled = np.interp(output_indices, input_indices, samples)
    return resampled.astype(np.int16).tobytes()


async def send_audio_realtime(websocket, pcm_data: bytes, sample_rate: int) -> None:
    """Send PCM chunks at realtime pace."""
    chunk_bytes = int(sample_rate * 2 * CHUNK_MS / 1000)
    total_chunks = (len(pcm_data) + chunk_bytes - 1) // chunk_bytes

    print("Outbound sample rate:", sample_rate)
    print("Chunk bytes:", chunk_bytes)
    print("Total chunks:", total_chunks)

    for index, start in enumerate(range(0, len(pcm_data), chunk_bytes), start=1):
        chunk = pcm_data[start:start + chunk_bytes]
        await websocket.send(chunk)
        print(f"Sent chunk {index}/{total_chunks}: {len(chunk)} bytes")
        await asyncio.sleep(CHUNK_MS / 1000)

    # Leave the websocket open so STT/LLM/TTS can finish and stream the reply
    # before we ask the server to shut the session down.
    await asyncio.sleep(1.0)


async def receive_audio_stream(websocket) -> None:
    """Receive either chunked raw PCM replies or single WAV fallback replies."""
    reply_chunks: dict[str, list[tuple[int, bytes]]] = {}
    completed_replies: set[str] = set()  # Track which replies we've already saved

    try:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=30)

            if isinstance(message, bytes):
                print("Received unexpected binary message:", len(message))
                continue

            payload = json.loads(message)
            if payload.get("type") != "streamAudio":
                print("Received non-audio message:", payload)
                continue

            data = payload["data"]
            reply_id = data.get("replyId", "single")
            chunk_index = data.get("chunkIndex", 0)
            is_final = data.get("isFinalChunk", False)
            audio_data_type = data.get("audioDataType")
            encoding = data.get("encoding")
            sample_rate = data.get("sampleRate")
            audio_b64 = data.get("audioData", "")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

            if audio_bytes:
                reply_chunks.setdefault(reply_id, []).append((chunk_index, audio_bytes))

            print(
                "Received streamAudio:",
                f"reply_id={reply_id}",
                f"chunk_index={chunk_index}",
                f"is_final={is_final}",
                f"audio_type={audio_data_type}",
                f"encoding={encoding}",
                f"sample_rate={sample_rate}",
                f"bytes={len(audio_bytes)}",
            )

            if audio_data_type == "wav" and audio_bytes:
                output_path = REPLY_OUTPUT_DIR / f"reply_{reply_id}.wav"
                REPLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_bytes)
                print(f"Saved WAV fallback reply: {output_path}")
                break

            if is_final:
                # Skip empty final chunks - wait for real audio data
                chunks = reply_chunks.get(reply_id, [])
                if not chunks:
                    print(f"⏩ Skipping empty final chunk for {reply_id}, waiting for more data...")
                    continue
                
                if reply_id in completed_replies:
                    print(f"↩️ Already completed reply {reply_id}, skipping duplicate final chunk")
                    continue
                
                chunks.sort(key=lambda item: item[0])
                combined_audio = b"".join(chunk for _, chunk in chunks)
                print(f"Reply {reply_id} complete: {len(chunks)} chunks, {len(combined_audio)} bytes")
                save_reply_wav(reply_id, combined_audio, sample_rate)
                completed_replies.add(reply_id)
                
                # Wait a bit more for any additional replies, then exit
                print("Waiting 2 seconds for any additional replies...")
                try:
                    await asyncio.wait_for(asyncio.sleep(2), timeout=2)
                except asyncio.TimeoutError:
                    pass
                break

    except asyncio.TimeoutError:
        print("Timed out waiting for streamed audio reply")
    except ConnectionClosedOK:
        print("WebSocket closed cleanly by server")
    except ConnectionClosed as exc:
        print(f"WebSocket closed while waiting for reply: code={exc.code} reason={exc.reason}")


def save_reply_wav(reply_id: str, pcm_data: bytes, sample_rate: int) -> Path:
    """Persist the streamed raw PCM reply as a WAV file for listening."""
    REPLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPLY_OUTPUT_DIR / f"reply_{reply_id}.wav"

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    print(f"Saved reply WAV: {output_path}")
    return output_path


async def send_audio():
    pcm_data, source_rate = load_pcm_audio(AUDIO_FILE)
    pcm_8k = resample_pcm(pcm_data, from_rate=source_rate, to_rate=TARGET_SAMPLE_RATE)

    async with websockets.connect(WS_URL, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "type": "config",
            "responseAudioMode": RESPONSE_AUDIO_MODE,
        }))
        print(f"Requested response audio mode: {RESPONSE_AUDIO_MODE}")
        sender_task = asyncio.create_task(send_audio_realtime(websocket, pcm_8k, TARGET_SAMPLE_RATE))
        receiver_task = asyncio.create_task(receive_audio_stream(websocket))

        await sender_task
        print("Audio sent, waiting for TTS response...")
        
        try:
            # Wait up to 15 seconds for the receiver to finish processing
            await asyncio.wait_for(receiver_task, timeout=15.0)
            print("Received all audio replies successfully")
        except asyncio.TimeoutError:
            print("⚠️ Receiver timeout - TTS may still be processing")
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

        # Give the server a moment to finish any cleanup
        await asyncio.sleep(0.5)
        
        try:
            await websocket.send("stop")
            print("Sent stop")
        except ConnectionClosedOK:
            print("Server had already closed the WebSocket cleanly")
        except ConnectionClosed as exc:
            print(f"WebSocket closed before stop could be sent: code={exc.code} reason={exc.reason}")


if __name__ == "__main__":
    asyncio.run(send_audio())
 
