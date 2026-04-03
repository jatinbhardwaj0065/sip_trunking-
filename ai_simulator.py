from pathlib import Path
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import traceback
import wave
import base64
import json
import asyncio
import uuid
import numpy as np

from app.services.stt_service import transcribe_audio_streaming_continuous
from app.services.llm_service import process_text_streaming
from app.services.tts_service import generate_speech_stream_chunked
from app.services.language_utils import detect_language_from_script

DEFAULT_RESPONSE_AUDIO_MODE = "stream"


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

def resample_pcm(
    pcm_data: bytes,
    from_rate: int,
    to_rate: int,
) -> bytes:
    """
    Resample mono 16-bit PCM for the 8k<->16k paths used by this bridge.
    Handles odd-byte chunks by padding.
    """
    if from_rate == to_rate or not pcm_data:
        return pcm_data

    # Ensure even number of bytes for 16-bit PCM
    if len(pcm_data) % 2 != 0:
        # Pad with a zero byte to make it even
        pcm_data = pcm_data + b"\x00"

    samples = np.frombuffer(pcm_data, dtype=np.int16)
    if samples.size == 0:
        return b""

    if from_rate == 8000 and to_rate == 16000:
        return np.repeat(samples, 2).astype(np.int16).tobytes()

    if from_rate == 16000 and to_rate == 8000:
        return samples[::2].astype(np.int16).tobytes()

    raise ValueError(f"Unsupported resample path: {from_rate} -> {to_rate}")


class PCMStreamResampler:
    """Stateful mono PCM resampler for chunked 8k<->16k streaming."""

    def __init__(self, from_rate: int, to_rate: int):
        self.from_rate = from_rate
        self.to_rate = to_rate
        self.pending = b""
        self.phase = 0

    def process(self, pcm_data: bytes) -> bytes:
        if not pcm_data:
            return b""

        pcm_data = self.pending + pcm_data
        if len(pcm_data) % 2 != 0:
            self.pending = pcm_data[-1:]
            pcm_data = pcm_data[:-1]
        else:
            self.pending = b""

        if not pcm_data:
            return b""

        samples = np.frombuffer(pcm_data, dtype=np.int16)
        if samples.size == 0:
            return b""

        if self.from_rate == 16000 and self.to_rate == 8000:
            out = samples[self.phase::2]
            self.phase = (self.phase + samples.size) % 2
            return out.astype(np.int16).tobytes()

        if self.from_rate == 8000 and self.to_rate == 16000:
            return np.repeat(samples, 2).astype(np.int16).tobytes()

        raise ValueError(f"Unsupported stream resample path: {self.from_rate} -> {self.to_rate}")

    def flush(self) -> bytes:
        if not self.pending:
            return b""

        pending = self.pending + b"\x00"
        self.pending = b""
        return self.process(pending)


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap mono 16-bit PCM in a WAV container."""
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def pcm_rms(pcm_data: bytes) -> float:
    """Compute RMS for mono 16-bit PCM."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))

async def stream_tts_reply(
    websocket: WebSocket,
    text: str,
    language_code: str,
    output_sample_rate: int = 8000,
    cancel_event: asyncio.Event = None,
) -> None:
    """
    Stream raw PCM chunks over the websocket as TTS audio arrives.

    Supports cancellation for barging - if cancel_event is set, stops streaming immediately.

    WAV is not a streaming container, so outbound streaming uses raw PCM chunks
    with sequence metadata and an explicit end marker.
    """
    reply_id = str(uuid.uuid4())
    chunk_index = 0
    resampler = PCMStreamResampler(from_rate=16000, to_rate=output_sample_rate)
    total_tts_bytes = 0
    total_pcm_bytes = 0

    print(f"🎬 Starting WebSocket audio stream to client (reply={reply_id})")

    try:
        async for chunk in generate_speech_stream_chunked(text, language_code=language_code):
            # Check for cancellation (barging)
            if cancel_event and cancel_event.is_set():
                print(f"🛑 TTS cancellation requested - stopping stream")
                break

            if not chunk:
                continue

            total_tts_bytes += len(chunk)
            pcm_8k = resampler.process(chunk)

            if not pcm_8k:
                continue

            total_pcm_bytes += len(pcm_8k)
            payload = {
                "type": "streamAudio",
                "data": {
                    "audioDataType": "raw",
                    "encoding": "pcm_s16le",
                    "sampleRate": output_sample_rate,
                    "replyId": reply_id,
                    "chunkIndex": chunk_index,
                    "isFinalChunk": False,
                    "audioData": base64.b64encode(pcm_8k).decode("utf-8"),
                },
            }
            await websocket.send_text(json.dumps(payload))
            print(f"📤 WebSocket chunk {chunk_index}: {len(chunk)} bytes TTS → {len(pcm_8k)} bytes PCM")
            chunk_index += 1

        # Check before sending final chunks
        if cancel_event and cancel_event.is_set():
            print(f"🛑 Skipping final chunks due to barging")
            return

        final_pcm = resampler.flush()
        if final_pcm:
            total_pcm_bytes += len(final_pcm)
            payload = {
                "type": "streamAudio",
                "data": {
                    "audioDataType": "raw",
                    "encoding": "pcm_s16le",
                    "sampleRate": output_sample_rate,
                    "replyId": reply_id,
                    "chunkIndex": chunk_index,
                    "isFinalChunk": False,
                    "audioData": base64.b64encode(final_pcm).decode("utf-8"),
                },
            }
            await websocket.send_text(json.dumps(payload))
            print(f"📤 WebSocket flush chunk {chunk_index}: {len(final_pcm)} bytes PCM")
            chunk_index += 1

        end_payload = {
            "type": "streamAudio",
            "data": {
                "audioDataType": "raw",
                "encoding": "pcm_s16le",
                "sampleRate": output_sample_rate,
                "replyId": reply_id,
                "chunkIndex": chunk_index,
                "isFinalChunk": True,
                "audioData": "",
            },
        }
        await websocket.send_text(json.dumps(end_payload))
        print(f"✅ Finished streaming AI reply {reply_id} in {chunk_index} chunks")
        print(f"   TTS→PCM: {total_tts_bytes} bytes → {total_pcm_bytes} bytes")
        print(f"   Stream rate: {output_sample_rate}Hz, {total_pcm_bytes/(output_sample_rate*2):.2f}s of audio")

    except Exception as e:
        print(f"❌ Error in TTS streaming: {e}")


async def send_wav_tts_reply(
    websocket: WebSocket,
    text: str,
    language_code: str,
    output_sample_rate: int = 8000,
) -> None:
    """Collect TTS PCM, resample, wrap as WAV, and send as one message."""
    pcm_parts = []
    total_tts_bytes = 0

    async for chunk in generate_speech_stream_chunked(text, language_code=language_code):
        if not chunk:
            continue
        total_tts_bytes += len(chunk)
        pcm_parts.append(resample_pcm(chunk, from_rate=16000, to_rate=output_sample_rate))

    pcm_8k = b"".join(pcm_parts)
    wav_bytes = pcm_to_wav_bytes(pcm_8k, sample_rate=output_sample_rate)
    payload = {
        "type": "streamAudio",
        "data": {
            "audioDataType": "wav",
            "sampleRate": output_sample_rate,
            "isFinalChunk": True,
            "audioData": base64.b64encode(wav_bytes).decode("utf-8"),
        },
    }
    await websocket.send_text(json.dumps(payload))
    print(f"✅ Sent WAV AI reply in one message: {total_tts_bytes} bytes TTS → {len(pcm_8k)} bytes PCM → {len(wav_bytes)} bytes WAV")
    print(f"   Audio duration: {len(pcm_8k)/(output_sample_rate*2):.2f}s at {output_sample_rate}Hz")


class DuplexVoiceBridge:
    """Always-on call bridge: continuous inbound STT plus concurrent reply generation with barging support."""

    def __init__(self, websocket: WebSocket, input_sample_rate: int = 8000):
        self.websocket = websocket
        self.input_sample_rate = input_sample_rate
        self.stt_sample_rate = 16000
        self.sample_width = 2
        self.response_audio_mode = DEFAULT_RESPONSE_AUDIO_MODE

        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.full_audio_buffer = bytearray()

        self.reply_lock = asyncio.Lock()
        self.running = True
        self.background_tasks: set[asyncio.Task] = set()

        # Barging support
        self.reply_cancel_event = asyncio.Event()  # Signal to cancel ongoing reply
        self.user_speaking = False  # Track if user is actively speaking
        self.speech_threshold = 0.03  # RMS threshold for speech detection

    def start(self) -> None:
        self._track(asyncio.create_task(self._run_stt_receiver()))
        self._track(asyncio.create_task(self._run_reply_worker()))

    def _track(self, task: asyncio.Task) -> None:
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def ingest_audio_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return

        self.full_audio_buffer.extend(chunk)

        # Detect speech energy for barging
        try:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if samples.size > 0:
                rms = np.sqrt(np.mean(samples * samples)) / 32768.0
                was_speaking = self.user_speaking
                self.user_speaking = rms > self.speech_threshold

                # Trigger barging if user starts speaking during AI response
                if self.user_speaking and not was_speaking and self.reply_lock.locked():
                    print(f"🎤 BARGE DETECTED! (speech energy: {rms:.4f})")
                    self.reply_cancel_event.set()
        except Exception as e:
            pass  # Ignore speech detection errors

        pcm_16k = resample_pcm(chunk, from_rate=self.input_sample_rate, to_rate=self.stt_sample_rate)
        await self.audio_queue.put(pcm_16k)

    async def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        await self.audio_queue.put(None)
        await self.transcript_queue.put(None)

        tasks = list(self.background_tasks)
        for task in tasks:
            try:
                await task
            except Exception as exc:
                print(f"Background task failed: {exc!r}")

    async def _run_stt_receiver(self) -> None:
        try:
            async for transcript_data in transcribe_audio_streaming_continuous(
                self.audio_queue,
                sample_rate=self.stt_sample_rate,
                process_interim=True,  # Enable interim results for low-latency
            ):
                # STT now returns (text, lang, is_final)
                transcript_text, detected_lang, is_final = transcript_data

                if not transcript_text:
                    continue

                if is_final:
                    print(f"📝 FINAL Transcript: {transcript_text!r} (lang={detected_lang})")
                    await self.transcript_queue.put((transcript_text, detected_lang, is_final))
                else:
                    # Interim result - process immediately for low-latency LLM
                    print(f"🎤 INTERIM: {transcript_text!r} (lang={detected_lang})")
                    await self.transcript_queue.put((transcript_text, detected_lang, is_final))
        except Exception as exc:
            print(f"Continuous STT receiver failed: {exc!r}")
        finally:
            await self.transcript_queue.put(None)

    async def _run_reply_worker(self) -> None:
        current_llm_task = None
        current_tts_task = None

        while True:
            item = await self.transcript_queue.get()
            if item is None:
                break

            transcript_text, detected_lang, is_final = item

            # Skip interim results - only process final
            if not is_final:
                print(f"   (queued interim result, waiting for final...)")
                continue

            # Cancel any ongoing reply before starting new one
            if current_tts_task and not current_tts_task.done():
                print(f"🛑 Cancelling previous TTS task for barging...")
                self.reply_cancel_event.set()
                try:
                    await asyncio.wait_for(current_tts_task, timeout=1.0)
                except asyncio.TimeoutError:
                    current_tts_task.cancel()

            try:
                # Reset cancel event for new reply
                self.reply_cancel_event.clear()

                # Stream LLM response and collect final text
                full_response = ""
                final_response = None

                async for token in process_text_streaming(transcript_text, language_code=detected_lang):
                    if "__TRANSLATED__" in token:
                        # Extract translated response
                        parts = token.split("__TRANSLATED__")
                        if len(parts) > 1:
                            final_response = parts[1].split("__END_TRANSLATED__")[0]
                    else:
                        # Accumulate tokens
                        full_response += token

                # Use translated response if available, otherwise use accumulated tokens
                response_text = final_response if final_response else full_response

                if not response_text:
                    continue

                # Auto-detect the actual language of the response
                actual_lang = detect_language_from_script(response_text)
                print(f"🤖 AI response: {response_text!r}")
                print(f"   Response language detected as: {actual_lang} (input was: {detected_lang})")

                async with self.reply_lock:
                    try:
                        if self.response_audio_mode == "wav":
                            current_tts_task = asyncio.create_task(send_wav_tts_reply(
                                self.websocket,
                                response_text,
                                language_code=actual_lang,
                                output_sample_rate=self.input_sample_rate,
                            ))
                        else:
                            current_tts_task = asyncio.create_task(stream_tts_reply(
                                self.websocket,
                                response_text,
                                language_code=actual_lang,
                                output_sample_rate=self.input_sample_rate,
                                cancel_event=self.reply_cancel_event,
                            ))

                        # Wait for TTS to complete or be cancelled
                        await current_tts_task

                    except asyncio.CancelledError:
                        print(f"🛑 TTS cancelled (barging)")
                        break
                    except Exception as exc:
                        print(f"Reply send failed: {exc!r}")
                        break

            except Exception as exc:
                print(f"LLM failed: {exc!r}")
                continue


async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")

    bridge = DuplexVoiceBridge(websocket, input_sample_rate=8000)
    bridge.start()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                print("WebSocket disconnect frame received")
                break

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                print(f"Received chunk: {len(chunk)} bytes, total={len(bridge.full_audio_buffer) + len(chunk)}")
                await bridge.ingest_audio_chunk(chunk)

            elif message.get("text") is not None:
                text_message = message["text"]
                print("Received text:", text_message)

                try:
                    payload = json.loads(text_message)
                except json.JSONDecodeError:
                    payload = None

                if isinstance(payload, dict) and payload.get("type") == "config":
                    requested_mode = payload.get("responseAudioMode")
                    if requested_mode in {"stream", "wav"}:
                        bridge.response_audio_mode = requested_mode
                        print(f"Response audio mode set to: {requested_mode}")
                    continue

                if text_message.strip().lower() == "stop":
                    break

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", repr(e))
        traceback.print_exc()

    finally:
        await bridge.stop()

        print("Final total bytes:", len(bridge.full_audio_buffer))
        if bridge.full_audio_buffer:
            try:
                path = save_wav_file(bytes(bridge.full_audio_buffer), sample_rate=8000)
                print("Saved WAV:", path)
            except Exception as e:
                print("Save error:", repr(e))
                traceback.print_exc()

        try:
            await websocket.close()
        except Exception:
            pass
 