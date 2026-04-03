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

DEFAULT_RESPONSE_AUDIO_MODE = "wav"
FIXED_FREESWITCH_REPLY_PATH = Path("/tmp/python_reply.wav")


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


def persist_reply_wav(wav_bytes: bytes, target_path: Path = FIXED_FREESWITCH_REPLY_PATH) -> Path:
    """Save the latest synthesized reply to a predictable path for FreeSWITCH playback."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(wav_bytes)
    print(f"Saved FreeSWITCH reply WAV: {target_path}")
    return target_path


def clear_fixed_reply_wav(target_path: Path = FIXED_FREESWITCH_REPLY_PATH) -> None:
    """Remove any stale reply file so FreeSWITCH cannot replay old audio by mistake."""
    try:
        target_path.unlink()
        print(f"Removed stale FreeSWITCH reply WAV: {target_path}")
    except FileNotFoundError:
        print(f"No stale FreeSWITCH reply WAV at call start: {target_path}")


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
            try:
                await websocket.send_text(json.dumps(payload))
                print(f"📤 WebSocket chunk {chunk_index}: {len(chunk)} bytes TTS → {len(pcm_8k)} bytes PCM")
            except Exception as e:
                print(f"❌ WebSocket send error: {e} - client likely disconnected")
                return
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
            try:
                await websocket.send_text(json.dumps(payload))
                print(f"📤 WebSocket flush chunk {chunk_index}: {len(final_pcm)} bytes PCM")
            except Exception as e:
                print(f"❌ WebSocket send error during flush: {e}")
                return
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
        try:
            await websocket.send_text(json.dumps(end_payload))
        except Exception as e:
            print(f"❌ WebSocket send error during end marker: {e}")
            return
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
    """Collect TTS PCM, resample, wrap as WAV, and save it for FreeSWITCH playback."""
    pcm_parts = []
    total_tts_bytes = 0

    async for chunk in generate_speech_stream_chunked(text, language_code=language_code):
        if not chunk:
            continue
        total_tts_bytes += len(chunk)
        pcm_parts.append(resample_pcm(chunk, from_rate=16000, to_rate=output_sample_rate))

    pcm_8k = b"".join(pcm_parts)
    if not pcm_8k:
        print("No TTS audio generated, skipping reply WAV save.")
        return

    wav_bytes = pcm_to_wav_bytes(pcm_8k, sample_rate=output_sample_rate)
    persist_reply_wav(wav_bytes)
    print(f"✅ Reply ready: {FIXED_FREESWITCH_REPLY_PATH}")
    print(f"   TTS→PCM→WAV: {total_tts_bytes} bytes → {len(pcm_8k)} bytes → {len(wav_bytes)} bytes")
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
        self.generated_reply_path: Path | None = None

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
        print("Closing inbound audio stream and draining STT/LLM/TTS pipeline...")
        await self.audio_queue.put(None)

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
        current_tts_task = None
        debounce_timer = None
        pending_transcript = None
        pending_lang = None

        async def process_transcript_with_streaming_tts(text: str, lang: str) -> None:
            """Process transcript: stream LLM tokens continuously, then TTS on complete response."""
            nonlocal current_tts_task
            
            try:
                # Reset cancel event for new reply
                self.reply_cancel_event.clear()

                full_response = ""
                final_response = None
                token_count = 0

                async with self.reply_lock:
                    try:
                        # Stream LLM tokens continuously
                        print(f"🚀 Starting LLM stream processing...")
                        async for token in process_text_streaming(text, language_code=lang):
                            if "__TRANSLATED__" in token:
                                # Extract translated response
                                parts = token.split("__TRANSLATED__")
                                if len(parts) > 1:
                                    final_response = parts[1].split("__END_TRANSLATED__")[0]
                            else:
                                # Accumulate tokens
                                full_response += token
                                token_count += 1
                                print(f"🤖 Token {token_count}: '{token}'", end="", flush=True)
                        
                        print()  # Newline after token stream
                        
                        # Use complete response for TTS (with translation if available)
                        response_text = final_response if final_response else full_response
                        
                        if response_text:
                            detected_lang = detect_language_from_script(response_text)
                            print(f"📝 Complete response ready for TTS ({token_count} tokens): '{response_text}' ({detected_lang})\n")

                            if self.response_audio_mode == "wav":
                                current_tts_task = asyncio.create_task(
                                    send_wav_tts_reply(
                                        self.websocket,
                                        response_text,
                                        language_code=detected_lang,
                                        output_sample_rate=self.input_sample_rate,
                                    )
                                )
                            else:
                                current_tts_task = asyncio.create_task(
                                    stream_tts_reply(
                                        self.websocket,
                                        response_text,
                                        language_code=detected_lang,
                                        output_sample_rate=self.input_sample_rate,
                                        cancel_event=self.reply_cancel_event,
                                    )
                                )
                            await current_tts_task
                            if self.response_audio_mode == "wav" and FIXED_FREESWITCH_REPLY_PATH.exists():
                                self.generated_reply_path = FIXED_FREESWITCH_REPLY_PATH
                                print(f"✅ Fresh FreeSWITCH reply file ready: {self.generated_reply_path}")

                    except asyncio.CancelledError:
                        print(f"🛑 TTS cancelled (barging)")
                    except Exception as exc:
                        print(f"Reply send failed: {exc!r}")

            except Exception as exc:
                print(f"LLM streaming failed: {exc!r}")

        while True:
            try:
                # Use a short timeout to implement debouncing
                item = await asyncio.wait_for(self.transcript_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                # Debounce timer expired - process pending transcript if it exists
                if pending_transcript and (debounce_timer is None or asyncio.get_event_loop().time() >= debounce_timer):
                    text_to_process = pending_transcript
                    lang_to_process = pending_lang
                    pending_transcript = None
                    pending_lang = None
                    
                    # Cancel ongoing reply if new transcript arrived
                    if current_tts_task and not current_tts_task.done():
                        print(f"⏱️ Debounce expired, processing buffered interim result...")
                        self.reply_cancel_event.set()
                        try:
                            await asyncio.wait_for(current_tts_task, timeout=0.5)
                        except asyncio.TimeoutError:
                            current_tts_task.cancel()
                    
                    await process_transcript_with_streaming_tts(text_to_process, lang_to_process)
                continue

            if item is None:
                break

            transcript_text, detected_lang, is_final = item

            # For interim results: buffer and debounce to avoid too many LLM calls
            if not is_final:
                print(f"🎤 INTERIM (buffering): {transcript_text!r}")
                pending_transcript = transcript_text
                pending_lang = detected_lang
                debounce_timer = asyncio.get_event_loop().time() + 0.2  # 200ms debounce
                continue

            # Final result arrived - process it immediately
            print(f"📝 FINAL Transcript: {transcript_text!r}")
            pending_transcript = None  # Clear any pending interim
            
            # Cancel ongoing reply for final result
            if current_tts_task and not current_tts_task.done():
                print(f"🛑 Cancelling previous response for final transcript...")
                self.reply_cancel_event.set()
                try:
                    await asyncio.wait_for(current_tts_task, timeout=0.5)
                except asyncio.TimeoutError:
                    current_tts_task.cancel()
            
            await process_transcript_with_streaming_tts(transcript_text, detected_lang)


async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")
    clear_fixed_reply_wav()

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
                    print("Stop signal received - waiting for any ongoing TTS to finish...")
                    # Give the reply worker a moment to complete current TTS streaming
                    await asyncio.sleep(0.5)
                    break

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", repr(e))
        traceback.print_exc()

    finally:
        await bridge.stop()
        
        # Wait a moment for any remaining I/O to complete
        await asyncio.sleep(0.2)

        print("Final total bytes:", len(bridge.full_audio_buffer))
        if bridge.full_audio_buffer:
            try:
                path = save_wav_file(bytes(bridge.full_audio_buffer), sample_rate=8000)
                print("Saved WAV:", path)
            except Exception as e:
                print("Save error:", repr(e))
                traceback.print_exc()

        if bridge.generated_reply_path and bridge.generated_reply_path.exists():
            print(f"Reply generation completed for this call: {bridge.generated_reply_path}")
        else:
            print("No fresh reply WAV was generated for this call.")

        try:
            await websocket.close()
        except Exception:
            pass
 
