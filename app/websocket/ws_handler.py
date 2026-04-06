from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from fastapi import WebSocket, WebSocketDisconnect
import traceback
import wave
import base64
import json
import asyncio
import uuid
from time import perf_counter
import numpy as np

from app.services.stt_service import transcribe_audio_streaming_continuous
from app.services.llm_service import process_text_streaming, warmup_llm_connection
from app.services.tts_service import generate_speech_stream_chunked, warmup_tts_connection

DEFAULT_RESPONSE_AUDIO_MODE = "stream"
FIXED_FREESWITCH_REPLY_PATH = Path("/tmp/python_reply.wav")
INTERIM_DEBOUNCE_SECONDS = 0.03  # ULTRA-OPTIMIZED: 30ms for maximum dispatch speed
TRANSCRIPT_POLL_SECONDS = 0.02  # Poll every 20ms - even tighter loop
LOCAL_ENDPOINT_SILENCE_SECONDS = 0.08  # ULTRA-OPTIMIZED: 80ms silence (was 100ms)
MIN_INTERIM_DISPATCH_CHARS = 8  # Dispatch at 8+ chars for ultra-fast LLM dispatch
MIN_INTERIM_DISPATCH_WORDS = 1  # Even 1 word triggers dispatch
RECENT_REPLY_WINDOW_SECONDS = 8.0
SIMILAR_TRANSCRIPT_MAX_EXTRA_WORDS = 2
SIMILAR_TRANSCRIPT_MAX_EXTRA_CHARS = 18


def wall_clock_now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.0f}ms"


def summarize_text(text: str, limit: int = 60) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def normalize_transcript(text: str) -> str:
    return " ".join(text.split()).strip()


def transcript_word_count(text: str) -> int:
    normalized = normalize_transcript(text)
    return len(normalized.split()) if normalized else 0


def is_similar_transcript(previous: str, current: str) -> bool:
    previous_normalized = normalize_transcript(previous)
    current_normalized = normalize_transcript(current)

    if not previous_normalized or not current_normalized:
        return False

    if previous_normalized == current_normalized:
        return True

    if current_normalized.startswith(previous_normalized):
        delta = current_normalized[len(previous_normalized):].strip()
    elif previous_normalized.startswith(current_normalized):
        delta = previous_normalized[len(current_normalized):].strip()
    else:
        return False

    return (
        len(delta) <= SIMILAR_TRANSCRIPT_MAX_EXTRA_CHARS
        or len(delta.split()) <= SIMILAR_TRANSCRIPT_MAX_EXTRA_WORDS
    )


@dataclass
class ReplyTiming:
    reply_number: int
    source: str
    transcript: str
    language_code: str
    call_started_at: float
    utterance_started_at: float | None
    last_voice_activity_at: float | None
    dispatch_started_at: float
    llm_first_token_at: float | None = None
    llm_completed_at: float | None = None
    tts_started_at: float | None = None
    first_audio_sent_at: float | None = None
    completed_at: float | None = None

    def log(self, stage: str, event_time: float, extra: str = "") -> None:
        parts = [
            f"[{wall_clock_now()}] ⏱️ reply#{self.reply_number}",
            stage,
            f"source={self.source}",
            f"call+{format_ms(event_time - self.call_started_at)}",
            f"dispatch+{format_ms(event_time - self.dispatch_started_at)}",
        ]

        if self.utterance_started_at is not None:
            parts.append(f"utterance+{format_ms(event_time - self.utterance_started_at)}")

        if self.last_voice_activity_at is not None:
            parts.append(f"after_speech+{format_ms(event_time - self.last_voice_activity_at)}")

        if extra:
            parts.append(extra)

        print(" | ".join(parts))

    def log_dispatch(self) -> None:
        self.log(
            "dispatch",
            self.dispatch_started_at,
            extra=f"lang={self.language_code} text='{summarize_text(self.transcript)}'",
        )

    def log_summary(self, audio_seconds: float | None = None) -> None:
        if self.completed_at is None:
            return

        extra_parts = []
        if self.llm_first_token_at is not None:
            extra_parts.append(f"llm_first={format_ms(self.llm_first_token_at - self.dispatch_started_at)}")
        if self.tts_started_at is not None:
            extra_parts.append(f"tts_start={format_ms(self.tts_started_at - self.dispatch_started_at)}")
        if self.first_audio_sent_at is not None:
            extra_parts.append(f"first_audio={format_ms(self.first_audio_sent_at - self.dispatch_started_at)}")
        if audio_seconds is not None:
            extra_parts.append(f"audio={audio_seconds:.2f}s")

        self.log("complete", self.completed_at, extra=" ".join(extra_parts))


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


async def stream_tts_reply_with_streaming_text(
    websocket: WebSocket,
    token_generator,
    language_code: str,
    output_sample_rate: int = 8000,
    cancel_event: asyncio.Event = None,
    timing: ReplyTiming = None,
) -> None:
    """
    🚀 OPTIMIZED: Stream TTS while LLM is still generating tokens.
    
    Instead of waiting for full response, buffers tokens until a sentence boundary
    (., !, ?, or enough content), then synthesizes TTS immediately while continuing
    to accumulate more tokens.
    
    This eliminates the 200-500ms wait for full LLM response completion!
    
    Args:
        token_generator: Async generator yielding individual tokens from LLM
        language_code: Language for TTS synthesis
        output_sample_rate: 8000 for FreeSWITCH, 16000 for modern clients
        cancel_event: Set to cancel streaming (barging)
        timing: ReplyTiming object for logging
    """
    reply_id = str(uuid.uuid4())
    chunk_index = 0
    resampler = PCMStreamResampler(from_rate=16000, to_rate=output_sample_rate)
    total_tts_bytes = 0
    total_pcm_bytes = 0
    
    # Token accumulation for sentence-by-sentence TTS
    token_buffer = ""
    tts_queue = asyncio.Queue()  # Queue of text chunks to synthesize
    pending_tts_task = None
    
    print(f"🎬 Starting optimized streaming TTS (reply={reply_id})")
    if timing:
        timing.tts_started_at = perf_counter()
        timing.log("tts_start", timing.tts_started_at, extra=f"reply_id={reply_id} mode=streaming_tokens")
    
    def is_sentence_boundary(text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        return any(text.rstrip().endswith(p) for p in [".", "!", "?", "。", "।", "।", "؟"])
    
    def should_flush_buffer(buffer: str, incoming_token: str = "") -> bool:
        """Decide if we should send buffered text to TTS now."""
        candidate = buffer + incoming_token
        # ULTRA-OPTIMIZED: Send at 12+ chars - start TTS synthesis ASAP with minimal latency
        # Even short phrases like "I'm here" or "One sec" will start synthesizing immediately
        return is_sentence_boundary(buffer) or len(buffer.strip()) >= 12
    
    async def tts_synthesizer():
        """Worker task: consume text chunks and generate TTS audio."""
        nonlocal total_tts_bytes, total_pcm_bytes, chunk_index
        
        while True:
            try:
                text_chunk = await asyncio.wait_for(tts_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # TTS queue is empty, LLM finished
                break
            
            if text_chunk is None:  # Sentinel to stop synthesizer
                break
            
            if not text_chunk.strip():
                continue
            
            # Generate TTS for this text chunk
            try:
                async for tts_chunk in generate_speech_stream_chunked(
                    text_chunk, 
                    language_code=language_code
                ):
                    if cancel_event and cancel_event.is_set():
                        print(f"🛑 TTS cancelled mid-synthesis")
                        return
                    
                    if not tts_chunk:
                        continue
                    
                    total_tts_bytes += len(tts_chunk)
                    pcm_8k = resampler.process(tts_chunk)
                    
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
                        if timing and timing.first_audio_sent_at is None:
                            timing.first_audio_sent_at = perf_counter()
                            timing.log(
                                "first_audio_sent",
                                timing.first_audio_sent_at,
                                extra=f"reply_id={reply_id} early_via_streaming"
                            )
                    except Exception as e:
                        print(f"❌ WebSocket send error: {e}")
                        return
                    
                    chunk_index += 1
            except Exception as e:
                print(f"⚠️ TTS synthesis error: {e}")
                continue
    
    try:
        # Start TTS synthesizer in background
        tts_task = asyncio.create_task(tts_synthesizer())
        
        # Process tokens from LLM
        try:
            async for token in token_generator:
                if cancel_event and cancel_event.is_set():
                    print("🛑 LLM token streaming cancelled (barging)")
                    break
                
                token_buffer += token
                
                # When we have enough for TTS, send to queue
                if should_flush_buffer(token_buffer):
                    text_to_synthesize = token_buffer.rstrip()
                    print(f"📢 Queuing TTS: '{text_to_synthesize[:40]}...'")
                    await tts_queue.put(text_to_synthesize)
                    token_buffer = ""
        
        except Exception as e:
            print(f"⚠️ Error processing LLM tokens: {e}")
        
        # Flush any remaining buffered text
        if token_buffer.strip():
            print(f"📢 Queuing final TTS chunk: '{token_buffer[:40]}...'")
            await tts_queue.put(token_buffer)
        
        # Signal synthesizer to finish
        await tts_queue.put(None)
        
        # Wait for TTS to complete
        await tts_task
        
        # Send final flush and end marker ONLY if audio was generated
        if chunk_index > 0:
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
                except Exception:
                    pass
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
            except Exception:
                pass
        else:
            print(f"⏭️ Skipping empty final chunk (no audio synthesized for {reply_id})")
        
        print(f"✅ Finished optimized streaming reply {reply_id} in {chunk_index} chunks")
        print(f"   TTS→PCM: {total_tts_bytes} bytes → {total_pcm_bytes} bytes")
        if total_pcm_bytes > 0:
            print(f"   Audio duration: {total_pcm_bytes/(output_sample_rate*2):.2f}s")
        
        if timing:
            timing.completed_at = perf_counter()
            timing.log_summary(audio_seconds=total_pcm_bytes / (output_sample_rate * 2) if total_pcm_bytes else None)
    
    except Exception as e:
        print(f"❌ Error in optimized TTS streaming: {e}")


async def stream_tts_reply(
    websocket: WebSocket,
    text: str,
    language_code: str,
    output_sample_rate: int = 8000,
    cancel_event: asyncio.Event = None,
    timing: ReplyTiming | None = None,
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
    if timing:
        timing.tts_started_at = perf_counter()
        timing.log("tts_start", timing.tts_started_at, extra=f"reply_id={reply_id}")

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
                if timing and timing.first_audio_sent_at is None:
                    timing.first_audio_sent_at = perf_counter()
                    timing.log("first_audio_sent", timing.first_audio_sent_at, extra=f"reply_id={reply_id}")
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

        # Only send final chunk marker if audio was actually generated
        if chunk_index > 0:
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
        else:
            print(f"⏭️ Skipping empty final chunk (no audio synthesized for {reply_id})")
            
        print(f"✅ Finished streaming AI reply {reply_id} in {chunk_index} chunks")
        print(f"   TTS→PCM: {total_tts_bytes} bytes → {total_pcm_bytes} bytes")
        if total_pcm_bytes > 0:
            print(f"   Stream rate: {output_sample_rate}Hz, {total_pcm_bytes/(output_sample_rate*2):.2f}s of audio")
        if timing:
            timing.completed_at = perf_counter()
            timing.log_summary(audio_seconds=total_pcm_bytes / (output_sample_rate * 2) if total_pcm_bytes else None)

    except Exception as e:
        print(f"❌ Error in TTS streaming: {e}")


async def send_wav_tts_reply(
    websocket: WebSocket,
    text: str,
    language_code: str,
    output_sample_rate: int = 8000,
    timing: ReplyTiming | None = None,
) -> None:
    """Collect TTS PCM, resample, wrap as WAV, and save it for FreeSWITCH playback."""
    pcm_parts = []
    total_tts_bytes = 0

    if timing:
        timing.tts_started_at = perf_counter()
        timing.log("tts_start", timing.tts_started_at, extra="mode=wav")

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
    if timing:
        now = perf_counter()
        if timing.first_audio_sent_at is None:
            timing.first_audio_sent_at = now
        timing.completed_at = now
        timing.log_summary(audio_seconds=len(pcm_8k) / (output_sample_rate * 2))


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
        self.last_dispatched_transcript = ""
        self.last_final_transcript = ""
        self.last_reply_transcript = ""
        self.last_reply_source = ""
        self.last_reply_completed_at_perf: float | None = None
        self.stop_requested = False
        self.last_voice_activity_at = asyncio.get_event_loop().time()
        self.call_started_at = perf_counter()
        self.last_voice_activity_at_perf = self.call_started_at
        self.current_utterance_started_at_perf: float | None = None
        self.reply_counter = 0

    def start(self) -> None:
        self._track(asyncio.create_task(warmup_llm_connection()))  # Prime LLM API connection
        self._track(asyncio.create_task(warmup_tts_connection()))
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
                now_perf = perf_counter()
                rms = np.sqrt(np.mean(samples * samples)) / 32768.0
                was_speaking = self.user_speaking
                self.user_speaking = rms > self.speech_threshold

                if self.user_speaking and not was_speaking:
                    self.current_utterance_started_at_perf = now_perf

                if self.user_speaking:
                    self.last_voice_activity_at = asyncio.get_event_loop().time()
                    self.last_voice_activity_at_perf = now_perf

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
        self.stop_requested = True
        self.reply_cancel_event.set()
        print("Closing inbound audio stream and draining STT/LLM/TTS pipeline...")
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
        current_reply_task: asyncio.Task | None = None
        current_reply_text = ""
        current_reply_source = ""
        debounce_timer = None
        pending_transcript = None
        pending_lang = None

        def should_dispatch_transcript(text: str, is_final: bool) -> bool:
            normalized = normalize_transcript(text)
            if len(normalized) < 4:
                return False

            if not is_final:
                if (
                    len(normalized) < MIN_INTERIM_DISPATCH_CHARS
                    or transcript_word_count(normalized) < MIN_INTERIM_DISPATCH_WORDS
                ):
                    return False

            if self.last_dispatched_transcript:
                if normalized == self.last_dispatched_transcript:
                    return False

                if not is_final and self.last_dispatched_transcript.startswith(normalized):
                    return False

            if is_final and normalized == self.last_final_transcript:
                return False

            if (
                is_final
                and self.last_reply_source == "interim_pause"
                and self.last_reply_completed_at_perf is not None
                and (perf_counter() - self.last_reply_completed_at_perf) <= RECENT_REPLY_WINDOW_SECONDS
                and is_similar_transcript(self.last_reply_transcript, normalized)
            ):
                print(
                    "🔁 Skipping final transcript because a recent interim reply already covered it: "
                    f"'{normalized}'"
                )
                return False

            reference = self.last_dispatched_transcript
            if reference and normalized.startswith(reference):
                delta = normalized[len(reference):].strip()
                return is_final or len(delta) >= 12 or len(normalized.split()) - len(reference.split()) >= 3

            return True

        def local_pause_detected() -> bool:
            return (
                not self.user_speaking
                and (asyncio.get_event_loop().time() - self.last_voice_activity_at) >= LOCAL_ENDPOINT_SILENCE_SECONDS
            )

        async def process_transcript_with_streaming_tts(text: str, lang: str, source: str) -> None:
            """
            🚀 OPTIMIZED: Stream TTS while LLM is generating.
            
            Start TTS synthesis sentence-by-sentence IMMEDIATELY while LLM tokens arrive.
            This eliminates 200-500ms wait for full response completion!
            """
            try:
                if self.stop_requested or not self.running:
                    return

                # Reset cancel event for new reply
                self.reply_cancel_event.clear()

                self.reply_counter += 1
                timing = ReplyTiming(
                    reply_number=self.reply_counter,
                    source=source,
                    transcript=text,
                    language_code=lang,
                    call_started_at=self.call_started_at,
                    utterance_started_at=self.current_utterance_started_at_perf,
                    last_voice_activity_at=self.last_voice_activity_at_perf,
                    dispatch_started_at=perf_counter(),
                )
                timing.log_dispatch()

                async with self.reply_lock:
                    try:
                        # For WAV mode, still collect full response
                        if self.response_audio_mode == "wav":
                            full_response = ""
                            token_count = 0
                            
                            print(f"🚀 Starting LLM stream processing (WAV mode)...")
                            async for token in process_text_streaming(text, language_code=lang):
                                if self.stop_requested or self.reply_cancel_event.is_set():
                                    print("🛑 Stopping LLM stream due to stop/barging")
                                    break
                                if timing.llm_first_token_at is None:
                                    timing.llm_first_token_at = perf_counter()
                                    timing.log("llm_first_token", timing.llm_first_token_at)
                                full_response += token
                                token_count += 1
                                print(f"🤖 Token {token_count}: '{token}'", end="", flush=True)

                            print()
                            timing.llm_completed_at = perf_counter()
                            timing.log("llm_complete", timing.llm_completed_at, extra=f"tokens={token_count}")

                            response_text = full_response.strip()
                            if response_text and not self.stop_requested and not self.reply_cancel_event.is_set():
                                await send_wav_tts_reply(
                                    self.websocket,
                                    response_text,
                                    language_code=lang,
                                    output_sample_rate=self.input_sample_rate,
                                    timing=timing,
                                )
                        else:
                            # STREAMING MODE: Use optimized TTS-as-tokens-arrive
                            print(f"🚀 Starting optimized LLM→TTS streaming...")
                            
                            # Create token generator that also logs tokens
                            async def token_generator_with_logging():
                                token_count = 0
                                async for token in process_text_streaming(text, language_code=lang):
                                    if self.stop_requested or self.reply_cancel_event.is_set():
                                        print("🛑 Stopping LLM stream due to stop/barging")
                                        break
                                    if timing.llm_first_token_at is None:
                                        timing.llm_first_token_at = perf_counter()
                                        timing.log("llm_first_token", timing.llm_first_token_at)
                                    token_count += 1
                                    print(f"🤖 Token {token_count}: '{token}'", end="", flush=True)
                                    yield token
                                
                                print()
                                timing.llm_completed_at = perf_counter()
                                timing.log("llm_complete", timing.llm_completed_at, extra=f"tokens={token_count} early_audio_start")
                            
                            await stream_tts_reply_with_streaming_text(
                                self.websocket,
                                token_generator_with_logging(),
                                language_code=lang,
                                output_sample_rate=self.input_sample_rate,
                                cancel_event=self.reply_cancel_event,
                                timing=timing,
                            )
                        
                        self.last_reply_transcript = normalize_transcript(text)
                        self.last_reply_source = source
                        self.last_reply_completed_at_perf = perf_counter()
                        if self.response_audio_mode == "wav" and FIXED_FREESWITCH_REPLY_PATH.exists():
                            self.generated_reply_path = FIXED_FREESWITCH_REPLY_PATH
                            print(f"✅ Fresh FreeSWITCH reply file ready: {self.generated_reply_path}")

                    except asyncio.CancelledError:
                        print(f"🛑 TTS cancelled (barging)")
                    except Exception as exc:
                        print(f"Reply send failed: {exc!r}")

            except Exception as exc:
                print(f"LLM streaming failed: {exc!r}")

        async def replace_active_reply(text: str, lang: str, source: str) -> None:
            nonlocal current_reply_task, current_reply_text, current_reply_source

            if current_reply_task and not current_reply_task.done():
                print(f"🛑 Replacing active reply with {source} transcript...")
                self.reply_cancel_event.set()
                try:
                    await asyncio.wait_for(current_reply_task, timeout=0.75)
                except asyncio.TimeoutError:
                    current_reply_task.cancel()
                    try:
                        await current_reply_task
                    except asyncio.CancelledError:
                        pass

            current_reply_text = normalize_transcript(text)
            current_reply_source = source
            current_reply_task = asyncio.create_task(process_transcript_with_streaming_tts(text, lang, source))

        while True:
            try:
                # Use a short timeout to implement debouncing
                item = await asyncio.wait_for(self.transcript_queue.get(), timeout=TRANSCRIPT_POLL_SECONDS)
            except asyncio.TimeoutError:
                # When the caller pauses, promote the latest stable interim without waiting
                # for the STT provider to formally end the utterance.
                if self.stop_requested:
                    break

                if (
                    pending_transcript
                    and debounce_timer is not None
                    and asyncio.get_event_loop().time() >= debounce_timer
                    and local_pause_detected()
                ):
                    text_to_process = pending_transcript
                    lang_to_process = pending_lang
                    pending_transcript = None
                    pending_lang = None
                    self.last_dispatched_transcript = normalize_transcript(text_to_process)

                    print("⏱️ Debounce expired, dispatching buffered interim result...")
                    await replace_active_reply(text_to_process, lang_to_process, "interim_pause")
                continue

            if item is None:
                break

            if self.stop_requested:
                continue

            latest_item = item
            while True:
                try:
                    candidate = self.transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if candidate is None:
                    latest_item = None
                    self.stop_requested = True
                    break

                candidate_text, _, candidate_is_final = candidate
                if candidate_is_final:
                    latest_item = candidate
                    break
                latest_item = candidate

            if latest_item is None:
                break

            transcript_text, detected_lang, is_final = latest_item

            # For interim results: buffer and debounce to avoid too many LLM calls
            if not is_final:
                normalized_interim = normalize_transcript(transcript_text)
                print(f"🎤 INTERIM (buffering): {normalized_interim!r}")
                if not should_dispatch_transcript(transcript_text, is_final=False):
                    continue
                if pending_transcript:
                    normalized_pending = normalize_transcript(pending_transcript)
                    if (
                        len(normalized_interim) + 6 < len(normalized_pending)
                        and normalized_pending.startswith(normalized_interim)
                    ):
                        print(
                            "↩️ Ignoring regressive interim transcript in favor of the more complete buffered one: "
                            f"'{normalized_interim}'"
                        )
                        continue
                pending_transcript = transcript_text
                pending_lang = detected_lang
                debounce_timer = asyncio.get_event_loop().time() + INTERIM_DEBOUNCE_SECONDS
                continue

            # Final result arrived - process it immediately
            print(f"📝 FINAL Transcript: {transcript_text!r}")
            if not should_dispatch_transcript(transcript_text, is_final=True):
                continue
            pending_transcript = None  # Clear any pending interim
            pending_lang = None
            self.last_final_transcript = normalize_transcript(transcript_text)

            if (
                current_reply_task
                and not current_reply_task.done()
                and current_reply_source == "interim_pause"
                and is_similar_transcript(current_reply_text, self.last_final_transcript)
            ):
                print(
                    "✅ Keeping active interim reply because the final transcript confirms the same intent: "
                    f"'{self.last_final_transcript}'"
                )
                self.last_dispatched_transcript = self.last_final_transcript
                continue

            self.last_dispatched_transcript = self.last_final_transcript
            await replace_active_reply(transcript_text, detected_lang, "stt_final")

        if current_reply_task and not current_reply_task.done():
            try:
                await current_reply_task
            except asyncio.CancelledError:
                pass


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
 
