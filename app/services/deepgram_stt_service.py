"""
Speech-to-Text module using Deepgram API.
Implements the same public async interfaces as the existing Soniox service.
"""

import asyncio
import json
import os
from collections import Counter
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv

from .language_utils import detect_language

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_STT_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_STT_ENDPOINTING_MS", "60"))  # ULTRA: 60ms from 80ms - faster silence detection
DEEPGRAM_STT_UTTERANCE_END_MS = max(300, int(os.getenv("DEEPGRAM_STT_UTTERANCE_END_MS", "300")))  # ULTRA: 300ms from 500ms - trigger faster
DEEPGRAM_STT_KEEPALIVE_SECONDS = float(os.getenv("DEEPGRAM_STT_KEEPALIVE_SECONDS", "3"))  # Reduced from 4s for faster keepalive
DEEPGRAM_STT_TIMEOUT_SECONDS = float(os.getenv("DEEPGRAM_STT_TIMEOUT_SECONDS", "30"))  # Reduced from 65s for faster failure detection
DEEPGRAM_REST_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _session

    if _session and not _session.closed:
        return _session

    async with _session_lock:
        if _session and not _session.closed:
            return _session

        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
        _session = aiohttp.ClientSession(timeout=timeout)
        return _session


def _auth_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _normalize_language_tag(language_tag: str | None) -> str:
    if not language_tag:
        return ""
    primary = language_tag.split("-")[0].strip().lower()
    return primary


def _extract_language_from_alternative(alternative: dict, transcript: str) -> str:
    words = alternative.get("words") or []
    word_languages = [
        _normalize_language_tag(word.get("language"))
        for word in words
        if _normalize_language_tag(word.get("language"))
    ]
    if word_languages:
        return Counter(word_languages).most_common(1)[0][0]

    languages = alternative.get("languages") or []
    if languages:
        normalized = _normalize_language_tag(languages[0])
        if normalized:
            return normalized

    return detect_language(transcript)


def _get_alternative(payload: dict) -> dict:
    if payload.get("channel"):
        channels = ((payload.get("channel") or {}).get("alternatives")) or []
    else:
        channels = ((((payload.get("results") or {}).get("channels")) or [{}])[0].get("alternatives")) or []
    if channels:
        return channels[0]
    return {}


def _join_segments(segments: list[str]) -> str:
    return " ".join(segment.strip() for segment in segments if segment and segment.strip()).strip()


def _build_streaming_url(sample_rate: int, process_interim: bool) -> str:
    utterance_end_ms = max(1000, DEEPGRAM_STT_UTTERANCE_END_MS)
    params = {
        "model": DEEPGRAM_STT_MODEL,
        "language": "multi",
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "channels": "1",
        "interim_results": "true" if process_interim else "false",
        "smart_format": "true",
        "punctuate": "true",
        "vad_events": "true",
        "endpointing": str(DEEPGRAM_STT_ENDPOINTING_MS),
    }
    if process_interim:
        params["utterance_end_ms"] = str(utterance_end_ms)
    return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"


def _build_prerecorded_url(audio_format: str) -> tuple[str, str]:
    normalized = (audio_format or "wav").strip().lower()
    params = {
        "model": DEEPGRAM_STT_MODEL,
        "language": "multi",
        "smart_format": "true",
        "punctuate": "true",
    }

    content_type_map = {
        "wav": "audio/wav",
        "wave": "audio/wav",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "webm": "audio/webm",
        "m4a": "audio/mp4",
    }

    if normalized in {"pcm", "pcm_s16le", "linear16", "raw"}:
        params["encoding"] = "linear16"
        params["sample_rate"] = "16000"
        content_type = "audio/raw"
    else:
        content_type = content_type_map.get(normalized, "application/octet-stream")

    return f"{DEEPGRAM_REST_URL}?{urlencode(params)}", content_type


def build_transcript_from_responses(responses: list[dict]) -> str:
    """Build a transcript string from Deepgram REST or streaming responses."""
    final_parts: list[str] = []

    for response in responses:
        alternative = _get_alternative(response)
        transcript = (alternative.get("transcript") or "").strip()
        if transcript:
            final_parts.append(transcript)

    return _join_segments(final_parts)


async def transcribe_audio(audio_data: bytes, audio_format: str = "wav") -> str:
    """
    Transcribe audio data using Deepgram STT.
    """
    if not DEEPGRAM_API_KEY:
        raise Exception("DEEPGRAM_API_KEY not set in environment")

    if not audio_data:
        return ""

    try:
        session = await _get_session()
        url, content_type = _build_prerecorded_url(audio_format)
        headers = _auth_headers(content_type)

        async with session.post(url, headers=headers, data=audio_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Deepgram STT request failed: {response.status} - {error_text}")

            payload = await response.json()

        alternatives = (((payload.get("results") or {}).get("channels") or [{}])[0].get("alternatives") or [{}])
        transcript = (alternatives[0].get("transcript") or "").strip()
        return transcript

    except Exception as exc:
        print(f"❌ Deepgram STT Error: {exc}")
        return ""


async def transcribe_audio_streaming(audio_queue: asyncio.Queue, sample_rate: int = 16000):
    """
    Stream audio chunks to Deepgram and yield final transcripts.

    Yields:
        Tuple of (transcript, language_code)
    """
    async for transcript, language_code, is_final in transcribe_audio_streaming_continuous(
        audio_queue,
        sample_rate=sample_rate,
        process_interim=False,
    ):
        if is_final and transcript:
            yield (transcript, language_code)


async def transcribe_audio_streaming_continuous(
    audio_queue: asyncio.Queue,
    sample_rate: int = 16000,
    process_interim: bool = True,
):
    """
    Keep one Deepgram session open and yield transcripts as they become available.

    With process_interim=True, yields PARTIAL results for low-latency LLM processing.
    With process_interim=False, yields only FINAL results.

    Yields:
        Tuple of (transcript, language_code, is_final)
    """
    if not DEEPGRAM_API_KEY:
        raise Exception("DEEPGRAM_API_KEY not set in environment")

    response_count = 0
    last_interim_text = ""
    last_interim_lang = "en"
    last_emitted_final = ""
    final_segments: list[str] = []
    final_lang = "en"
    had_final_result = False

    def flush_final_segments() -> tuple[str, str] | None:
        nonlocal final_segments, final_lang, last_interim_text, had_final_result, last_emitted_final

        combined = _join_segments(final_segments)
        if not combined or combined == last_emitted_final:
            final_segments = []
            last_interim_text = ""
            return None

        last_emitted_final = combined
        had_final_result = True
        final_segments = []
        last_interim_text = ""
        return combined, final_lang

    try:
        session = await _get_session()
        ws_url = _build_streaming_url(sample_rate=sample_rate, process_interim=process_interim)

        async with session.ws_connect(
            ws_url,
            headers=_auth_headers(),
            heartbeat=max(DEEPGRAM_STT_KEEPALIVE_SECONDS, 3),
            autoping=True,
        ) as ws:
            print("📝 Continuous Deepgram STT streaming initialized")
            send_task = asyncio.create_task(_send_audio_chunks_to_deepgram(ws, audio_queue, sample_rate=sample_rate))

            try:
                while True:
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=DEEPGRAM_STT_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        flushed = flush_final_segments()
                        if flushed:
                            transcript_text, detected_lang = flushed
                            print(
                                "⚠️ Deepgram STT timeout - promoting buffered final: "
                                f"'{transcript_text}' ({detected_lang})"
                            )
                            yield (transcript_text, detected_lang, True)
                        elif not had_final_result and last_interim_text:
                            print(
                                "⚠️ Deepgram STT timeout - promoting last interim to final: "
                                f"'{last_interim_text}' ({last_interim_lang})"
                            )
                            yield (last_interim_text, last_interim_lang, True)
                        print("⚠️ Continuous Deepgram STT timeout - closing stream")
                        break

                    if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                        flushed = flush_final_segments()
                        if flushed:
                            transcript_text, detected_lang = flushed
                            yield (transcript_text, detected_lang, True)
                        break

                    if message.type == aiohttp.WSMsgType.ERROR:
                        print(f"❌ Deepgram STT WebSocket error: {ws.exception()!r}")
                        break

                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue

                    response_count += 1
                    data = json.loads(message.data)
                    message_type = data.get("type")

                    if message_type == "Results":
                        alternative = _get_alternative(data)
                        transcript_text = (alternative.get("transcript") or "").strip()

                        if not transcript_text:
                            if data.get("speech_final"):
                                flushed = flush_final_segments()
                                if flushed:
                                    transcript_text, detected_lang = flushed
                                    print(f"✅ Deepgram STT FINAL #{response_count}: '{transcript_text}' ({detected_lang})")
                                    yield (transcript_text, detected_lang, True)
                            continue

                        detected_lang = _extract_language_from_alternative(alternative, transcript_text)

                        if data.get("is_final"):
                            final_segments.append(transcript_text)
                            final_lang = detected_lang
                            print(
                                f"✅ Deepgram STT final segment #{response_count}: "
                                f"'{transcript_text}' ({detected_lang})"
                            )

                            if data.get("speech_final") or data.get("from_finalize"):
                                flushed = flush_final_segments()
                                if flushed:
                                    transcript_text, detected_lang = flushed
                                    print(f"✅ Deepgram STT FINAL #{response_count}: '{transcript_text}' ({detected_lang})")
                                    yield (transcript_text, detected_lang, True)

                        elif process_interim:
                            combined_interim = _join_segments(final_segments + [transcript_text])
                            if (
                                combined_interim
                                and combined_interim != last_interim_text
                                and len(combined_interim) > 2
                            ):
                                last_interim_text = combined_interim
                                last_interim_lang = detected_lang
                                print(f"🎤 Deepgram interim: '{combined_interim}' ({detected_lang})")
                                yield (combined_interim, detected_lang, False)

                    elif message_type == "UtteranceEnd":
                        flushed = flush_final_segments()
                        if flushed:
                            transcript_text, detected_lang = flushed
                            print(f"✅ Deepgram STT utterance end: '{transcript_text}' ({detected_lang})")
                            yield (transcript_text, detected_lang, True)

                    elif message_type == "Metadata":
                        request_id = data.get("request_id")
                        if request_id:
                            print(f"🔎 Deepgram STT request_id={request_id}")

                    elif message_type == "Error":
                        print(f"❌ Deepgram STT provider error: {data}")
                        break

            finally:
                try:
                    await send_task
                except Exception as exc:
                    print(f"⚠️ Deepgram audio sender stopped with error: {exc}")

        if not had_final_result and last_interim_text:
            print(
                "⚠️ Deepgram stream ended without final result, promoting last interim to final: "
                f"'{last_interim_text}' ({last_interim_lang})"
            )
            yield (last_interim_text, last_interim_lang, True)

    except Exception as exc:
        print(f"❌ Continuous Deepgram STT Streaming Error: {exc}")


async def _send_audio_chunks_to_deepgram(
    ws: aiohttp.ClientWebSocketResponse,
    audio_queue: asyncio.Queue,
    sample_rate: int = 16000,
):
    """
    Send audio chunks from queue to Deepgram WebSocket.
    """
    chunk_count = 0
    detected_chunk_size = None
    bytes_per_sample = 2

    try:
        while True:
            try:
                audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=DEEPGRAM_STT_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "KeepAlive"})
                continue

            if audio_chunk is None:
                print(f"📝 Sending CloseStream to Deepgram STT (sent {chunk_count} chunks)")
                await ws.send_json({"type": "CloseStream"})
                break

            if detected_chunk_size is None and len(audio_chunk) > 0:
                detected_chunk_size = len(audio_chunk)
                samples_per_chunk = detected_chunk_size // bytes_per_sample
                chunk_duration = samples_per_chunk / sample_rate if sample_rate else 0
                print(
                    "📊 Deepgram STT streaming: "
                    f"{detected_chunk_size} bytes/chunk = {samples_per_chunk} samples = "
                    f"{chunk_duration * 1000:.1f}ms per chunk"
                )

            await ws.send_bytes(audio_chunk)
            chunk_count += 1

            if chunk_count % 5 == 0 and detected_chunk_size:
                total_bytes = chunk_count * detected_chunk_size
                duration_sec = total_bytes / (sample_rate * bytes_per_sample)
                print(
                    f"📤 Sent {chunk_count} audio chunks to Deepgram "
                    f"({total_bytes} bytes, ~{duration_sec:.1f}s audio)"
                )

    except Exception as exc:
        print(f"❌ Error sending audio chunks to Deepgram: {exc}")
