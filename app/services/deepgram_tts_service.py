"""
Text-to-Speech module using Deepgram Aura TTS.
Implements the same public async interfaces as the existing Azure service.
"""

import asyncio
import os
import re
import tempfile
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv

from .language_utils import get_language_name

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_TIMEOUT_SECONDS = int(os.getenv("DEEPGRAM_TTS_TIMEOUT_SECONDS", "15"))  # Reduced from 20s
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "16000"))
DEEPGRAM_TTS_CHUNK_SIZE = int(os.getenv("DEEPGRAM_TTS_CHUNK_SIZE", "1024"))  # Reduced from 2048 for faster streaming
DEEPGRAM_TTS_MAX_TEXT_CHARS = int(os.getenv("DEEPGRAM_TTS_MAX_TEXT_CHARS", "1800"))
DEEPGRAM_TTS_MIP_OPT_OUT = os.getenv("DEEPGRAM_TTS_MIP_OPT_OUT", "false").strip().lower() == "true"
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

DEEPGRAM_TTS_VOICE_MAP = {
    "en": "aura-2-thalia-en",
    "es": "aura-2-celeste-es",
    "fr": "aura-2-agathe-fr",
    "de": "aura-2-viktoria-de",
    "nl": "aura-2-rhea-nl",
    "it": "aura-2-livia-it",
    "ja": "aura-2-izanami-ja",
}

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _session

    if _session and not _session.closed:
        return _session

    async with _session_lock:
        if _session and not _session.closed:
            return _session

        timeout = aiohttp.ClientTimeout(total=DEEPGRAM_TTS_TIMEOUT_SECONDS)
        _session = aiohttp.ClientSession(timeout=timeout)
        return _session


def _normalize_language_code(language_code: str | None) -> str:
    if not language_code:
        return "en"
    return language_code.split("-")[0].strip().lower() or "en"


def _select_model(voice: str | None, language_code: str) -> tuple[str, str]:
    normalized_lang = _normalize_language_code(language_code)
    if voice:
        return voice, normalized_lang

    mapped_model = DEEPGRAM_TTS_VOICE_MAP.get(normalized_lang)
    if mapped_model:
        return mapped_model, normalized_lang

    return DEEPGRAM_TTS_MODEL, "en"


def _build_tts_url(model: str, sample_rate: int) -> str:
    params = {
        "model": model,
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "container": "none",
    }
    if DEEPGRAM_TTS_MIP_OPT_OUT:
        params["mip_opt_out"] = "true"
    return f"{DEEPGRAM_TTS_URL}?{urlencode(params)}"


def _split_text_for_tts(text: str, max_chars: int = DEEPGRAM_TTS_MAX_TEXT_CHARS) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for part in parts:
        if not part:
            continue

        if len(part) > max_chars:
            flush_current()
            words = part.split()
            oversize = ""
            for word in words:
                candidate = f"{oversize} {word}".strip()
                if len(candidate) <= max_chars:
                    oversize = candidate
                else:
                    if oversize:
                        chunks.append(oversize)
                    oversize = word
            if oversize:
                chunks.append(oversize)
            continue

        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            current = part

    flush_current()
    return chunks or [normalized]


async def warmup_tts_connection() -> None:
    """Prime the shared HTTP session before first reply."""
    if not DEEPGRAM_API_KEY:
        return

    try:
        await _get_session()
        print("🔥 Deepgram TTS warmup complete")
    except Exception as exc:
        print(f"⚠️ Deepgram TTS warmup failed: {exc}")


async def generate_speech_stream(text: str, voice: str = None, language_code: str = "en") -> bytes:
    """
    Generate speech audio from text using Deepgram Aura TTS.

    Returns:
        Raw 16kHz mono PCM bytes.
    """
    audio_parts = bytearray()

    async for chunk in generate_speech_stream_chunked(text, voice=voice, language_code=language_code):
        if chunk:
            audio_parts.extend(chunk)

    return bytes(audio_parts)


async def save_audio_file(audio_data: bytes, format: str = "mp3") -> str:
    """
    Save audio data to a temporary file.
    """
    if not audio_data:
        return None

    try:
        fd, path = tempfile.mkstemp(suffix=f".{format}")
        os.write(fd, audio_data)
        os.close(fd)
        return path
    except Exception as exc:
        print(f"Error saving audio file: {exc}")
        return None


async def generate_speech_stream_chunked(text: str, voice: str = None, language_code: str = "en"):
    """
    Generate speech audio from text using Deepgram Aura TTS with chunked streaming.

    Yields:
        Raw 16kHz mono PCM chunks.
    """
    if not DEEPGRAM_API_KEY:
        print("❌ DEEPGRAM_API_KEY not set in environment")
        return

    if not text or not text.strip():
        print("❌ No text provided for TTS")
        return

    model, effective_lang = _select_model(voice, language_code)
    lang_name = get_language_name(language_code)

    if not voice and effective_lang != _normalize_language_code(language_code):
        print(
            f"⚠️ Deepgram TTS does not currently provide a native voice for '{language_code}', "
            f"falling back to model: {model}"
        )

    text_chunks = _split_text_for_tts(text)
    session = await _get_session()
    total_chunks = 0

    print(
        f"🔊 Converting to speech with Deepgram ({lang_name}): "
        f"'{text[:50]}...' using model: {model}"
    )

    try:
        for segment_index, segment in enumerate(text_chunks, start=1):
            url = _build_tts_url(model=model, sample_rate=DEEPGRAM_TTS_SAMPLE_RATE)
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"text": segment}

            print(
                f"🎵 Sending Deepgram TTS request {segment_index}/{len(text_chunks)} "
                f"({len(segment)} chars)"
            )
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Deepgram TTS request failed: {response.status} - {error_text}")
                    return

                async for chunk in response.content.iter_chunked(DEEPGRAM_TTS_CHUNK_SIZE):
                    if not chunk:
                        continue
                    total_chunks += 1
                    yield chunk

        print(f"✅ Deepgram TTS streaming complete: {total_chunks} chunks")

    except Exception as exc:
        print(f"❌ Deepgram TTS Streaming Error: {exc}")
