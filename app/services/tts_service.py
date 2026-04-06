"""
Text-to-Speech module using Azure TTS
Supports both batch and streaming audio generation
Multilingual support with language-specific voices
"""

import os
import asyncio
import aiohttp
import tempfile
from time import monotonic
from dotenv import load_dotenv
from .language_utils import get_azure_voice_for_language, get_language_name

load_dotenv()
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "centralindia")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-IN-ArjunNeural")
AZURE_TTS_TIMEOUT_SECONDS = int(os.getenv("AZURE_TTS_TIMEOUT_SECONDS", "10"))
AZURE_TTS_TOKEN_TTL_SECONDS = int(os.getenv("AZURE_TTS_TOKEN_TTL_SECONDS", "540"))

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()
_token_cache = {"value": None, "expires_at": 0.0}
_token_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _session

    if _session and not _session.closed:
        return _session

    async with _session_lock:
        if _session and not _session.closed:
            return _session

        timeout = aiohttp.ClientTimeout(total=AZURE_TTS_TIMEOUT_SECONDS)
        _session = aiohttp.ClientSession(timeout=timeout)
        return _session


async def _get_access_token() -> str:
    cached_token = _token_cache["value"]
    if cached_token and monotonic() < _token_cache["expires_at"]:
        return cached_token

    async with _token_lock:
        cached_token = _token_cache["value"]
        if cached_token and monotonic() < _token_cache["expires_at"]:
            return cached_token

        session = await _get_session()
        token_url = f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        print(f"🔐 Fetching Azure TTS token from {AZURE_TTS_REGION}...")
        async with session.post(token_url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Failed to get access token: {response.status} - {error_text}")

            access_token = await response.text()

        _token_cache["value"] = access_token
        _token_cache["expires_at"] = monotonic() + AZURE_TTS_TOKEN_TTL_SECONDS
        return access_token


async def warmup_tts_connection() -> None:
    """Prime the shared HTTP session and Azure auth token before first reply."""
    if not AZURE_TTS_KEY:
        return

    try:
        await _get_session()
        await _get_access_token()
        print("🔥 Azure TTS warmup complete")
    except Exception as exc:
        # Warmup should never block the live call path.
        print(f"⚠️ Azure TTS warmup failed: {exc}")


def _build_ssml(text: str, voice: str, language_code: str) -> str:
    clean_text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    return f"""
    <speak version='1.0' xml:lang='{language_code}'>
        <voice xml:lang='{language_code}' name='{voice}'>
            {clean_text}
        </voice>
    </speak>
    """.strip()

async def generate_speech_stream(text: str, voice: str = None, language_code: str = "en") -> bytes:
    """
    Generate speech audio from text using Azure TTS
    Supports multilingual output with language-specific voices

    Args:
        text: Text to convert to speech
        voice: Voice to use (optional, will auto-select if not provided)
        language_code: ISO 639-1 language code (e.g., 'hi', 'en')

    Returns:
        Audio data in MP3 format
    """
    if not AZURE_TTS_KEY:
        print("❌ AZURE_TTS_KEY not set in environment")
        return b""

    if not text.strip():
        print("❌ No text provided for TTS")
        return b""

    # Auto-select voice based on language if not provided
    if not voice:
        voice = get_azure_voice_for_language(language_code)

    lang_name = get_language_name(language_code)
    print(f"🔊 Converting to speech ({lang_name}): '{text[:50]}...' using voice: {voice}")

    try:
        session = await _get_session()
        access_token = await _get_access_token()
        tts_url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        ssml = _build_ssml(text, voice, language_code)
        tts_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio/wav;codec=pcm;samplerate=16000",
            "User-Agent": "FreeSWITCH-AI-Bridge",
        }

        print("🎵 Sending TTS request...")
        async with session.post(tts_url, headers=tts_headers, data=ssml) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ TTS request failed: {response.status} - {error_text}")
                return b""

            audio_data = await response.read()
            print(f"✅ Generated TTS audio: {len(audio_data)} bytes")
            return audio_data

    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        import traceback
        print(f"❌ Full error: {traceback.format_exc()}")
        return b""

async def save_audio_file(audio_data: bytes, format: str = "mp3") -> str:
    """
    Save audio data to a temporary file

    Args:
        audio_data: Audio bytes
        format: File format (mp3, wav, etc.)

    Returns:
        Path to saved audio file
    """
    if not audio_data:
        return None

    try:
        fd, path = tempfile.mkstemp(suffix=f".{format}")
        os.write(fd, audio_data)
        os.close(fd)
        return path
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")
        return None

async def generate_speech_stream_chunked(text: str, voice: str = None, language_code: str = "en"):
    """
    Generate speech audio from text using Azure TTS with streaming
    Yields audio chunks as they are generated for real-time playback
    Supports multilingual output with language-specific voices

    Args:
        text: Text to convert to speech
        voice: Voice to use (optional, will auto-select if not provided)
        language_code: ISO 639-1 language code (e.g., 'hi', 'en')

    Yields:
        Audio chunks in MP3 format
    """
    if not AZURE_TTS_KEY:
        print("❌ AZURE_TTS_KEY not set in environment")
        return

    if not text.strip():
        print("❌ No text provided for TTS")
        return

    # Auto-select voice based on language if not provided
    if not voice:
        voice = get_azure_voice_for_language(language_code)

    lang_name = get_language_name(language_code)
    print(f"🔊 Converting to speech (streaming, {lang_name}): '{text[:50]}...' using voice: {voice}")

    try:
        session = await _get_session()
        access_token = await _get_access_token()
        tts_url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        ssml = _build_ssml(text, voice, language_code)
        tts_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-16khz-16bit-mono-pcm",
            "User-Agent": "FreeSWITCH-AI-Bridge",
        }

        print("🎵 Sending TTS streaming request...")
        async with session.post(tts_url, headers=tts_headers, data=ssml) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ TTS request failed: {response.status} - {error_text}")
                return

            chunk_count = 0
            async for chunk in response.content.iter_chunked(2048):
                if chunk:
                    chunk_count += 1
                    print(f"🎵 Audio chunk {chunk_count}: {len(chunk)} bytes")
                    yield chunk

            print(f"✅ TTS streaming complete: {chunk_count} chunks")

    except Exception as e:
        print(f"❌ TTS Streaming Error: {str(e)}")
        import traceback
        print(f"❌ Full error: {traceback.format_exc()}")
