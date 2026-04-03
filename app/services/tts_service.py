"""
Text-to-Speech module using Azure TTS
Supports both batch and streaming audio generation
Multilingual support with language-specific voices
"""

import os
import asyncio
import aiohttp
import tempfile
from dotenv import load_dotenv
from .language_utils import get_azure_voice_for_language, get_language_name

load_dotenv()
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "centralindia")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-IN-ArjunNeural")

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
        # Get access token
        token_url = f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_TTS_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        async with aiohttp.ClientSession() as session:
            # Get access token
            print(f"🔐 Getting access token from {AZURE_TTS_REGION}...")
            async with session.post(token_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Failed to get access token: {response.status} - {error_text}")
                    return b""
                access_token = await response.text()
                print("✅ Access token obtained")

            # Generate speech
            tts_url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

            # Clean text for SSML (escape special characters)
            clean_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

            # Use language code in SSML - xml:lang should be just language code (e.g., 'kn'), not 'kn-IN'
            ssml = f"""
            <speak version='1.0' xml:lang='{language_code}'>
                <voice xml:lang='{language_code}' name='{voice}'>
                    {clean_text}
                </voice>
            </speak>
            """

            tts_headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'audio/wav;codec=pcm;samplerate=16000',
                'User-Agent': 'FreeSWITCH-AI-Bridge'
            }

            print(f"🎵 Sending TTS request...")
            async with session.post(tts_url, headers=tts_headers, data=ssml.strip()) as response:
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
        # Get access token
        token_url = f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_TTS_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        async with aiohttp.ClientSession() as session:
            # Get access token
            print(f"🔐 Getting access token from {AZURE_TTS_REGION}...")
            async with session.post(token_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Failed to get access token: {response.status} - {error_text}")
                    return
                access_token = await response.text()
                print("✅ Access token obtained")

            # Generate speech with streaming
            tts_url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

            # Clean text for SSML (escape special characters)
            clean_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

            # Use language code in SSML - xml:lang should be just language code (e.g., 'kn'), not 'kn-IN'
            ssml = f"""
            <speak version='1.0' xml:lang='{language_code}'>
                <voice xml:lang='{language_code}' name='{voice}'>
                    {clean_text}
                </voice>
            </speak>
            """

            tts_headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'raw-16khz-16bit-mono-pcm',
                'User-Agent': 'FreeSWITCH-AI-Bridge'
            }

            print(f"🎵 Sending TTS streaming request...")
            async with session.post(tts_url, headers=tts_headers, data=ssml.strip()) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ TTS request failed: {response.status} - {error_text}")
                    return

                # Stream audio chunks as they arrive
                chunk_count = 0
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        chunk_count += 1
                        print(f"🎵 Audio chunk {chunk_count}: {len(chunk)} bytes")
                        yield chunk

                print(f"✅ TTS streaming complete: {chunk_count} chunks")

    except Exception as e:
        print(f"❌ TTS Streaming Error: {str(e)}")
        import traceback
        print(f"❌ Full error: {traceback.format_exc()}")