# stt.py
"""
Speech-to-Text module using Soniox API
Supports multilingual detection and transcription
"""

import os
import asyncio
import json
import tempfile
import websockets
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any
import unicodedata

load_dotenv()
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# Import language utilities
from .language_utils import detect_language


async def transcribe_audio(audio_data: bytes, audio_format: str = "wav") -> str:
    """
    Transcribe audio data using Soniox STT

    Args:
        audio_data: Raw audio bytes
        audio_format: Format of audio (wav, mp3, etc.)

    Returns:
        Transcribed text
    """
    if not SONIOX_API_KEY:
        raise Exception("SONIOX_API_KEY not set in environment")

    try:
        # Convert audio_data to the format expected by Soniox
        # For simplicity, assuming audio_data is already in the right format
        # In production, you'd convert different formats to pcm_s16le

        responses = []

        async with websockets.connect(SONIOX_WS_URL, max_size=None) as ws:
            # Send configuration
            config = {
                "api_key": SONIOX_API_KEY,
                "model": "stt-rt-v3",
                "audio_format": "pcm_s16le",
                "num_channels": 1,
                "sample_rate": 16000,
                "language_hints": ["en"],
                "enable_speaker_diarization": False,
                "enable_language_identification": False,
                "enable_profanity_filter": False,
                "enable_endpoint_detection": True,
                "enable_dictation": False
            }
            await ws.send(json.dumps(config))

            # Send audio data
            await ws.send(audio_data)

            # Send end-of-stream
            await ws.send(b"")

            # Collect responses
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    if isinstance(response, bytes):
                        continue

                    data = json.loads(response)
                    responses.append(data)

                    if data.get("finished") or data.get("error_code"):
                        if data.get("error_code"):
                            raise Exception(f"Soniox error: {data.get('error_message')}")
                        break
            except asyncio.TimeoutError:
                pass

        # Build transcript from responses
        return build_transcript_from_responses(responses)

    except Exception as e:
        print(f"STT Error: {str(e)}")
        return ""

def build_transcript_from_responses(responses: List[Dict[str, Any]]) -> str:
    """Build transcript from Soniox response tokens"""
    all_tokens = []
    last_non_error_tokens = []

    for r in responses:
        tokens = r.get("tokens", []) or []
        if tokens:
            all_tokens.extend(tokens)
            if not r.get("error_code"):
                last_non_error_tokens = tokens

    # Prefer final tokens
    final_tokens = [t for t in all_tokens if t.get("is_final")]
    if final_tokens:
        pieces = [t.get("text", "") for t in final_tokens]
    elif last_non_error_tokens:
        pieces = [t.get("text", "") for t in last_non_error_tokens]
    else:
        return ""

    return "".join(pieces).strip()

async def transcribe_audio_streaming(audio_queue: asyncio.Queue, sample_rate: int = 16000):
    """
    Stream audio chunks to Soniox and get real-time partial transcripts
    FIXED: Supports multilingual detection

    Args:
        audio_queue: AsyncQueue containing audio chunks (bytes)
        sample_rate: Audio sample rate (default: 16000)

    Yields:
        Tuple of (transcript, language_code) 
        language_code: ISO 639-1 code (e.g., 'en', 'hi', 'es')
    """
    if not SONIOX_API_KEY:
        raise Exception("SONIOX_API_KEY not set in environment")

    try:
        full_transcript = ""
        received_final = False
        response_count = 0
        last_non_english_transcript = ""  # Fallback option

        async with websockets.connect(SONIOX_WS_URL, max_size=None) as ws:
            # Multilingual with English bias to prevent false non-English detection
            config = {
                "api_key": SONIOX_API_KEY,
                "model": "stt-rt-v3",
                "audio_format": "pcm_s16le",
                "num_channels": 1,
                "sample_rate": sample_rate,
                "language_hints": ["en", "hi", "kn", "ta", "te", "ml", "gu", "mr", "bn"],  # English first, then other languages
                "enable_speaker_diarization": False,
                "enable_language_identification": False,
                "enable_profanity_filter": False,
                "enable_endpoint_detection": True,
                "enable_dictation": False
            }
            await ws.send(json.dumps(config))
            print("📝 STT streaming initialized (multilingual mode)")

            # Create task to send audio chunks
            send_task = asyncio.create_task(
                _send_audio_chunks_to_soniox(ws, audio_queue, sample_rate=sample_rate)
            )

            # Receive and yield transcripts
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=45.0)
                    if isinstance(response, bytes):
                        continue

                    response_count += 1
                    data = json.loads(response)

                    # Extract tokens
                    tokens = data.get("tokens", []) or []
                    if tokens:
                        # Build text from tokens
                        text_parts = []
                        has_final = False

                        for token in tokens:
                            text = token.get("text", "")
                            if text and text != "<end>":  # Skip endpoint markers
                                text_parts.append(text)
                            if token.get("is_final"):
                                has_final = True

                        transcript_text = "".join(text_parts).strip()

                        # Log for debugging
                        if has_final:
                            print(f"📝 Response #{response_count}: FINAL - '{transcript_text}'")

                        # Accept any final transcript (any language)
                        if has_final and transcript_text and not received_final:
                            # Detect the language from the script
                            detected_lang = detect_language(transcript_text)
                            full_transcript = transcript_text
                            print(f"✅ STT ACCEPTED: '{full_transcript}' (Language: {detected_lang})")
                            yield (full_transcript, detected_lang)
                            received_final = True

                    # Check if finished
                    if data.get("finished"):
                        if received_final:
                            print(f"✅ STT stream finished normally")
                        else:
                            print(f"⚠️ STT finished without final result")
                        break

                    if data.get("error_code"):
                        error_code = data.get('error_code')
                        error_msg = data.get('error_message', 'Unknown error')
                        print(f"❌ STT ERROR {error_code}: {error_msg}")

            except asyncio.TimeoutError:
                if full_transcript:
                    detected_lang = detect_language(full_transcript)
                    print(f"⚠️ STT timeout - using: '{full_transcript}'")
                    yield (full_transcript, detected_lang)
                else:
                    print("⚠️ STT timeout - no valid transcript received")
            finally:
                try:
                    await send_task
                except:
                    pass

    except Exception as e:
        print(f"❌ STT Streaming Error: {str(e)}")


async def transcribe_audio_streaming_continuous(audio_queue: asyncio.Queue, sample_rate: int = 16000, process_interim: bool = True):
    """
    Keep one Soniox session open and yield transcripts as they become available.

    With process_interim=True, yields PARTIAL results for low-latency LLM processing.
    With process_interim=False, yields only FINAL results.

    Args:
        audio_queue: AsyncQueue containing audio chunks (bytes)
        sample_rate: Audio sample rate (default: 16000)
        process_interim: If True, yield partial results immediately for low-latency processing

    Yields:
        Tuple of (transcript, language_code, is_final) for every utterance
    """
    if not SONIOX_API_KEY:
        raise Exception("SONIOX_API_KEY not set in environment")

    try:
        response_count = 0
        last_interim_text = ""  # Track recent partials to avoid duplicates
        last_interim_lang = "en"
        had_final_result = False

        async with websockets.connect(SONIOX_WS_URL, max_size=None) as ws:
            config = {
                "api_key": SONIOX_API_KEY,
                "model": "stt-rt-v3",
                "audio_format": "pcm_s16le",
                "num_channels": 1,
                "sample_rate": sample_rate,
                "language_hints": ["en", "hi", "kn", "ta", "te", "ml", "gu", "mr", "bn"],
                "enable_speaker_diarization": False,
                "enable_language_identification": False,
                "enable_profanity_filter": False,
                "enable_endpoint_detection": True,
                "enable_dictation": False,
            }
            await ws.send(json.dumps(config))
            print("📝 Continuous STT streaming initialized (with interim results for low-latency)")

            send_task = asyncio.create_task(_send_audio_chunks_to_soniox(ws, audio_queue, sample_rate=sample_rate))

            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    if isinstance(response, bytes):
                        continue

                    response_count += 1
                    data = json.loads(response)
                    tokens = data.get("tokens", []) or []

                    if tokens:
                        text_parts = []
                        has_final = False

                        for token in tokens:
                            text = token.get("text", "")
                            if text and text != "<end>":
                                text_parts.append(text)
                            if token.get("is_final"):
                                has_final = True

                        transcript_text = "".join(text_parts).strip()

                        # Yield FINAL results (always)
                        if has_final and transcript_text:
                            detected_lang = detect_language(transcript_text)
                            print(f"✅ STT FINAL #{response_count}: '{transcript_text}' ({detected_lang})")
                            yield (transcript_text, detected_lang, True)
                            had_final_result = True
                            last_interim_text = ""
                            last_interim_lang = detected_lang

                        # Yield INTERIM results (if enabled and different from last)
                        elif process_interim and transcript_text and transcript_text != last_interim_text and len(transcript_text) > 2:
                            detected_lang = detect_language(transcript_text)
                            print(f"🎤 STT interim: '{transcript_text}' ({detected_lang})")
                            yield (transcript_text, detected_lang, False)
                            last_interim_text = transcript_text
                            last_interim_lang = detected_lang

                    if data.get("finished"):
                        if not had_final_result and last_interim_text:
                            print(f"⚠️ STT finished without final result, promoting last interim to final: '{last_interim_text}' ({last_interim_lang})")
                            yield (last_interim_text, last_interim_lang, True)
                        print("✅ Continuous STT stream finished")
                        break

                    if data.get("error_code"):
                        error_code = data.get("error_code")
                        error_msg = data.get("error_message", "Unknown error")
                        if error_code == 408:
                            print(f"ℹ️ STT stream ended without more speech ({error_code}: {error_msg})")
                            if not had_final_result and last_interim_text:
                                print(f"⚠️ Promoting last interim to final after STT timeout: '{last_interim_text}' ({last_interim_lang})")
                                yield (last_interim_text, last_interim_lang, True)
                        else:
                            print(f"❌ STT ERROR {error_code}: {error_msg}")
                        break

            except asyncio.TimeoutError:
                if not had_final_result and last_interim_text:
                    print(f"⚠️ Continuous STT timeout - promoting last interim to final: '{last_interim_text}' ({last_interim_lang})")
                    yield (last_interim_text, last_interim_lang, True)
                print("⚠️ Continuous STT timeout - closing stream")
            finally:
                try:
                    await send_task
                except Exception:
                    pass

    except Exception as e:
        print(f"❌ Continuous STT Streaming Error: {str(e)}")

async def _send_audio_chunks_to_soniox(ws, audio_queue: asyncio.Queue, sample_rate: int = 16000):
    """
    Send audio chunks from queue to Soniox WebSocket
    Sends audio chunks to Soniox as they arrive from the live websocket stream.

    Args:
        ws: WebSocket connection
        audio_queue: Queue of audio chunks
        sample_rate: Sample rate of audio (default 16000 Hz)
    """
    try:
        chunk_count = 0
        detected_chunk_size = None
        bytes_per_sample = 2  # 16-bit PCM

        while True:
            try:
                # Get audio chunk from queue
                audio_chunk = await asyncio.wait_for(
                    audio_queue.get(), 
                    timeout=90.0  # 90 second timeout for long pauses between utterances
                )

                if audio_chunk is None:  # Sentinel value for end-of-stream
                    print(f"📝 Sending end-of-stream to STT (sent {chunk_count} chunks)")
                    await ws.send(b"")
                    break

                # Detect actual chunk size for observability
                if detected_chunk_size is None and len(audio_chunk) > 0:
                    detected_chunk_size = len(audio_chunk)
                    samples_per_chunk = detected_chunk_size // bytes_per_sample
                    chunk_duration = samples_per_chunk / sample_rate
                    print(f"📊 STT streaming: {detected_chunk_size} bytes/chunk = {samples_per_chunk} samples = {chunk_duration*1000:.1f}ms per chunk")

                # Send audio chunk
                await ws.send(audio_chunk)
                chunk_count += 1

                # Log every 5 chunks to reduce spam
                if chunk_count % 5 == 0 and chunk_count > 0:
                    total_bytes = chunk_count * detected_chunk_size if detected_chunk_size else 0
                    duration_sec = total_bytes / (sample_rate * bytes_per_sample) if total_bytes else 0
                    print(f"📤 Sent {chunk_count} audio chunks ({total_bytes} bytes, ~{duration_sec:.1f}s audio)")

            except asyncio.TimeoutError:
                print(f"⚠️ Audio queue timeout after {chunk_count} chunks - sending end-of-stream")
                await ws.send(b"")
                break

    except Exception as e:
        print(f"❌ Error sending audio chunks: {str(e)}")
 
