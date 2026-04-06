"""
Deepgram-backed websocket handler.

This module reuses the existing websocket bridge logic and swaps only the
provider functions at import time so the current Soniax/Azure code path stays
untouched.
"""

from app.services.deepgram_stt_service import transcribe_audio_streaming_continuous
from app.services.deepgram_tts_service import generate_speech_stream_chunked, warmup_tts_connection
from app.websocket import ws_handler as base_ws_handler

base_ws_handler.transcribe_audio_streaming_continuous = transcribe_audio_streaming_continuous
base_ws_handler.generate_speech_stream_chunked = generate_speech_stream_chunked
base_ws_handler.warmup_tts_connection = warmup_tts_connection

handle_websocket = base_ws_handler.handle_websocket
