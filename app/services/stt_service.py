async def speech_to_text(audio_bytes: bytes) -> str:
    decoded = audio_bytes.decode("utf-8", errors="ignore")
    return f"transcribed: {decoded}"
