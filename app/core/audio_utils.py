from pathlib import Path
from datetime import datetime
import wave


def save_wav_file(
    audio_bytes: bytes,
    output_dir: str = "recordings",
    sample_rate: int = 8000 ,
    channels: int = 1,
    sample_width: int = 2,
) -> str:
    """
    Save raw PCM audio bytes as a WAV file.

    Args:
        audio_bytes: Raw PCM audio data.
        output_dir: Directory where WAV file will be saved.
        sample_rate: Samples per second.
        channels: Number of audio channels.
        sample_width: Bytes per sample (2 = 16-bit audio).

    Returns:
        Path to the saved WAV file as string.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = output_path / filename

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return str(file_path)
