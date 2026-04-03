# language_utils.py
"""
Language detection and translation utilities
Supports automatic language detection and translation
"""

import unicodedata
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Language to script mapping
LANGUAGE_SCRIPTS = {
    "hi": ("Devanagari", 0x0900, 0x097F),      # Hindi
    "mr": ("Devanagari", 0x0900, 0x097F),      # Marathi
    "sa": ("Devanagari", 0x0900, 0x097F),      # Sanskrit
    "gu": ("Gujarati", 0x0A80, 0x0AFF),        # Gujarati
    "bn": ("Bengali", 0x0980, 0x09FF),         # Bengali
    "pa": ("Gurmukhi", 0x0A00, 0x0A7F),        # Punjabi
    "ta": ("Tamil", 0x0B80, 0x0BFF),           # Tamil
    "te": ("Telugu", 0x0C00, 0x0C7F),          # Telugu
    "kn": ("Kannada", 0x0C80, 0x0CFF),         # Kannada
    "ml": ("Malayalam", 0x0D00, 0x0D7F),       # Malayalam
    "or": ("Odia", 0x0B00, 0x0B7F),            # Odia
    "ar": ("Arabic", 0x0600, 0x06FF),          # Arabic
    "he": ("Hebrew", 0x0590, 0x05FF),          # Hebrew
    "zh": ("Han", 0x4E00, 0x9FFF),             # Chinese
    "ja": ("Japanese", 0x3040, 0x309F),        # Japanese (Hiragana)
    "ko": ("Hangul", 0xAC00, 0xD7AF),          # Korean
    "ru": ("Cyrillic", 0x0400, 0x04FF),        # Russian
    "el": ("Greek", 0x0370, 0x03FF),           # Greek
    "th": ("Thai", 0x0E00, 0x0E7F),            # Thai
    "en": ("Latin", 0x0000, 0x007F),           # English
}

# Voice mapping for Azure TTS (language code -> voice names)
# Using language-appropriate voices for proper TTS synthesis
AZURE_VOICE_MAP = {
    "en": ["en-IN-ArjunNeural"],  # English - Indian English voice
    "hi": ["hi-IN-MadhurNeural"],  # Hindi
    "es": ["es-ES-AlvaroNeural"],  # Spanish
    "fr": ["fr-FR-DeniseNeural"],  # French
    "de": ["de-DE-ConradNeural"],  # German
    "it": ["it-IT-DiegoNeural"],  # Italian
    "ja": ["ja-JP-KeitaNeural"],  # Japanese
    "zh": ["zh-CN-XiaoxiaoNeural"],  # Chinese (Simplified)
    "ko": ["ko-KR-InJoonNeural"],  # Korean
    "pt": ["pt-BR-AntonioNeural"],  # Portuguese (Brazil)
    "ru": ["ru-RU-DmitryNeural"],  # Russian
    "ar": ["ar-SA-HamedNeural"],  # Arabic
    "bn": ["bn-IN-BashkarNeural"],  # Bengali
    "ta": ["ta-IN-SarathNeural"],  # Tamil
    "te": ["te-IN-MohanNeural"],  # Telugu
    "kn": ["kn-IN-GaranNeural"],  # Kannada
    "ml": ["ml-IN-MidhunNeural"],  # Malayalam
    "gu": ["gu-IN-DhwaniNeural"],  # Gujarati
    "mr": ["mr-IN-AarohNeural"],  # Marathi
}

def detect_language_from_script(text: str) -> str:
    """
    Detect language based on script/characters used.
    Special override: If Telugu (or any Indian script) contains ENGLISH phonetic syllables,
    force the language to English.
    """

    if not text:
        return "unknown"

    # 1) Strip punctuation
    punctuation = set('.,!?;:()[]{}"“”\'`|&/\\-')
    normalized = ''.join(c for c in text if c not in punctuation).strip()

    # ----------------------------------------------------------------------------------
    # 2) ENGLISH-PHONETIC DETECTION (EXTENDED FORCE-ENGLISH MODE)
    # ----------------------------------------------------------------------------------

    # A large list of English-like syllables commonly rendered phonetically in Indian scripts
    english_syllables = [
        # core english words
        "మై", "నేమ్", "ఇస్", "యువర్", "యోర్", "హౌ", "హౌ", "ఆర్", "యు", "యూ",
        "హాయ్", "హలో", "బై", "థాంక్", "యూ", "సారీ",

        # alphabet spellings
        "టి", "హెచ్", "ఐ", "ఎల్", "ఏ", "కే", "బి", "సి", "డి", "ఎఫ్", "జి", "జె",
        "ఎం", "ఎన్", "ఓ", "పి", "క్యూ", "ఆర్", "ఎస్", "టి", "యు", "వీ", "డబ్ల్యూ",
        "ఎక్స్", "వై", "జెడ్",

        # english borrow words often spelled in indian scripts
        "ప్లీజ్", "ఒకే", "ఓకే", "సారీ", "కంఫర్మ్", "అడ్రస్", "టైమ్", "డేట్",
        "నెంబర్", "నంబర్", "మొబైల్", "ఫోన్"
    ]

    # If **more than 30%** of tokens look like English → FORCE ENGLISH
    tokens = normalized.split()
    english_like_count = 0

    for token in tokens:
        for syl in english_syllables:
            if syl in token:
                english_like_count += 1
                break

    if tokens and (english_like_count / len(tokens)) >= 0.30:
        print(f"🚨 Phonetic-English Override: {english_like_count}/{len(tokens)} syllables detected")
        print("🌍 Correcting final detected language to English")
        return "en"

    # ----------------------------------------------------------------------------------
    # 3) NORMAL SCRIPT DETECTION (fallback)
    # ----------------------------------------------------------------------------------
    script_counts = {}

    for lang_code, (script_name, start, end) in LANGUAGE_SCRIPTS.items():
        count = sum(1 for c in normalized if start <= ord(c) <= end)
        if count > 0:
            script_counts[lang_code] = count

    if script_counts:
        detected = max(script_counts, key=script_counts.get)
        print(f"🌍 Detected language by script = {detected}")
        return detected

    # default
    return "en"


def get_azure_voice_for_language(language_code: str) -> str:
    """
    Get appropriate Azure TTS voice for language

    Args:
        language_code: ISO 639-1 language code (e.g., 'hi', 'en')

    Returns:
        Azure voice name
    """
    voices = AZURE_VOICE_MAP.get(language_code, AZURE_VOICE_MAP["en"])
    voice = voices[0]  # Use first voice by default
    print(f"🎤 Selected voice for {language_code}: {voice}")
    return voice

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using OpenAI

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translated text
    """
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set in environment")

    if source_lang == target_lang:
        return text

    try:
        # Get language names
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the text from {source_name} to {target_name}. Respond with only the translated text, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.0,

        )

        translated = response.choices[0].message.content.strip()
        print(f"🔄 Translated ({source_name}→{target_name}): '{text[:50]}...' → '{translated[:50]}...'")
        return translated

    except Exception as e:
        print(f"❌ Translation error: {str(e)}")
        return text

def get_language_name(lang_code: str) -> str:
    """Get human-readable language name"""
    language_names = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "gu": "Gujarati",
        "mr": "Marathi",
    }
    return language_names.get(lang_code, lang_code.upper())
 