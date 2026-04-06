"""
Optimized Language Processing for Real-Time Streaming
- Session-based detection
- Translation caching
- Chunk buffering
- Async optimized
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# LANGUAGE + VOICE CONFIG
# -------------------------------

LANGUAGE_SCRIPTS = {
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "gu": (0x0A80, 0x0AFF),
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "en": (0x0000, 0x007F),
}

AZURE_VOICE_MAP = {
    "en": "en-IN-ArjunNeural",
    "hi": "hi-IN-MadhurNeural",
    "ta": "ta-IN-SarathNeural",
    "te": "te-IN-MohanNeural",
    "kn": "kn-IN-GaranNeural",
    "ml": "ml-IN-MidhunNeural",
    "bn": "bn-IN-BashkarNeural",
    "mr": "mr-IN-AarohNeural",
    "gu": "gu-IN-DhwaniNeural",
}

# -------------------------------
# SESSION CLASS (CORE OPTIMIZATION)
# -------------------------------

class LanguageSession:
    def __init__(self):
        self.detected_lang = None
        self.voice = None
        self.translation_cache = {}

# -------------------------------
# STREAM BUFFER
# -------------------------------

class StreamBuffer:
    def __init__(self, threshold=40):
        self.buffer = ""
        self.threshold = threshold

    def add(self, chunk: str):
        self.buffer += " " + chunk

        if len(self.buffer) >= self.threshold:
            data = self.buffer.strip()
            self.buffer = ""
            return data

        return None

# -------------------------------
# LANGUAGE DETECTION (RUN ONCE)
# -------------------------------

def detect_language(text: str) -> str:
    text = text.strip()

    # phonetic English detection
    english_markers = ["మై", "నేమ్", "హలో", "బై", "థాంక్"]
    if any(marker in text for marker in english_markers):
        return "en"

    counts = {}
    for lang, (start, end) in LANGUAGE_SCRIPTS.items():
        count = sum(1 for c in text if start <= ord(c) <= end)
        if count > 0:
            counts[lang] = count

    if counts:
        return max(counts, key=counts.get)

    return "en"

def detect_once(session: LanguageSession, text: str):
    if session.detected_lang is None:
        session.detected_lang = detect_language(text)
        session.voice = AZURE_VOICE_MAP.get(session.detected_lang, "en-IN-ArjunNeural")

    return session.detected_lang

# -------------------------------
# TRANSLATION (CACHED + ASYNC)
# -------------------------------

async def translate_text(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Translate from {src} to {tgt}. Only output translated text."
            },
            {"role": "user", "content": text}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()

async def translate_cached(session, text, src, tgt):
    key = f"{src}:{tgt}:{text}"

    if key in session.translation_cache:
        return session.translation_cache[key]

    translated = await translate_text(text, src, tgt)
    session.translation_cache[key] = translated

    return translated

# -------------------------------
# MAIN PROCESS FUNCTION
# -------------------------------

async def process_chunk(session, buffer, chunk, llm_func):
    """
    llm_func: async function that takes text and returns response
    """

    # 1. Detect language once
    lang = detect_once(session, chunk)

    # 2. Buffer chunks
    full_text = buffer.add(chunk)
    if not full_text:
        return None

    # 3. Translate to English (parallelizable)
    if lang != "en":
        full_text = await translate_cached(session, full_text, lang, "en")

    # 4. Call LLM
    llm_response = await llm_func(full_text)

    # 5. Translate back
    if lang != "en":
        llm_response = await translate_cached(session, llm_response, "en", lang)

    return {
        "text": llm_response,
        "voice": session.voice,
        "lang": lang
    }

# -------------------------------
# HELPER FUNCTIONS (backward compatibility)
# -------------------------------

def detect_language_from_script(text: str) -> str:
    """Backward compatible wrapper for detect_language"""
    return detect_language(text)

def get_azure_voice_for_language(language_code: str) -> str:
    """Get Azure voice for a given language code"""
    return AZURE_VOICE_MAP.get(language_code, "en-IN-ArjunNeural")

def get_language_name(language_code: str) -> str:
    """Get display name for a language code"""
    language_names = {
        "en": "English",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "bn": "Bengali",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
    }
    return language_names.get(language_code, language_code.upper())

# -------------------------------
# OPTIONAL: WARMUP (reduce cold start)
# -------------------------------

async def warmup():
    try:
        await translate_text("hello", "en", "hi")
    except:
        pass