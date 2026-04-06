"""
Large Language Model processing using OpenAI GPT-4o-mini
Supports both batch and streaming responses
Multilingual support with direct in-language responses
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .language_utils import get_language_name

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
VOICE_REPLY_MAX_TOKENS = int(os.getenv("VOICE_REPLY_MAX_TOKENS", "60"))  # Reduced from 80 for faster streaming

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _build_messages(text: str, language_code: str = "en", system_prompt: str = None):
    target_language = get_language_name(language_code)
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant on a live phone call. "
                    f"Reply in {target_language}. Keep the answer conversational, natural, and brief. "
                    "Prefer 1 short sentence or 2 very short sentences. Avoid lists unless the caller asks for them."
                ),
            }
        )

    messages.append({"role": "user", "content": text})
    return messages


async def warmup_llm_connection() -> None:
    """Prime the OpenAI connection and API endpoint before first call."""
    if not OPENAI_API_KEY:
        return

    try:
        # Send a minimal request to warm up the API and establish connection
        await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "ok"}],
            temperature=0.0,
            max_tokens=5,
        )
        print("🔥 LLM warmup complete")
    except Exception as exc:
        # Warmup should never block the live call
        print(f"⚠️ LLM warmup failed: {exc}")


async def process_text(text: str, language_code: str = "en", system_prompt: str = None) -> str:
    """
    Process text through OpenAI GPT-4o-mini
    Supports multilingual input with automatic translation
    Responses are returned in the user's detected language

    Args:
        text: User input text
        language_code: ISO 639-1 language code (e.g., 'hi', 'en')
        system_prompt: Optional system prompt for context

    Returns:
        AI response text (in user's language)
    """
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set in environment")

    if not text.strip():
        return "I didn't hear anything. Could you please repeat that?"

    try:
        messages = _build_messages(text, language_code=language_code, system_prompt=system_prompt)

        # Get response from OpenAI
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=VOICE_REPLY_MAX_TOKENS,
        )

        ai_response = (response.choices[0].message.content or "").strip()
        print(f"🤖 AI Response ({language_code}): {ai_response}")
        return ai_response

    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return "I'm sorry, I'm having trouble processing that right now. Could you try again?"

async def process_text_streaming(text: str, language_code: str = "en", system_prompt: str = None):
    """
    Process text through OpenAI GPT-4o-mini with streaming
    Yields tokens as they arrive for real-time processing
    Supports multilingual input with automatic translation
    Responses are returned in the user's detected language

    Args:
        text: User input text
        language_code: ISO 639-1 language code (e.g., 'hi', 'en')
        system_prompt: Optional system prompt for context

    Yields:
        Individual tokens from the AI response (in user's language)
    """
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set in environment")

    if not text.strip():
        yield "I didn't hear anything. Could you please repeat that?"
        return

    try:
        messages = _build_messages(text, language_code=language_code, system_prompt=system_prompt)

        # Get streaming response from OpenAI
        stream = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=VOICE_REPLY_MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(f"🤖 Token: {token}", end="", flush=True)
                yield token

        print()

    except Exception as e:
        print(f"LLM Streaming Error: {str(e)}")
        yield "I'm sorry, I'm having trouble processing that right now. Could you try again?"
