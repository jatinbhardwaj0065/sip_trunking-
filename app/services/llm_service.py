"""
Large Language Model processing using OpenAI GPT-4o-mini
Supports both batch and streaming responses
Multilingual support with automatic translation
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .language_utils import translate_text, get_language_name

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
        # Translate input to English if needed
        english_text = text
        if language_code != "en":
            lang_name = get_language_name(language_code)
            print(f"🔄 Translating from {lang_name} to English...")
            english_text = await translate_text(text, language_code, "en")

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Default system prompt for voice calls
            messages.append({
                "role": "system", 
                "content": "You are a helpful AI assistant on a phone call. Keep responses conversational, concise, and under 100 words. Be friendly and natural."
            })

        # Add user message (in English)
        messages.append({"role": "user", "content": english_text})

        # Get response from OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
              # Keep responses short for voice calls
        )

        ai_response = response.choices[0].message.content
        print(f"🤖 AI Response (English): {ai_response}")

        # Translate response back to user's language if needed
        if language_code != "en":
            lang_name = get_language_name(language_code)
            print(f"🔄 Translating response back to {lang_name}...")
            ai_response = await translate_text(ai_response, "en", language_code)
            print(f"🤖 AI Response ({lang_name}): {ai_response}")

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
        # Translate input to English if needed
        english_text = text
        if language_code != "en":
            lang_name = get_language_name(language_code)
            print(f"🔄 Translating from {lang_name} to English...")
            english_text = await translate_text(text, language_code, "en")

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Default system prompt for voice calls
            messages.append({
                "role": "system", 
                "content": "You are a helpful AI assistant on a phone call. Keep responses conversational, concise, and under 100 words. Be friendly and natural."
            })

        # Add user message (in English)
        messages.append({"role": "user", "content": english_text})

        # Get streaming response from OpenAI
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            stream=True  # Enable streaming
        )

        # Collect all tokens for translation if needed
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                print(f"🤖 Token: {token}", end="", flush=True)
                yield token  # Yield English tokens for real-time display

        print()  # New line after streaming complete

        # Translate response back to user's language if needed
        if language_code != "en":
            lang_name = get_language_name(language_code)
            print(f"🔄 Translating response back to {lang_name}...")
            translated_response = await translate_text(full_response, "en", language_code)
            print(f"🤖 Final response ({lang_name}): {translated_response}")
            # Replace the already-yielded English tokens with translated version
            # by yielding a special marker followed by the full translated text
            yield f"\n__TRANSLATED__{translated_response}__END_TRANSLATED__"

    except Exception as e:
        print(f"LLM Streaming Error: {str(e)}")
        yield "I'm sorry, I'm having trouble processing that right now. Could you try again?"