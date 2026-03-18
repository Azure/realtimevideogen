"""
Language-code utilities shared across all apps.

Whisper returns ISO 639-1 codes (e.g. "en", "es", "fr").
Kokoro TTS uses its own single-letter codes (e.g. "a", "e", "f").
WHISPER_TO_KOKORO maps between the two so callers do not need to hard-code
the relationship in multiple places.
"""

# Maps Whisper ISO 639-1 language codes to Kokoro TTS single-letter codes.
# Kokoro codes:
#   'a' => American English   'b' => British English
#   'e' => Spanish (es)       'f' => French (fr)
#   'g' => German (de)        'h' => Hindi (hi)
#   'i' => Italian (it)       'j' => Japanese (ja)
#   'k' => Korean (ko)        'p' => Brazilian Portuguese (pt)
#   'r' => Russian (ru)       'z' => Mandarin Chinese (zh)
WHISPER_TO_KOKORO: dict[str, str] = {
    "en": "a",  # American English (default; use "b" for British)
    "es": "e",
    "fr": "f",
    "de": "g",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "ko": "k",
    "pt": "p",
    "ru": "r",
    "zh": "z",
}


def to_language(lang_code: str) -> str:
    """Convert a Kokoro TTS single-letter language code to a human-readable language name."""
    lang_map: dict[str, str] = {
        "a": "American English",
        "b": "British English",
        "c": "Chinese",
        "e": "Spanish",
        "f": "French",
        "g": "German",
        "h": "Hindi",
        "i": "Italian",
        "j": "Japanese",
        "k": "Korean",
        "p": "Portuguese",
        "r": "Russian",
        "z": "Chinese",
    }
    return lang_map.get(lang_code.lower(), "American English")
