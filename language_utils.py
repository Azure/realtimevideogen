"""
Language-code utilities shared across all apps and wrappers.

Internal standard: ISO 639-1 codes (e.g. "en", "es", "fr") as returned by
Whisper.  Engine-specific codes are mapped to/from ISO 639-1 via the
dictionaries below.  Human-readable language names are resolved through the
``langcodes`` library so the mapping stays in sync with the IANA language
subtag registry.
"""

import langcodes

# Maps Whisper / ISO 639-1 language codes to Kokoro TTS single-letter codes.
# Kokoro codes:
#   'a' => American English (en-US)   'b' => British English (en-GB)
#   'e' => Spanish (es)               'f' => French (fr)
#   'g' => German (de)                'h' => Hindi (hi)
#   'i' => Italian (it)               'j' => Japanese (ja)
#   'k' => Korean (ko)                'p' => Brazilian Portuguese (pt-BR)
#   'r' => Russian (ru)               'z' => Mandarin Chinese (zh)
WHISPER_TO_KOKORO: dict[str, str] = {
    "en": "a",   # American English (default; use "b" for British)
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

# Reverse mapping: Kokoro single-letter codes → BCP 47 / ISO 639-1 tags.
KOKORO_TO_ISO: dict[str, str] = {
    "a": "en-US",
    "b": "en-GB",
    "e": "es",
    "f": "fr",
    "g": "de",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "k": "ko",
    "p": "pt-BR",
    "r": "ru",
    "z": "zh",
}


def to_language(lang_code: str) -> str:
    """Return a human-readable language name for an ISO 639-1 or Kokoro code.

    Kokoro single-letter codes (e.g. ``"a"``, ``"e"``) are first translated to
    their BCP 47 equivalents via :data:`KOKORO_TO_ISO`; ISO codes are passed
    directly to ``langcodes``.  Returns a descriptive ``"Unknown language"``
    string for codes that ``langcodes`` cannot resolve, and falls back to the
    original *lang_code* if an unexpected exception occurs.
    """
    iso = KOKORO_TO_ISO.get(lang_code.lower(), lang_code)
    try:
        return langcodes.get(iso).display_name()
    except Exception:
        return lang_code
