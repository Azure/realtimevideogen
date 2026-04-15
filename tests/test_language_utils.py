"""
Unit tests for language_utils.
"""

from language_utils import WHISPER_TO_KOKORO, KOKORO_TO_ISO, to_language


def test_whisper_to_kokoro() -> None:
    """WHISPER_TO_KOKORO maps Whisper ISO 639-1 codes to Kokoro single-letter codes."""
    assert WHISPER_TO_KOKORO["en"] == "a"
    assert WHISPER_TO_KOKORO["es"] == "e"
    assert WHISPER_TO_KOKORO["fr"] == "f"
    assert WHISPER_TO_KOKORO["de"] == "g"
    assert WHISPER_TO_KOKORO["hi"] == "h"
    assert WHISPER_TO_KOKORO["it"] == "i"
    assert WHISPER_TO_KOKORO["ja"] == "j"
    assert WHISPER_TO_KOKORO["ko"] == "k"
    assert WHISPER_TO_KOKORO["pt"] == "p"
    assert WHISPER_TO_KOKORO["ru"] == "r"
    assert WHISPER_TO_KOKORO["zh"] == "z"


def test_kokoro_to_iso() -> None:
    """KOKORO_TO_ISO maps Kokoro single-letter codes to BCP 47 tags."""
    assert KOKORO_TO_ISO["a"] == "en-US"
    assert KOKORO_TO_ISO["b"] == "en-GB"
    assert KOKORO_TO_ISO["e"] == "es"
    assert KOKORO_TO_ISO["f"] == "fr"
    assert KOKORO_TO_ISO["g"] == "de"
    assert KOKORO_TO_ISO["h"] == "hi"
    assert KOKORO_TO_ISO["i"] == "it"
    assert KOKORO_TO_ISO["j"] == "ja"
    assert KOKORO_TO_ISO["k"] == "ko"
    assert KOKORO_TO_ISO["p"] == "pt-BR"
    assert KOKORO_TO_ISO["r"] == "ru"
    assert KOKORO_TO_ISO["z"] == "zh"


def test_whisper_kokoro_roundtrip() -> None:
    """Every ISO code in WHISPER_TO_KOKORO has a reverse entry in KOKORO_TO_ISO."""
    for iso_code, kokoro_code in WHISPER_TO_KOKORO.items():
        assert kokoro_code in KOKORO_TO_ISO, f"Kokoro code '{kokoro_code}' (from '{iso_code}') missing in KOKORO_TO_ISO"


def test_kokoro_iso_roundtrip() -> None:
    """Every Kokoro code in KOKORO_TO_ISO except 'b' (British English) is reachable from WHISPER_TO_KOKORO.

    'b' (British English) is intentionally absent from WHISPER_TO_KOKORO because
    Whisper uses 'en' for all English variants.
    """
    whisper_to_kokoro_values = set(WHISPER_TO_KOKORO.values())
    kokoro_only_codes = {"b"}  # British English: no Whisper equivalent
    for kokoro_code in KOKORO_TO_ISO:
        if kokoro_code in kokoro_only_codes:
            continue
        assert kokoro_code in whisper_to_kokoro_values, (
            f"Kokoro code '{kokoro_code}' in KOKORO_TO_ISO has no entry in WHISPER_TO_KOKORO"
        )


def test_to_language_kokoro_codes() -> None:
    """to_language converts Kokoro single-letter codes to human-readable names."""
    assert to_language("a") == "English (United States)"
    assert to_language("b") == "English (United Kingdom)"
    assert to_language("e") == "Spanish"
    assert to_language("f") == "French"
    assert to_language("g") == "German"
    assert to_language("h") == "Hindi"
    assert to_language("i") == "Italian"
    assert to_language("j") == "Japanese"
    assert to_language("k") == "Korean"
    assert to_language("p") == "Portuguese (Brazil)"
    assert to_language("r") == "Russian"
    assert to_language("z") == "Chinese"


def test_to_language_iso_codes() -> None:
    """to_language also accepts ISO 639-1 codes directly."""
    assert to_language("en") == "English"
    assert to_language("es") == "Spanish"
    assert to_language("fr") == "French"
    assert to_language("zh") == "Chinese"


def test_to_language_case_insensitive() -> None:
    """to_language is case-insensitive for Kokoro codes."""
    assert to_language("E") == to_language("e")
    assert to_language("F") == to_language("f")
    assert to_language("A") == to_language("a")


def test_to_language_unknown_code() -> None:
    """to_language returns a descriptive string for unrecognised codes (via langcodes)."""
    assert to_language("xyz") == "Unknown language [xyz]"
