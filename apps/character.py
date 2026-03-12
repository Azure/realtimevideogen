"""Character information for StreamWise."""

import logging
import random

from typing import Dict
from typing import Optional
from typing import Any

from PIL import Image

# https://github.com/hexgrad/kokoro/blob/main/kokoro.js/src/voices.js
# https://github.com/hexgrad/kokoro/blob/main/kokoro.js/README.md
KOKORO_VOICES = {
    "female": [
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky"
    ],
    "male": [
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa"
    ]
}

VIBEVOICE_VOICES = {
    "Female": [
        "af_bella",
        "af_heart",
        "af_kore",
        "af_nicole",
        "woman_000",
    ],
    "Male": [
        "am_adam",
        "am_fenrir",
        "am_michael",
        "am_puck",
        "girish_001",
        "girish_002",
        "ricardo_001",
        "ricardo_002",
    ]
}


# Comes from javascript/bootstrap:
# const colorClasses = ["text-primary", "text-success", "text-danger", "text-warning", "text-info", "text-secondary"];
CHARACTER_COLOR_CLASSES = {
    "text-primary": "#0d6efd",  # blue
    "text-success": "#198754",  # green
    "text-danger": "#dc3545",  # red
    "text-warning": "#ffc107",  # yellow
    "text-info": "#0dcaf0",  # cyan
    "text-secondary": "#6c757d",  # gray
}


class Character:
    """Character information."""
    def __init__(
        self,
        name: str,
        gender: str = "Unknown",
        description: Optional[str] = None,
        speech_speed: float = 1.1,
    ) -> None:
        self.name = name
        self.gender = gender
        self.voice = self.get_random_voice()
        self.speech_speed = speech_speed
        self.image: Optional[Image.Image] = None
        self.description = description

    def get_random_voice(self) -> str:
        """Get a random voice based on the gender."""
        gender = self.gender.lower()
        gender_voices = KOKORO_VOICES.get(gender, [])
        if not gender_voices:
            logging.warning(f"Unknown gender: {self.gender}.")
            return "af_heart"
        return random.choice(gender_voices)


class Characters:
    """Manage multiple characters."""
    def __init__(self) -> None:
        self.characters: Dict[str, Character] = {}

    def __len__(self) -> int:
        return len(self.characters)

    def __contains__(self, name: str) -> bool:
        return name in self.characters

    def __getitem__(self, name: str) -> Character:
        if name not in self.characters:
            raise KeyError(f"Character '{name}' not found.")
        return self.characters[name]

    def __setitem__(self, name: str, character: Character) -> None:
        if not isinstance(character, Character):
            raise ValueError("character must be an instance of Character")
        self.characters[name] = character

    def __iter__(self) -> Any:
        return iter(self.characters.values())

    def __repr__(self) -> str:
        return f"Characters({list(self.characters.keys())})"

    def get_index(self, name: str) -> int:
        """Get index of character by name."""
        if name not in self.characters:
            return 0
        return list(self.characters).index(name)

    def get_by_index(self, index: int) -> Character:
        """Get character by index."""
        if index < 0 or index >= len(self.characters):
            raise IndexError("Character index out of range")
        character = list(self.characters.values())[index]
        return character

    def get_position(
        self,
        name: str,
    ) -> str:
        """Get character position in the image: left, center, right."""
        index = self.get_index(name)
        num_characters = len(self.characters)

        if num_characters == 1:
            return "center"
        if num_characters == 2:
            if index == 0:
                return "left"
            elif index == 1:
                return "right"
        if num_characters == 3:
            if index == 0:
                return "left"
            elif index == 1:
                return "center"
            elif index == 2:
                return "right"
        return "left"

    def get_color(self, name: str) -> str:
        index = self.get_index(name) % len(CHARACTER_COLOR_CLASSES)
        color = CHARACTER_COLOR_CLASSES[list(CHARACTER_COLOR_CLASSES.keys())[index]]
        return color
