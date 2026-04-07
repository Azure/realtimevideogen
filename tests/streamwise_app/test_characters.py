#!/usr/bin/env python3
"""
Unit tests for StreamWise apps common utilities.
"""

import os
import sys
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("apps", "apps/streamcast"):
    from character import Character
    from character import Characters
    from character import CHARACTER_COLOR_CLASSES


def test_get_random_voice() -> None:
    character = Character("Test")
    assert character.voice is not None
    character = Character("Jane", gender="Female")
    assert character.voice is not None
    character = Character("Joe", gender="Male")
    assert character.voice is not None
    character = Character("Random", gender="Random")
    assert character.voice is not None


def test_characters() -> None:
    characters = Characters()
    characters["Charlie"] = Character("Charlie")

    with pytest.raises(ValueError):
        characters["Charlie"] = "string"

    assert len(characters) == 1
    assert "Charlie" in characters
    assert characters["Charlie"] is not None
    assert characters.get_by_index(0).name == "Charlie"
    with pytest.raises(IndexError):
        characters.get_by_index(1)

    for character in characters:
        assert character.name == "Charlie"

    assert f"{characters}" == "Characters(['Charlie'])"

    assert "Unknown" not in characters
    with pytest.raises(KeyError):
        characters["Unknown"]


def test_get_character_position() -> None:
    characters = Characters()

    assert characters.get_position("Joe") == "left"

    characters["Alice"] = Character("Alice")
    assert characters.get_position("Alice") == "center"

    characters["Bob"] = Character("Bob")
    assert characters.get_position("Alice") == "left"
    assert characters.get_position("Bob") == "right"

    characters["Charlie"] = Character("Charlie")
    assert characters.get_position("Alice") == "left"
    assert characters.get_position("Bob") == "center"
    assert characters.get_position("Charlie") == "right"

    # Unknown options
    characters["Dave"] = Character("Dave")
    assert characters.get_position("Dave") == "left"


def test_get_color() -> None:
    characters = Characters()
    characters["Alice"] = Character("Alice")
    characters["Bob"] = Character("Bob")
    characters["Charlie"] = Character("Charlie")
    characters["Dave"] = Character("Dave")
    characters["Eve"] = Character("Eve")

    colors = list(CHARACTER_COLOR_CLASSES.values())

    assert characters.get_color("Alice") == colors[0]
    assert characters.get_color("Bob") == colors[1]
    assert characters.get_color("Charlie") == colors[2]
    assert characters.get_color("Dave") == colors[3]
    assert characters.get_color("Eve") == colors[4]
