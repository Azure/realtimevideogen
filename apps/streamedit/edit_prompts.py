"""Prompts for editing videos in StreamEdit."""

EDIT_PROMPT_SYSTEM = """
You are an expert video editor. Given a video segment, apply the specified edits while preserving
the original subject matter, framing, and audio content.
Maintain consistent timing and motion across the edit.
""".strip()

EDIT_PROMPT_DEFAULT = """
Apply subtle visual enhancements: improve color grading, enhance sharpness, and smooth any abrupt transitions.
Keep the overall look and feel consistent with the original footage.
""".strip()


def build_edit_prompt(edit_instructions: str = "") -> str:
    """Build the full edit prompt from optional user instructions."""
    instructions = edit_instructions.strip() if edit_instructions else EDIT_PROMPT_DEFAULT
    return f"{EDIT_PROMPT_SYSTEM}\n\nEdits to apply:\n{instructions}"


# Default prompt used when no instructions are supplied
EDIT_PROMPT = build_edit_prompt()
