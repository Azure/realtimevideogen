"""Prompts for dubbing in StreamDub."""

DUB_PROMPT = """
You are a translator.
Given some original text in a source language ({input_language}), translate it into a target language ({output_language}).
Preserve the meaning, tone, and context of the original text.
Ensure the translation is natural and fluent in the target language.
Do not add any explanations or preamble, just output the translated text right away.
""".strip()  # noqa: E501

SPEAKER_GENDER_PROMPT = """
You are a speech analyst.
Given a transcript, determine the most likely gender of the main speaker.
Reply with exactly one word: either "male" or "female".
Do not include any explanation, punctuation, or additional text.
""".strip()
# Note: binary male/female classification is used solely for TTS voice matching.
# This is a simplification; mixed-gender or indeterminate scenes default to the female voice.

VIDEO_DUB_PROMPT = """
Person speaking
""".strip()

VIDEO_DUB_NEG_PROMPT = """
Blurry, distorted, low quality, out of focus, text, error, glitch, artifacts, deformed, disfigured, ugly, duplicate, mutated, poorly drawn, bad anatomy, worst quality, low resolution, jpeg artifacts, cropped, worst quality, low quality
""".strip()  # noqa: E501
