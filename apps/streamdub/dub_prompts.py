"""Prompts for dubbing in StreamDub."""

DUB_PROMPT = """
You are a translator.
Given some original text in a source language ({input_language}), translate it into a target language ({output_language}).
Preserve the meaning, tone, and context of the original text.
Ensure the translation is natural and fluent in the target language.
Do not add any explanations or preamble, just output the translated text right away.
""".strip()  # noqa: E501

VIDEO_DUB_PROMPT = """
Person speaking
""".strip()

VIDEO_DUB_NEG_PROMPT = """
Blurry, distorted, low quality, out of focus, text, error, glitch, artifacts, deformed, disfigured, ugly, duplicate, mutated, poorly drawn, bad anatomy, worst quality, low resolution, jpeg artifacts, cropped, worst quality, low quality
""".strip()  # noqa: E501
