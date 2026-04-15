"""Prompts for image and video generation in StreamPersona."""

IMG_PROMPT = """
Front shot of a man with a white background, smiling, professional lighting, high resolution, photorealistic
""".strip()
IMG_NEG_PROMPT = "blurry, lowres, deformed, disfigured, bad anatomy"

VIDEO_PROMPT = """
Person speaking.
""".strip()
VIDEO_NEG_PROMPT = """
camera movement, shaky cam, zoom, pan, tilt, flashy lights, jitter, dolly shot, handheld,
color shift, lighting flicker, exposure change, hue variation
""".strip()
