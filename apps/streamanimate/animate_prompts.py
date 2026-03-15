"""Prompts for image and video generation in StreamAnimate."""

# Image generation
IMG_PROMPT_BASE = """
Highly photorealistic, ultra-detailed, cinematic photograph.
Sharp focus, natural lighting, high dynamic range.
""".strip()

IMG_PROMPT = """
A vivid, visually striking scene ready to be brought to life through animation.
Rich colors, expressive subject, clear composition.
""".strip()
IMG_PROMPT += " " + IMG_PROMPT_BASE

IMG_NEG_PROMPT = """
blurry, low quality, overexposed, underexposed, noisy, grainy, out of focus,
cartoon, anime, sketch, low resolution, dull, flat lighting
""".strip()

# Video / animation generation
VIDEO_PROMPT = """
Smooth, natural animation with fluid motion.
Subtle organic movement. Stable framing, no camera shake.
Consistent lighting throughout. Lifelike and expressive.
""".strip()

VIDEO_NEG_PROMPT = """
camera movement, shaky cam, zoom, pan, tilt, jitter, flicker,
color shift, lighting change, abrupt cut, freeze frame, stutter
""".strip()
