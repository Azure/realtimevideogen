"""
Prompts for StreamChat app.
"""

IMG_PROMPT = """
Front shot of a person high resolution.
""".strip()

IMG_NEG_PROMPT = """camera move, fancy lights, cartoon, illustration, anime, 3D render, CGI, digital art, painting,
smooth plastic skin, waxy texture, low detail, overexposed, blurred faces, uncanny valley"""

VIDEO_PROMPT = """
Character in the image is speaking.
""".strip()
VIDEO_NEG_PROMPT = """
camera movement, shaky cam, zoom, pan, tilt, flashy lights, jitter, dolly shot, handheld,
color shift, lighting flicker, exposure change, hue variation
""".strip()

CHAT_PROMPT = """
You are a helpful assistant.
Do not be too verbose when replying.
""".strip()
