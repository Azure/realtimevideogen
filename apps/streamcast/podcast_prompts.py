"""Prompts for image and video generation in StreamCast."""

IMG_PROMPT_BASE = """
Medium shot with a balanced composition where each person occupies less than 1/4 of the shot.
Can clearly see the faces of all characters so one can zoom into their faces.
Highly photorealistic photograph.
Ultra-detailed, cinematic, true-to-life photograph.
Captured with a Canon EOS R5, 35mm f/2.8 lens, shallow depth of field, realistic skin texture,
natural facial expressions, detailed hair, and lifelike clothing fabric.
""".strip()
IMG_PROMPT = """
A photorealistic podcast setup featuring a woman and a man sitting across each other at a wooden table in a
professional recording studio.
The studio has red acoustic panels on the walls, warm lighting, and a large off screen in the background.
Both wear headphones and speak into high-quality podcast microphones mounted on adjustable arms.
Scene captured from a slightly elevated front-facing perspective, showing their upper bodies and expressive
gestures as they engage in conversation.
Table equipped with coffee mugs, water bottles, and recording equipment.
""".strip()
IMG_PROMPT += " " + IMG_PROMPT_BASE
IMG_NEG_PROMPT = """camera move, fancy lights, cartoon, illustration, anime, 3D render, CGI, digital art, painting,
smooth plastic skin, waxy texture, low detail, overexposed, blurred faces, uncanny valley"""

IMG_ZOOM_PROMPT = """
Zoom of the main person in the image.
Just the main person and no other people are visible.
Crop into a medium close-up of a single speaker at eye level, framed from chest up with soft background blur,
positioned slightly to one side of the frame to preserve conversational eyeline.
""".strip()
# A high-resolution DSLR photo, natural skin texture, lifelike detail,
# cinematic portrait lighting, shallow depth of field.
# Realistic human appearance.

# Example: The "woman" on the "left" is speaking while the rest listens.
VIDEO_PROMPT = """
The %s on the %s is speaking while the rest listens.
The framing is static and symmetrical.
The background remains fixed.
The camera does not move and is completely still.
The shot is locked off, tripod-mounted, with no movement.
The lighting is consistent and neutral.
The colors remain stable across all frames.
""".strip()
VIDEO_NEG_PROMPT = """
camera movement, shaky cam, zoom, pan, tilt, flashy lights, jitter, dolly shot, handheld,
color shift, lighting flicker, exposure change, hue variation
""".strip()
