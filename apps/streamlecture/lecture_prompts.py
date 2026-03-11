"""Prompts for image and video generation in StreamLecture."""

# Image
IMG_PROMPT = """
A photorealistic university classroom during a lecture.
A professor stands at the front of the room behind a lectern, explaining concepts to an attentive audience of students
seated in rows of desks.
The classroom features a large whiteboard and a projection screen displaying lecture slides in the background.
Natural, evenly distributed classroom lighting with no dramatic effects.
The professor wears professional attire and uses natural hand gestures while speaking.
Scene captured from a slightly elevated, front-facing perspective, showing the professor and several students in the foreground.
Realistic academic environment with notebooks, laptops, and pens visible on desks.
""".strip()  # noqa: E501

IMG_NEG_PROMPT = """camera move, fancy lights, cartoon, illustration, anime, 3D render, CGI, digital art, painting,
smooth plastic skin, waxy texture, low detail, overexposed, blurred faces, uncanny valley"""

# Video
VIDEO_PROMPT = """
Professor explaining.
""".strip()

VIDEO_NEG_PROMPT = """
camera movement, shaky cam, zoom, pan, tilt, flashy lights, jitter, dolly shot, handheld,
color shift, lighting flicker, exposure change, hue variation
""".strip()
