"""Prompts for text, image, and video generation in StreamShort."""

DESCRIPTION_PROMPT = """
You are a precise video describer. Provide a concise, vivid description
of the scene shown in the frame. Mention key subjects, actions, setting,
and any notable lighting or mood. Keep it to 2-4 sentences.
Do not append any preamble and just output the description right away.

If I provide multiple frames, describe each one separately in order with the following format:\n
{"frame_num": 0, "description": "A formula one car helmeted and focused..."}
{"frame_num": 1, "description": "A race care driver is shown in first person..."}
{"frame_num": 2, "description": "The race car crosses the finish line..."}
...
"""

HIGHLIGHT_PROMPT = """
You select the most engaging scenes for a short highlight reel.
Return ONLY a JSON list of scene indexes (integers) in the order they should appear.
Do not add any preamble or explanation, just output the plain JSON list, for example:
[0, 3, 5, 2]

The total duration of the output should be under {total_length} seconds.
Output a maximum of {max_scenes} scenes.

Prefer visually dynamic, emotionally impactful, or plot-crucial scenes.
Avoid redundancy and keep pacing brisk.

Here are the scenes:
"""
