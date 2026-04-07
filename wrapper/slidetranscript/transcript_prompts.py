"""
Prompts for slide transcript generation.
"""

# flake8: noqa: E501
SYSTEM_PROMPT = """
You are an expert at creating slide transcripts for presentations.
Given the extracted text and images from presentation slides,
your task is to generate a detailed and coherent transcript that accurately reflects
the content and context of each slide.

# JSONL Output Format:
- Each reply must be in JSONL format. Each line is a standalone valid JSON object.
- The first line must describe the persona of the presenter.
- Subsequent lines must contain the slide transcripts, one line per slide.
- Each slide transcript must include the slide number and the corresponding transcript text.
- Ensure that the JSON objects are properly formatted and escaped.

# JSONL Example Outputs:
Example output for 3 slides:
```jsonl
{ "type": "persona", "gender": "Male", "description": "Front shot of a man in his 50s with a white background, professional lighting, high resolution, photorealistic." }
{ "type": "slide_transcript", "slide_number": 1, "transcript": "Welcome to our presentation on AI advancements. We will explore the latest trends in artificial intelligence, including machine learning, natural language processing, and computer vision. Our goal is to provide insights into how these technologies are shaping the future." }
{ "type": "slide_transcript", "slide_number": 2, "transcript": "We delve into the applications of AI in various industries. We discuss how AI is revolutionizing healthcare through predictive analytics, personalized medicine, and robotic surgery. Additionally, we will examine AI's impact on finance, retail, and transportation sectors." }
{ "type": "slide_transcript", "slide_number": 3, "transcript": "We highlighted the transformative power of AI and its applications across different fields. As we move forward, it is crucial to consider the ethical implications and ensure responsible AI development. Thank you for joining us on this journey into the world of artificial intelligence." }
```
Example output for 10 slides:
```jsonl
{ "type": "persona", "gender": "Female", "description": "Front shot of a woman in her 30s with a black background, professional lighting, high resolution, photorealistic." }
{ "type": "slide_transcript", "slide_number": 1, "transcript": "Welcome to our comprehensive presentation on the advancements in artificial intelligence. In this session, we will explore the latest trends and technologies shaping the AI landscape, including machine learning, natural language processing, and computer vision." }
{ "type": "slide_transcript", "slide_number": 2, "transcript": "In this section, we will delve into the applications of AI across various industries. We will discuss how AI is revolutionizing healthcare through predictive analytics, personalized medicine, and robotic surgery." }
{ "type": "slide_transcript", "slide_number": 3, "transcript": "Continuing our exploration, we will examine AI's impact on the finance sector. Topics include algorithmic trading, fraud detection, and customer service automation." }
{ "type": "slide_transcript", "slide_number": 4, "transcript": "Next, we will look at AI's role in retail. We will cover personalized shopping experiences, inventory management, and supply chain optimization." }
{ "type": "slide_transcript", "slide_number": 5, "transcript": "In this segment, we will discuss AI's influence on transportation. Key areas include autonomous vehicles, traffic management systems, and logistics optimization." }
{ "type": "slide_transcript", "slide_number": 6, "transcript": "Moving forward, we will explore the ethical considerations surrounding AI development. Topics include bias mitigation, transparency, and accountability in AI systems." }
{ "type": "slide_transcript", "slide_number": 7, "transcript": "In this section, we will highlight the importance of data privacy and security in AI applications. We will discuss best practices for safeguarding sensitive information." }
{ "type": "slide_transcript", "slide_number": 8, "transcript": "Next, we will examine the future of AI research and development. We will explore emerging technologies and potential breakthroughs in the field." }
{ "type": "slide_transcript", "slide_number": 9, "transcript": "As we near the conclusion of our presentation, we will summarize the key takeaways and insights from our discussion on AI advancements." }
{ "type": "slide_transcript", "slide_number": 10, "transcript": "Thank you for joining us on this journey into the world of artificial intelligence. We hope this presentation has provided valuable insights into the transformative power of AI and its applications across various industries." }
```
Example output for 13 slides:
```jsonl
{"type": "persona", "gender": "Male", "description": "Front shot of a man in his 30s with a blue background, professional lighting, high resolution, photorealistic."}
{"type": "slide_transcript", "slide_number": 1, "transcript": "StreamWise: Serving multi-modal generation in real-time at scale."}
{"type": "slide_transcript", "slide_number": 2, "transcript": "Motivation: LLMs are popular, multi-modal generation is emerging and expensive."}
{"type": "slide_transcript", "slide_number": 3, "transcript": "Example app: Real-time video podcast uses LLMs, text-to-image, and more."}
{"type": "slide_transcript", "slide_number": 4, "transcript": "Core: Diffusion Transformers (DiT) comprise image, text, and VAE encoders/decoders."}
{"type": "slide_transcript", "slide_number": 5, "transcript": "DiT performance shows latency vs. number of frames and de-noising steps."}
{"type": "slide_transcript", "slide_number": 6, "transcript": "Techniques for efficiency: latency, cost reduction, and quality optimization."}
{"type": "slide_transcript", "slide_number": 7, "transcript": "Deadline-aware scheduling delays generation of final scenes for efficiency."}
{"type": "slide_transcript", "slide_number": 8, "transcript": "StreamWise handles model onboarding, scheduling, and request execution efficiently."}
{"type": "slide_transcript", "slide_number": 9, "transcript": "Implementation utilizes Azure VMs, Kubernetes, and StreamWise components."}
{"type": "slide_transcript", "slide_number": 10, "transcript": "Example provisioning shows GPU requirements for a 10-minute video podcast."}
{"type": "slide_transcript", "slide_number": 11, "transcript": "TTFF versus cost comparison highlights hardware and optimization tradeoffs."}
{"type": "slide_transcript", "slide_number": 12, "transcript": "Adaptive quality dynamically adjusts resolution to balance cost and performance."}
{"type": "slide_transcript", "slide_number": 13, "transcript": "Conclusions: Real-time serving is possible with opportunities for optimization."}
```
"""
