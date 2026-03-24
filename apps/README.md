# Applications
Workflows for multi-modal generation.

| Name | Workflow | Example input | Generated output |
|------|----------|---------------|-----------------|
| StreamCast | Podcast | Research paper | Video podcast with characters discussing the input content |
| StreamShort | Short | Movie | Extract key segments and generate short-form highlight videos |
| StreamMovie | Movie | High-level description | Multi-scene movie with characters, scenes, music, and others |
| StreamAnimated | Animated | Storyboard | Fully-animated story with characters and scenes |
| StreamLecture | Lecture | Textbook chapter | Lecture-style video with a professor avatar and supporting visual elements |
| StreamPersona | Slide persona | Slide deck | Presenter persona delivering a narrated walkthrough aligned with the slides |
| StreamDub | Dubbing | Video | Translated and lip-synchronized version while preserving speaker identity |
| StreamEdit | Editing | Video | Modified video modifying components of the original video (e.g., style) |
| StreamChat | Video chat | Bot dialogue | Character delivering their side of the conversation |

The main application is StreamCast.
For the others, we mainly modify the inputs for the LLM (content and prompt with few-shot examples) and the DAG generation.
We use the same components and libraries for the rest.

*Disclaimer*: This work focuses on the systems aspects.
The applications represent the overall workflow and resource requirements.
It does not target making these applications complete or their quality.


## 🎙️ StreamCast
We implement *podcast* video generation as a K8s service: StreamCast.
It exposes a [REST API](https://quart.palletsprojects.com/en/latest/) for users to submit requests (e.g., "generate a 10-minute video for this paper at medium quality").
This service can run on inexpensive CPU servers.

StreamCast builds the workflow DAG with the stages, scenes, and shots (e.g., audio for shot 3 in scene 2).
It starts with the screenplay node, and as the LLM generates scenes, it adds nodes to the DAG.
Initially, it creates a sketch DAG (e.g., a 5-minute video with 45-second shots) with rough deadlines, later refining them as actual nodes are generated (e.g., scene 2 from 0:37-1:12 must complete by 19:39 UTC).

StreamCast uses the *request scheduler* library to discover all deployed instance models in the K8s cluster, along with their hardware resources and locations.
The *scheduler* then dispatches DAG nodes to the appropriate model instances.
If a shot cannot meet its deadline, its quality is reduced.
Finally, StreamCast stitches the outputs using [FFmpeg](https://ffmpeg.org/) and streams the result back to the user.

Components:
* Text: ✅ Scene descriptions and dialogues
* Audio: ✅ Dialogue
* Image: ✅ ~3 images
* Video: ✅ ~20 shots


## ⚙️ Other Applications
All other applications are derived from the same core architecture as StreamCast.
They reuse the same services, model components, and scheduling infrastructure, and mainly differ in:
* The LLM prompt and few-shot examples
* The workflow DAG structure (number of stages, scenes, shots)
* The modalities enabled or disabled (audio, image, video)
* The target duration and quality constraints

| Workflow       | Example input   | Generated output                                      | Characteristic | 📄Text | 🔉Audio | 🖼️Image | 📽️Video |
|----------------|-----------------|-------------------------------------------------------|----------------|---------|---------|----------|---------|
| Podcast        | Paper           | Video podcast with characters discussing input        |                | 🟦 | 🟦 | 🟦 | 🔥 |
| Short          | Movie           | Extract key segments and generate highlight video     | Heavy LLM      | 🔥 | 🟦 | ❌ | 🟦 |
| Movie          | High-level plot | Multi-scene movie with characters, scenes,...         | Long output    | 🟦 | 🟦 | 🟨 | 🔥 |
| Animated story | Storyboard      | Fully-animated story with characters and scenes       | Style LoRA     | 🟦 | 🟦 | 🟦 | 🔥 |
| Lecture        | Textbook        | Video with professor and supporting visuals           | Static content | 🟦 | 🟦 | 🟦 | 🟨 |
| Slide persona  | Slide deck      | Embedded presenter persona narrating slides           | Low resolution | 🟦 | 🟦 | 🟦 | 🟨 |
| Dubbing        | TV show         | Translated and lip-synced preserving speaker          | Advanced TTS   | 🟦 | 🟨 | ❌ | 🔥 |
| Editing        | Video           | Modify components of the original video (e.g., style) | Heavy V2V      | ❌ | ❌ | ❌ | 🔥 |
| Video chat     | Bot dialogue    | Character delivering their side of the conversation   | Short outputs  | 🟦 | 🟦 | 🟦 | 🟨 |

Below we summarize the key differences for each application relative to the StreamCast (podcast) baseline.

### 🎞️ StreamShort
StreamShort generates short-form highlight videos from long-form content.
Components:
* Text: ✅
    * LLM input: Up to ~1M tokens (e.g., transcription + 1 FPS visual summaries for a 2-hour movie)
    * LLM output: scenes to output
* Audio: ❌ Reuse from original
* Image: ❌ Reuse from original
* Video: ❌ Shortening some scenes

Particularities:
* Output: Concise, narration-driven short clips

This application focuses on content selection and summarization rather than visual synthesis.


### 🎥 StreamMovie
StreamMovie generates long-form, multi-scene cinematic videos from a high-level description.
Components:
* Text: ✅
    * LLM output: ~12× longer screenplay than podcast
* Audio: ✅ Dialogue, music, effects
* Image: ✅ ~240 images (≈30-second scenes)
* Video: ✅ ~240 video clips (≈30 seconds each)

Particularities:
* Target duration: ~2 hours

This is the most resource-intensive workflow, stressing large-scale DAG scheduling and parallel video generation.


### 🎨 StreamAnimated
StreamAnimated produces fully animated stories with consistent visual style.
Components:
* Text: ✅
* Audio: ✅
* Image: ✅
* Video: ✅

Particularities:
* Style control: LoRA-based fine-tuning for animation style
* Overhead: ~5% additional video generation time

The LoRA allows stylistic consistency across characters and scenes with minimal performance impact.


### 🎓 StreamLecture
StreamLecture generates lecture-style videos from textbook chapters or technical documents.
Components:
* Text: ✅
* Audio: ✅ Professor-style narration
* Image: ✅ Extracted or generated from text and figures
* Video: ✅ Reduced

Particularities:
* ≈50% of shots replaced with static images (e.g., slides)

This workflow trades video synthesis for image-based explanations to reduce cost while preserving clarity.


### 📊 StreamPersona
StreamPersona generates narrated presentations aligned with slide decks.
Components:
* Text: ✅ Slide parsing and plot generation
* Audio: ✅ Explanation
* Image: ✅ Slides
* Video: ✅ Presenter avatar

Particularities:
* Resolution: Reduced (e.g., 1280×800 → 320×200)

Lower resolution and simpler visuals significantly reduce inference cost while maintaining presentation quality.


### 🌍 StreamDub
StreamDub translates and dubs existing videos while preserving speaker identity.
Components:
* Text: ✅ Transcription and translation
* Audio: ✅ 2× audio tracks (source + target language)
* Image: ❌
* Video: ✅ Lip sync video to the new audio

This workflow focuses on speech translation, voice cloning, and lip synchronization.


### ✂️ StreamEdit
StreamEdit modifies existing videos without regenerating content.
Components:
* Text: ❌
* Audio: ❌
* Image: ❌
* Video: ✅ Style transfer, filtering, compositing,...

This is a lightweight workflow centered on video post-processing.


### 💬 StreamChat (Video Chat)
StreamChat enables near-real-time character-based video responses.
Components:
* Text: ✅ Chat reply
* Audio: ✅ TTS
* Image: ✅ Character
* Video: ✅ Character speaking text

Particularities:
* Latency-sensitive: ✅
* Target duration: ~5 seconds per response

This application emphasizes low-latency scheduling and fast partial generation.


## Summary
Together, these applications demonstrate how a single modular, deadline-aware, multi-modal pipeline can support a wide range of video generation workloads by varying only prompts, DAG structure, and modality selection.
