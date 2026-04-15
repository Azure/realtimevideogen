# ThinkSound Service

This service provides video-to-audio generation using the ThinkSound model.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the ThinkSound model checkpoints:
```bash
# Download the synchformer checkpoint
mkdir -p ckpts
wget -O ckpts/synchformer_state_dict.pth https://huggingface.co/FunAudioLLM/ThinkSound/resolve/main/synchformer_state_dict.pth
```

3. Clone and setup the ThinkSound repository:
```bash
git clone https://github.com/FunAudioLLM/ThinkSound.git
cd ThinkSound
pip install -e .
```

## Usage

### HTTP Server

Start the HTTP server:
```bash
python run_thinksound.py --host 0.0.0.0 --port 18087
```

### API Endpoint

POST to `/thinksound` with JSON payload:
```json
{
    "video": "base64_encoded_video_data",
    "caption": "Short description of the video",
    "caption_cot": "Detailed chain-of-thought description",
    "max_duration_sec": 10.0
}
```

Response: Audio data as WAV binary.

### Parameters

- `video`: Base64 encoded video file
- `caption`: Short caption describing the video content
- `caption_cot`: Detailed chain-of-thought description for better audio generation
- `max_duration_sec`: Maximum duration for generated audio (optional)

## Model Architecture

ThinkSound uses a two-stage approach:
1. **Feature Extraction**: Extract visual and textual features from input video and captions
2. **Audio Generation**: Generate audio using a diffusion model conditioned on the extracted features

## Requirements

- CUDA-compatible GPU with at least 8GB VRAM
- Python 3.8+
- FFmpeg for video processing
