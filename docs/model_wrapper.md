# 📦 Model wrapper and on-boarding
We wrap models publicly available on [Hugging Face](https://huggingface.co/models) using our own HTTP REST interface [wrapper](wrapper/wrapper_model.py).
We package each model as a Docker container, based on an [NVIDIA image](https://hub.docker.com/layers/nvidia/cuda/12.9.1-cudnn-devel-ubuntu24.04/images/) with GPU drivers and runtime tools.
Each container embeds our _Instance Manager_, which standardizes the interface for executing inference requests.
We adapt existing inference code (typically from [Hugging Face](https://huggingface.co/models)) to this interface and bundle it with the model weights.

## 📚 Libraries
We leverage the following:
* [Diffusers](https://github.com/huggingface/diffusers) to provide a simple and unified interface.
* [xDiT](https://github.com/xdit-project/xDiT) for parallelization of the models.
* [vLLM](https://github.com/vllm-project/vllm) for the OpenAI interface.

## 🤖 Models

| Model Name | Class |
|------------|-------|
| [Fantasy Talking](https://github.com/Fantasy-AMAP/fantasy-talking) | 🔤🖼️🔊➔🎥 Text+Image+Audio to Video |
| [FLUX](https://github.com/black-forest-labs/flux) | 🔤➔🖼️ Text to Image |
| [FLUX Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) | 🖼️➔🖼️ Image to Image |
| [FLUX Krea](https://github.com/krea-ai/flux-krea) | 🔤➔🖼️ Text to Image |
| [FLUX Kontext](https://github.com/black-forest-labs/flux) | 🖼️➔🖼️ Image to Image |
| [4KAgent](https://github.com/taco-group/4KAgent) | 🖼️➔🖼️ Image to Image |
| [HiDream I1](https://github.com/HiDream-ai/HiDream-I1) | 🔤➔🖼️ Text to Image |
| [Qwen Image](https://huggingface.co/Qwen/Qwen-Image) | 🔤➔🖼️ Text to Image |
| [Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | 🖼️➔🖼️ Image to Image |
| [Janus Pro](https://github.com/deepseek-ai/Janus) | 🔤➔🖼️ Text to Image |
| [LlamaGen](https://github.com/FoundationVision/LlamaGen) | 🔤➔🖼️ Text to Image |
| [Bagel](https://github.com/bytedance-seed/BAGEL) | 🖼️➔🖼️ Image to Image |
| [Hunyuan Image](https://huggingface.co/tencent/HunyuanImage-3.0) | 🔤➔🖼️ Text to Image |
| [Hunyuan FramePack](https://github.com/lllyasviel/FramePack) | 🔤🖼️➔🎥 Text+Image to Video |
| [Hunyuan FramePack F1](https://github.com/lllyasviel/FramePack) | 🔤🖼️➔🎥 Text+Image to Video |
| [Hunyuan Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar) | 🔤🖼️🔊➔🎥 Text+Image+Audio to Video |
| [Kokoro](https://github.com/hexgrad/kokoro) | 🔤➔🔊 Text to Audio  |
| [XTTS](https://github.com/coqui-ai/TTS) | 🔤➔🔊 Text to Audio |
| [ThinkSound](https://github.com/FunAudioLLM/ThinkSound) | 🎥➔🔊 Video to Audio |
| [VibeVoice](https://github.com/microsoft/VibeVoice) | 🔤➔🔊 Text to Audio |
| [Wan 2.1](https://github.com/Wan-Video/Wan2.1) | 🔤🖼️➔🎥 Text+Image to Video |
| [Wan 2.2](https://github.com/Wan-Video/Wan2.2) | 🔤🖼️➔🎥 Text+Image to Video |
| [YOLO](https://github.com/ultralytics/ultralytics) | 🖼️➔🖼️ Image to Image |
| [Image Resize](https://pillow.readthedocs.io/en/stable/reference/Image.html) | 🖼️➔🖼️ Image to Image |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 🖼️➔🖼️ Image to Image |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) | 🔤🖼️➔🎥 Text+Image to Video |
| [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) | 🔤🖼️➔🎥 Text+Image to Video |
| [Gemma 3](https://github.com/google-deepmind/gemma) | 🤖 LLM |
| [Llama 3.2](https://github.com/meta-llama/llama-models/) | 🤖 LLM |
| [whisper](https://github.com/openai/whisper) | 🔊➔🔤 Audio to Text |

The characteristics for each model are in ([services.json](../services.json)).
These characteristics include quality ([Elo ranking](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard)), frame rate (FPS), maximum number of frames (video length), number of attention heads, VAE compression ratios, supported resolutions, and other relevant attributes.

## 📊 Profiling
We generate simple model profiles to estimate runtime and resource usage, as key parameters (e.g., pixel count, frame count) scale proportionally.
We benchmark a representative configuration (e.g., 1+16 frames, 10 steps, 640 x 400 resolution) and validate it against additional test points.
We also measure peak power, energy, and temperature.
These data inform predictive models for performance, cost, and quality under different configurations.

## ⚡ Parallelism
Many diffusion models include native support for multi-GPU inference (e.g., [Wan](https://github.com/Wan-Video/Wan2.1)).
For those that do not, we use [USP](https://arxiv.org/abs/2503.06132) from [xDiT](https://github.com/xdit-project/xDiT).
We have enabled parallelism for four models (e.g., Fantasy Talking, Hunyuan FramePack), each requiring under two hours of work.
The [xfuser](https://github.com/xdit-project/xDiT) repository provides examples, and this process could be streamlined with LLM-based coding agents.

## 🎯 Accuracy
We use [scikit-learn](https://scikit-learn.org/stable/index.html) to fit linear models.
Our runtime and cost profiles are over 99.9% accurate.

## 🏆 Quality
When on-boarding the model, StreamWise uses the Elo rankings from [public leaderboards](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard).
