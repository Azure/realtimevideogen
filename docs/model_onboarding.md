# Model on-boarding
We package each model as a Docker container, based on an [NVIDIA image](https://hub.docker.com/layers/nvidia/cuda/12.9.1-cudnn-devel-ubuntu24.04/images/) with GPU drivers and runtime tools.
Each container embeds our _Instance Manager_, which standardizes the interface for executing inference requests.
We adapt existing inference code (typically from [Hugging Face](https://huggingface.co/models)) to this interface and bundle it with the model weights.

## Profiling
We generate simple model profiles to estimate runtime and resource usage, as key parameters (e.g., pixel count, frame count) scale proportionally.
We benchmark a representative configuration (e.g., 1+16 frames, 10 steps, 640 x $400 resolution) and validate it against additional test points.
We also measure peak power, energy, and temperature.
These data inform predictive models for performance, cost, and quality under different configurations.

## Parallelism
Many diffusion models include native support for multi-GPU inference (e.g., [Wan](https://github.com/Wan-Video/Wan2.1)).
For those that do not, we use [USP](https://arxiv.org/abs/2503.06132) from [xDiT](https://github.com/xdit-project/xDiT).
We have enabled parallelism for four models (e.g., Fantasy Talking, Hunyuan FramePack), each requiring under two hours of work.
The [xfuser](https://github.com/xdit-project/xDiT) repository provides examples, and this process could be streamlined with LLM-based coding agents.

## Profiling
We use [scikit-learn](https://scikit-learn.org/stable/index.html) to fit linear models.
Our runtime and cost profiles are over 99.9% accurate.

## Quality
When on-boarding the model, StreamWise uses the Elo rankings from [public leaderboards](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard).