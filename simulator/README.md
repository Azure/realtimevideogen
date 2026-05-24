# Multimodal Workflow Simulator

This directory contains code for running large-scale simulations of multimodal workflows to estimate costs, energy consumption, and resource allocation strategies.
The simulator uses **StreamCast** as a reference example, i.e., a video generation workflow composed of multiple generative model components including LLMs, audio generation, image generation, video generation, and lip-sync.

## Overview

The simulator helps answer critical questions for deploying multimodal AI workflows:
- **Cost Estimation**: What is the total cost to run X requests using Y GPU configurations?
- **Resource Allocation**: How should GPUs be distributed across different model components to minimize latency and cost?

The simulator operates on profiled data (latency and power measurements) from different GPU types (A100, H100, H200 etc.) and uses optimization algorithms to determine optimal resource allocation.

## Directory Structure

```text
simulator/
├── README.md                          # This file
├── constants.py                       # Workflow configuration constants
├── data_loading.py                    # Data loading utilities for profiles
├── data/                              # Latency and power profile data
│   ├── latency_*_*.csv                # Latency profiles per component and GPU type
│   ├── power_*_*.csv                  # Power consumption profiles
│   └── *.csv                          # Other benchmark data
...
```

## Key Components

### 1. Data Files (`data/`)

Contains empirically measured latency and power profiles:

- **Latency Profiles**: `latency_<component>_<gpu_type>.csv`
  - Measured execution time for each component on different GPU counts
  - Components: gemma (LLM), flux (image gen), hf (HunyuanFramePack video gen), ft (FantasyTalking lip-sync), upscaler

- **GPU Types**: A100, H100, H200, GB200

### 2. Supporting Files

- **`constants.py`**: Workflow configuration parameters (number of scenes, frames per scene, model hyperparameters)
- **`data_loading.py`**: Utilities to load and parse model (component) profile CSV files

## Example Workflow: StreamCast

The simulator models a video generation workflow with these stages:

1. **LLM (Gemma)**: Generate scene descriptions and narration
2. **Image Generation (Flux)**: Create key frame images from descriptions
3. **Video Generation (HunyuanFramePack)**: Generate video frames from keyframes
4. **VAE Decoding**: Decode latent representations to pixels
5. **Lip-Sync (FantasyTalking)**: Synchronize lip movements with audio
6. **Upscaling**: Enhance final video resolution

Each stage can be parallelized differently, and the simulator finds the optimal GPU allocation to minimize latency and cost.