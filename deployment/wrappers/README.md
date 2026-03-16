# StreamWise Wrappers Deployment

This directory contains deployment scripts and documentation for StreamWise model wrappers.

## Overview

Each subdirectory contains the Docker image building scripts for a specific [model wrapper](../../docs/model_wrapper.md).

## Available Wrappers

- **[Gemma](gemma/README.md)** - Google Gemma LLM deployment using vLLM
- **[Llama 3.2](llama32/README.md)** - Meta Llama 3.2 LLM deployment using vLLM
- **[Whisper](whisper/README.md)** - OpenAI Whisper audio transcription model

## General Deployment Workflow

For deploying any wrapper:

1. **Build the Docker image** — see [Building and Pushing Docker Images](../README.md#building-and-pushing-docker-images)
2. **Deploy to Kubernetes**
   - Use Helm charts from [deployment/helm](../helm/README.md)
   - Or use direct kubectl deployment with pod/service YAML files

## Prerequisites

Before deploying any wrapper:

- **Azure Container Registry**: See [ACR Setup](../acr/README.md)
- **Kubernetes Cluster**: Either [AKS](../aks/README.md) or manual setup
- **Hugging Face Token**: Required for gated models ([Get Token](https://huggingface.co/settings/tokens))
- **GPU Resources**: Most wrappers require GPU nodes in your cluster

## Common Configuration

All wrappers require:

- **Namespace**: Typically `rtgen` (see [Namespace Setup](../README.md#kubernetes-namespace-setup))
- **Secrets**: HF token and optionally ACR credentials (see [Secrets Configuration](../README.md#secrets-configuration))
- **Storage**: Persistent volumes for model caching (see [Storage Setup](../README.md#storage-setup))

## Building Images

For complete build instructions — including the required base image build step, `build_docker.sh` for building all images at once, and individual `setup_image.sh` usage — see [Building and Pushing Docker Images](../README.md#building-and-pushing-docker-images).

Each wrapper directory contains:
- `Dockerfile` — container image definition
- `setup_image.sh` — thin build script that delegates to the shared [`setup_image.sh`](setup_image.sh)
- `requirements.txt` — Python dependencies (if applicable)

Image names and tags are read from [`services.json`](../../services.json).

### setup_image.sh Flags

| Flag | Description |
|------|-------------|
| `--push` | Push the image to ACR after building |
| `--hf_token` | Include `HF_TOKEN` as a Docker build secret (for gated models) |
| `--platform <arch>` | Target platform, e.g. `linux/amd64` (default) or `linux/arm64` |
| `--folders <f1,f2>` | Copy additional subdirectories into the build context |
| `--parent <name>` | Include files from a parent wrapper directory |

> **Note:** Wrappers that require a Hugging Face token (e.g. `flux`, `gemma`, `llama32`) already pass `--hf_token` in their `setup_image.sh`. Ensure `HF_TOKEN` is set in [`set_properties.sh`](../set_properties.sh).

## Deployment

Refer to individual wrapper README files for specific deployment instructions and configuration options.

