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

1. **[Build the Docker image](#building-images)** — run `bash setup_image.sh` in the wrapper's directory
2. **[Push to Azure Container Registry](#pushing-to-acr)** — use `--push` flag or push manually
3. **Deploy to Kubernetes**
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

Each wrapper directory contains:
- `Dockerfile` - Container image definition
- `setup_image.sh` - Thin build script that calls the shared [`setup_image.sh`](setup_image.sh)
- `requirements.txt` - Python dependencies (if applicable)

Image names and tags are read from [`services.json`](../../services.json).

### Build All Wrapper Images

From the `deployment/` directory, use [`build_docker.sh`](../build_docker.sh) to build **and push** every wrapper and app image:

```bash
cd deployment
source set_properties.sh
bash build_docker.sh
```

### Build a Single Wrapper Image

Navigate to the wrapper directory and run its `setup_image.sh`:

```bash
cd deployment/wrappers/<wrapper-name>
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR
```

For example, to build and push the Wan video wrapper:
```bash
cd deployment/wrappers/wan
bash setup_image.sh --push
```

#### setup_image.sh Flags

| Flag | Description |
|------|-------------|
| `--push` | Push the image to ACR after building |
| `--hf_token` | Include `HF_TOKEN` as a Docker build secret (for gated models) |
| `--platform <arch>` | Target platform, e.g. `linux/amd64` (default) or `linux/arm64` |
| `--folders <f1,f2>` | Copy additional subdirectories into the build context |
| `--parent <name>` | Include files from a parent wrapper directory |

> **Note:** Wrappers that require a Hugging Face token (e.g., `flux`, `gemma`, `llama32`) already pass `--hf_token` in their `setup_image.sh`. Ensure `HF_TOKEN` is set in [`set_properties.sh`](../set_properties.sh).

## Pushing to ACR

If you built without `--push`, you can push manually:

```bash
source ../set_properties.sh
az acr login --name $ACR_NAME
docker push $ACR_URL/<wrapper-name>:<tag>
```

Check which images are available locally:
```bash
docker image ls "$ACR_URL/*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

Verify the image is in ACR:
```bash
az acr repository list --name $ACR_NAME -o table
az acr repository show-tags --name $ACR_NAME --repository <wrapper-name> -o table
```

For ACR creation and configuration, see [ACR Documentation](../acr/README.md).

## Deployment

Refer to individual wrapper README files for specific deployment instructions and configuration options.

