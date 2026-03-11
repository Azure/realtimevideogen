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

1. **Build the Docker image** (if not using pre-built images from ACR)
   - Navigate to the specific wrapper directory
   - Follow the build instructions in the wrapper's README

2. **Push to Azure Container Registry**
   - Tag the image with your ACR URL
   - Push to ACR (see [ACR Documentation](../acr/README.md))

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
- `setup_image.sh` - Image building script
- `requirements.txt` - Python dependencies (if applicable)
- Additional model-specific files

General build pattern:
```bash
cd <wrapper-directory>
./setup_image.sh
```

## Pushing to ACR

After building locally:
```bash
ACR_NAME="your-acr-name"  # TODO: Fill
docker tag <local-image-name> $ACR_NAME.azurecr.io/<wrapper-name>:latest
docker push $ACR_NAME.azurecr.io/<wrapper-name>:latest
```

For detailed ACR instructions, see [Deployment README](../README.md#azure-container-registry-acr).

## Deployment

Refer to individual wrapper README files for specific deployment instructions and configuration options.

