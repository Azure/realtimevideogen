#!/usr/bin/env bash

# Azure deployment
export AZ_SUBSCRIPTION_ID="12345678-1234-1234-1234-123456789012"  # TODO: fill this out
export AZ_RESOURCE_GROUP="myresourcegroup"  # TODO: fill this out
export AZ_REGION="SwedenCentral"  # TODO: fill with your region

# Hugging Face configuration
# https://huggingface.co/settings/tokens
export HF_TOKEN=""  # TODO: fill this out
# export HF_HOME="path/to/huggingface"


# Use our own ACR as the base repo
export ACR_NAME="xxxx"  # TODO: fill this out
export ACR_FULL_NAME="$ACR_NAME-yyyy"  # TODO: fill this out
export ACR_URL="$ACR_FULL_NAME.azurecr.io"

export ACR_USERNAME="xxxx"  # TODO: fill this out
export ACR_EMAIL="xxxx"  # TODO: fill this out

# This is the default Docker repository but may have compliance issues
export DOCKER_REPO="nvidia"
# shellcheck disable=SC2034
export DOCKER_REPO="$ACR_URL"

# Kubernetes
export K8S_NAMESPACE="rtgen"
