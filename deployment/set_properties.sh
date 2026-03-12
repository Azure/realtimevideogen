#!/usr/bin/env bash

# Azure deployment
export RESOURCE_GROUP="myresourcegroup"  # TODO: fill this out
export REGION="SwedenCentral"  # TODO: fill with your region

# Hugging Face configuration
# https://huggingface.co/settings/tokens
# export HF_TOKEN="hf_XXX"
# export HF_HOME="path/to/huggingface"

# This is the default Docker repository but may have compliance issues
export DOCKER_REPO="nvidia"

# Use our own ACR as the base repo
export ACR_NAME="xxxx"  # TODO: fill this out
export ACR_FULL_NAME="$ACR_NAME-xxxx"  # TODO: fill this out
export ACR_URL="$ACR_FULL_NAME.azurecr.io"

export ACR_USERNAME="xxxx"  # TODO: fill this out
export ACR_EMAIL="xxxx"  # TODO: fill this out

# shellcheck disable=SC2034
export DOCKER_REPO="$ACR_URL"
