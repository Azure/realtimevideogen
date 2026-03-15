#!/usr/bin/env bash

# shellcheck disable=SC1091
source ../set_properties.sh
# shellcheck source=deployment/setup_lib.sh
source ../setup_lib.sh

# Setup namespace
if ! kubectl get namespace "$K8S_NAMESPACE" &> /dev/null; then
    kubectl create namespace "$K8S_NAMESPACE"
else
    echo "Namespace $K8S_NAMESPACE already exists"
fi

# Setup storage
kubectl apply -f local-pv.yaml -n "$K8S_NAMESPACE"
kubectl apply -f local-pvc.yaml -n "$K8S_NAMESPACE"

# Azure Container Registry
if ! az account show &> /dev/null; then
    echo "Not logged in"
    az login
fi
ensure_acr_login "$ACR_NAME"
az account set --subscription "$AZ_SUBSCRIPTION_ID"
az acr list --output table
az acr update -n "$ACR_NAME" --admin-enabled true
az acr credential show --name "$ACR_NAME"

ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" | jq -r .passwords[0].value)
export ACR_PASSWORD

if kubectl get secret acr-secret -n "$K8S_NAMESPACE" &> /dev/null; then
    kubectl delete secret acr-secret -n "$K8S_NAMESPACE"
fi
kubectl create secret docker-registry acr-secret -n "$K8S_NAMESPACE" \
  --docker-server="$ACR_URL" \
  --docker-username="$ACR_USERNAME" \
  --docker-password="$ACR_PASSWORD" \
  --docker-email="$ACR_EMAIL"

# HF_TOKEN is required for Hugging Face models
if [ -z "$HF_TOKEN" ]; then
    read -r -p "Enter your Hugging Face token: " HF_TOKEN
fi
if kubectl get secret hf-token -n "$K8S_NAMESPACE" &> /dev/null; then
    kubectl delete secret hf-token -n "$K8S_NAMESPACE"
fi
kubectl create secret generic hf-token -n "$K8S_NAMESPACE" --from-literal=token="$HF_TOKEN"

# Deploy Helm chart
#helm install gpu-services . --namespace $K8S_NAMESPACE --create-namespace
IMAGE_TAGS=$(jq -r 'to_entries | map(select(.value | type == "object" and has("dockerImage"))) | map("images.\(.key).tag=\(.value.dockerImage.tag)") | join(",")' ../../services.json)
helm install gpu-services . \
  --namespace "$K8S_NAMESPACE" \
  --create-namespace \
  --set "registry=$ACR_URL" \
  --set "$IMAGE_TAGS"
