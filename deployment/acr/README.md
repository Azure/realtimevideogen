# Azure Container Registry (ACR)
For compliance, we only use images from a dedicated [ACR](https://learn.microsoft.com/en-us/azure/container-registry/).
We clone public Docker images (vLLM, Kubernetes,...) into our own ACR and upload the Docker images into it too.

## Create an ACR
```bash
SUBSCRIPTION_ID="12345678-1234-1234-1234-123456789abc"  # TODO fill
RESOURCE_GROUP="abcd-acr"  # TODO fill
REGION="eastus"  # TODO fill
ACR_NAME="abcd"  # TODO fill

az login
az account set --subscription $SUBSCRIPTION_ID

az group create \
  --name $RESOURCE_GROUP \
  --location $REGION

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard \
  --admin-enabled true

az acr show \
  -g $RESOURCE_GROUP \
  --name $ACR_NAME
```

## Login and Configuration
For login and credential setup, see [Deployment README](../README.md#azure-container-registry-acr).
