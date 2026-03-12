# Azure Deployment with Bicep

This directory contains Azure Bicep templates for deploying GPU-enabled VMs hosting StreamWise services on Kubernetes.

## Overview

We use [Azure Bicep](https://github.com/Azure/bicep) to deploy and manage the infrastructure:
- GPU-enabled VMs for model serving
- Kubernetes cluster setup across VMs
- Networking and bastion host for secure access

**Note:** For managed AKS deployment (recommended for most users), see [AKS Deployment Guide](../aks/README.md).

## Prerequisites

- Azure CLI installed and configured
- Azure subscription with GPU VM quota
- Bicep CLI (included with Azure CLI)

## Files

- `vm-gpu-k8s-deployment.bicep` - Main Bicep template
- `vm-gpu-k8s-deployment.bicepparam` - Parameters file
- [../acr/README.md](../acr/README.md#using-acr-with-kubernetes-and-aks) - ACR usage guide for this deployment (Kubernetes and AKS)


## Create VMs
We deploy all the components using:
```powershell
az bicep generate-params --file .\vm-gpu-k8s-deployment.bicep --output-format bicepparam --include-params all

$AZ_RESOURCE_GROUP="<choose>"
$AZ_REGION="<choose>"

az group create --name $AZ_RESOURCE_GROUP --location $AZ_REGION
az deployment group create --resource-group $AZ_RESOURCE_GROUP --parameters .\vm-gpu-k8s-deployment.bicepparam
```

## Connect to VMs
Connect to the control VM:
```powershell
az network bastion ssh -n gpu-bastion-host -g $AZ_RESOURCE_GROUP --target-ip 10.0.0.4 --resource-port 22 --auth-type password --username azureuser
```
For the other VMs, just change the IP.

## Setup K8s dashboard tunnel
Setup the tunnel in Azure Bastion:
```powershell
az network bastion tunnel -n gpu-bastion-host -g $AZ_RESOURCE_GROUP --target-ip 10.0.0.4 --resource-port 22 --port 12222
```

Open the SSH tunnel for the K8s dashboard:
```powershell
ssh -L 18443:localhost:8443 azureuser@localhost -p 12222
```

Get dashboard token:
```bash
kubectl -n kubernetes-dashboard create token admin-user
```

Setup the actual K8s dashboard tunnel through the proxy:
```bash
kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard-kong-proxy 8443:443
```

## Setup worker VMs
We should already have this in the key vault but we can do it manually:
```bash
kubeadm token create --print-join-command
```

Then in the worker:
```bash
kubeadm join 10.0.0.6:6443 --token XXXX --discovery-token-ca-cert-hash sha256:YYYY
```