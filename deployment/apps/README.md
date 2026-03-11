# StreamWise Apps Deployment

This directory contains deployment scripts for StreamWise applications.

## Overview

StreamWise applications orchestrate workflows across multiple model wrappers to deliver complete solutions. For more information about available applications, see [Apps README](../../apps/README.md).

## Prerequisites

Before deploying applications:

1. **Model Wrappers Deployed**: Applications depend on model wrapper services
   - See [Wrappers Deployment](../wrappers/README.md)
   - Deploy required wrappers using [Helm](../helm/README.md)

2. **Kubernetes Setup**: Cluster with required resources
   - Namespace configured (see [Namespace Setup](../README.md#kubernetes-namespace-setup))
   - Storage configured (see [Storage Setup](../README.md#storage-setup))
   - Secrets configured (see [Secrets Configuration](../README.md#secrets-configuration))

3. **Azure Container Registry**: Images pushed to ACR
   - See [ACR Documentation](../acr/README.md)

## Deployment

Each application directory contains:
- `Dockerfile` - Container image definition
- Build scripts for creating Docker images
- Kubernetes deployment manifests (YAML files)

### General Deployment Steps

1. **Build the application image** (if not using pre-built images):
   ```bash
   cd <application-directory>
   # Follow application-specific build instructions
   ```

2. **Push to ACR**:
   ```bash
   ACR_NAME="your-acr-name"  # TODO: Fill
   docker tag <app-image> $ACR_NAME.azurecr.io/<app-name>:latest
   docker push $ACR_NAME.azurecr.io/<app-name>:latest
   ```

3. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f <app-deployment>.yaml -n rtgen
   kubectl apply -f <app-service>.yaml -n rtgen
   ```

4. **Verify deployment**:
   ```bash
   kubectl get pods -n rtgen
   kubectl get svc -n rtgen
   ```

## Available Applications

Refer to [Apps README](../../apps/README.md) for the complete list of available applications and their specific requirements.

## Troubleshooting

If application pods fail to start:
- Verify all required model wrapper services are running
- Check pod logs: `kubectl logs <pod-name> -n rtgen`
- Verify secrets and configuration: `kubectl describe pod <pod-name> -n rtgen`
- Ensure ACR credentials are properly configured

