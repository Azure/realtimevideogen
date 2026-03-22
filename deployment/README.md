# Deployment
We deploy the images in Docker containers on top of Kubernetes.

Some microservices and containers:
* Text+image to video:
  * [Wan](wrappers/wan)
  * [Hunyuan FramePack](wrappers/hunyuanframepack)
* Text to image:
  * [Flux](wrappers/flux).
* Text to audio:
  * [Kokoro](wrappers/kokoro).
* Text+image+audio to video
  * [Fantasy Talking](wrappers/fantasytalking)
  * [Hunyuan Avatar](wrappers/hunyuanavatar)
* Documents to text (plot)
  * [Podcast Transcript](wrappers/podcasttranscript)
* [LLM](wrappers/gemma):
  * [Gemma](wrappers/gemma)

We also have the [StreamCast](apps/streamcast) that orchestrates the workflow across containers.

---

## Deployment Options Overview

Choose your deployment path based on your requirements:

| Option | Best For | Complexity | Prerequisites |
|--------|----------|------------|---------------|
| **1. Azure Kubernetes Service (AKS)** ✅ | Production deployments, simplest setup | ⭐ Low | Azure subscription, Azure CLI |
| **2. Manual Kubernetes Cluster** | On-premises or custom deployments | ⭐⭐ Medium | Ubuntu VMs with GPU, manual K8s setup |
| **3. VM Deployment with Docker** | Development, testing, single-node setups | ⭐⭐⭐ High | Ubuntu VMs with GPU, Docker |

**Recommended:** Start with **Option 1 (AKS)** for the simplest and fastest deployment experience.

---

## Option 1: Azure Kubernetes Service (AKS) - RECOMMENDED ✅

Azure Kubernetes Service (AKS) is the **preferred deployment method** because:
- **Simplest setup**: AKS handles cluster management, node provisioning, and infrastructure automatically
- **Production-ready**: Built-in monitoring, auto-scaling, and high availability
- **Integrated**: Seamless integration with Azure Container Registry (ACR) and other Azure services

### Prerequisites for AKS

- **Azure Account**: Active Azure subscription with appropriate permissions
- **Azure CLI**: Installed and configured ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **kubectl**: Kubernetes command-line tool ([Install Guide](https://kubernetes.io/docs/tasks/tools/))
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))

### Quick Start for AKS

Follow these steps in order:

1. **[Create ACR](acr/README.md)** (if not already created)
2. **[Login to ACR](acr/README.md#login-and-configuration)** - Authenticate to ACR before building
3. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
4. **[Deploy AKS Cluster](aks/README.md)** - Follow the complete AKS deployment guide (includes namespace, storage, secrets, GPU setup)
5. **Deploy Services** - Choose one of two approaches:
   - **[Deploy with Helm](helm/README.md)** (command-line approach)
   - **[Deploy with StreamWise Web UI](#deploy-with-streamwise-cluster-manager-web-ui)** (visual interface approach - recommended for beginners)

For detailed AKS-specific instructions, see the **[AKS Deployment Guide](aks/README.md)**.

---

## Option 2: Manual Kubernetes Cluster Setup

Use this option if you need to deploy on your own infrastructure or require custom Kubernetes configurations.

### When to Use Manual Kubernetes

- On-premises deployments
- Custom hardware configurations
- Specific Kubernetes version requirements
- Integration with existing Kubernetes infrastructure

### Prerequisites for Manual Kubernetes

- **Ubuntu VMs**: Multiple VMs with GPU support (at least one master, one or more workers)
- **Azure CLI**: Installed and configured ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **kubectl**: Kubernetes command-line tool ([Install Guide](https://kubernetes.io/docs/tasks/tools/))
- **Docker**: For building and pushing images
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))
- **Azure Container Registry (ACR)**: Created and accessible ([ACR Setup](acr/README.md))

### Manual Kubernetes Setup Steps

For the complete step-by-step guide, see the **[Generic Kubernetes Setup Guide](k8s/README.md)**.

Follow the sections in order:

1. **[Setup Kubernetes](k8s/README.md#kubernetes-installation)** - Install and configure K8s cluster
2. **[Configure ACR](acr/README.md)**
3. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
4. **[Configure Namespace](k8s/README.md#namespace-setup)**
5. **[Setup Storage](k8s/README.md#storage-setup)**
6. **[Configure Secrets](k8s/README.md#secrets-configuration)**
7. **[GPU Setup](k8s/README.md#gpu-setup)** - NVIDIA device plugin and optional MIG setup
8. **[Install Helm](k8s/README.md#helm-installation)** (optional if using StreamWise web UI)
9. **Deploy Services** - Choose one of two approaches:
   - **[Deploy with Helm](helm/README.md)** (command-line approach)
   - **[Deploy with StreamWise Web UI](#deploy-with-streamwise-cluster-manager-web-ui)** (visual interface approach)

---

## Option 3: VM Deployment with Docker

Use this option for development, testing, or when you need to run containers directly on VMs without Kubernetes orchestration.

### When to Use VM with Docker

- Development and testing environments
- Single-node deployments
- Learning and experimentation
- Debugging specific containers

### Prerequisites for VM Deployment

- **Ubuntu VMs**: VMs with GPU support and NVIDIA drivers installed
- **Docker**: Docker with NVIDIA container runtime
- **Azure CLI**: For ACR access ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))
- **Azure Container Registry (ACR)**: Created and accessible ([ACR Setup](acr/README.md))

### VM with Docker Setup Steps

For detailed setup instructions, see the **[VM Deployment Guide](vm/README.md)**. The guide covers:

1. **[Setup Disks](vm/README.md#setup-disks)** - Configure NVMe drives for Docker storage
2. **[Install Docker](vm/README.md#docker-installation)** - Install Docker with NVIDIA runtime
3. **[Configure ACR](vm/README.md#azure-container-registry-acr-setup)** - Login and configure ACR access
4. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
5. **[Run Containers Manually](vm/README.md#running-containers-manually)** - Pull and run individual containers using `docker run`


---

## Deploy with StreamWise Cluster Manager Web UI

**Applies to:** Options 1, 2

Instead of using Helm from the command line, you can deploy the **StreamWise Cluster Manager** first and then use its web UI to deploy and manage the rest of the components.

**Benefits of this approach:**
- **Visual interface**: Manage deployments, pods, and services through a web UI
- **Simplified workflow**: Deploy components without memorizing kubectl/helm commands
- **Real-time monitoring**: View cluster status, resource usage, and logs in one place
- **Interactive management**: Start, stop, and configure services dynamically

### Deploy StreamWise Cluster Manager

**Prerequisites:** Complete namespace setup, storage setup, and secrets configuration (see [Generic Kubernetes Setup Guide](k8s/README.md) or [AKS Deployment Guide](aks/README.md)).

1. **Deploy StreamWise Service Account**:
   ```bash
   kubectl apply -f k8s/streamwise-service-account.yaml
   ```

2. **Configure environment variables** (for AKS with Load Balancer):
   ```bash
   export LOAD_BALANCER_IP=$(az network public-ip show --resource-group $AZ_RESOURCE_GROUP --name aks-pods-public-ip | jq -r .ipAddress)
   export RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP
   ```

3. **Deploy StreamWise Pod and Service**:
   ```bash
   cd deployment/aks
   envsubst < streamwise-pod.yaml | kubectl apply -f -
   ```

4. **Verify deployment**:
   ```bash
   kubectl get pods -n rtgen | grep streamwise
   kubectl get svc -n rtgen | grep streamwise
   ```

5. **Access the Web UI**:
   ```bash
   # Or use port forwarding for local access
   kubectl port-forward -n rtgen svc/streamwise 8080:8080
   ```
   Open your browser to `http://localhost:8080` (if using port-forward) or `http://$LOAD_BALANCER_IP:8081`.

### Deploy Components from Web UI

Once the StreamWise web UI is accessible:

1. **Navigate to the Web UI** in your browser
2. **View cluster status**: The dashboard shows all nodes, pods, and services
3. **Deploy services**: Use the web interface to submit jobs, configure resource allocations, and deploy model wrappers
4. **Monitor deployments**: Track pod status, logs, and resource usage in real-time
5. **Manage services**: Start, stop, restart, or delete services as needed

---


## Complete Deployment Example (AKS)

**Applies to:** Option 1 (Azure Kubernetes Service)

For a complete end-to-end AKS deployment workflow with all steps from prerequisites to testing, see the **[AKS Deployment Guide - Complete Example](aks/README.md#complete-end-to-end-deployment-example)**.

---

## Building and Pushing Docker Images

Every component — the shared base image, model wrappers (in [`wrappers/`](wrappers/)), the StreamWise cluster manager (in [`streamwise/`](streamwise/)), and applications (in [`apps/`](apps/)) — follows the same build pattern. Each directory contains:

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image definition |
| `setup_image.sh` | Build script |
| `requirements.txt` | Python dependencies (if applicable) |

Image names and tags are defined centrally in [`services.json`](../services.json).

> **Build order:** the `base` image must be built before any wrapper or app image, because all other images inherit from it.

### Prerequisites

Key variables set by `set_properties.sh`:

| Variable | Description | Example |
|----------|-------------|---------|
| `ACR_NAME` | Short ACR registry name | `myregistry` |
| `ACR_URL` | Full ACR login server URL | `myregistry-abc.azurecr.io` |
| `DOCKER_REPO` | Docker image prefix (set to `$ACR_URL`) | `myregistry-abc.azurecr.io` |
| `HF_TOKEN` | Hugging Face token for gated models | `hf_...` |

`DOCKER_REPO` is always the same as `ACR_URL`. All built images are tagged `$DOCKER_REPO/<name>:<tag>` so they can be pushed directly to ACR.

```bash
cd deployment
source set_properties.sh       # loads the variables above
az acr login --name $ACR_NAME  # authenticate to ACR
```

> **CI / agent environments:** if `DOCKER_REPO` is already set in the environment, `set_properties.sh` and the ACR login step are skipped automatically by the individual `setup_image.sh` scripts. Set `DOCKER_REPO=<acr-url>` before calling any build script to enable headless builds.

### Build All Images at Once

[`build_docker.sh`](build_docker.sh) reads every service from `services.json`, builds the base image first, then every wrapper and app, and finally pushes everything to ACR:

```bash
cd deployment
source set_properties.sh
bash build_docker.sh
```

### Build the Base Image

All other images depend on the base image. Build it first when setting up a new environment or after updating [`deployment/base/Dockerfile`](base/Dockerfile):

```bash
cd deployment/base
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR
```

### Build a Single Wrapper Image

```bash
cd deployment/wrappers/<wrapper-name>
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR
```

Wrappers that require a Hugging Face token for gated model access (e.g., `flux`, `gemma`, `llama32`) pass `--hf_token` automatically — just make sure `HF_TOKEN` is set in `set_properties.sh`.

Available flags for wrapper builds:

| Flag | Description |
|------|-------------|
| `--push` | Push the image to ACR after building |
| `--hf_token` | Include `HF_TOKEN` as a Docker build secret (for gated models) |
| `--platform <arch>` | Target platform, e.g. `linux/amd64` (default) or `linux/arm64` |
| `--folders <f1,f2>` | Copy additional subdirectories into the build context |
| `--parent <name>` | Include files from a parent wrapper directory |

### Build a Single App or StreamWise Image

```bash
cd deployment/apps/<app-name>   # e.g. deployment/apps/streamcast
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR

# StreamWise cluster manager
cd deployment/streamwise
bash setup_image.sh --push
```

### Push an Image Manually

If you built an image without `--push`, push it separately:

```bash
source set_properties.sh
az acr login --name $ACR_NAME
docker push $ACR_URL/<image-name>:<tag>
```

### Verify Images in ACR

```bash
az acr repository list --name $ACR_NAME -o table
az acr repository show-tags --name $ACR_NAME --repository <image-name> -o table
```

For ACR creation and configuration, see [ACR Documentation](acr/README.md).
