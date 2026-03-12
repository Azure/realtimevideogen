# Project: Real-Time Multi-Modal Generation

Modular, adaptive serving stack for real-time multi-modal generation (video, audio, images).
It dynamically balances latency, cost, and quality, and supports streaming (real-time playback).
It runs on a Kubernetes cluster with GPU nodes.

## Repository layout

| Path | Purpose |
|------|---------|
| `services.json` | Registry of every model: Docker image tags, input/output types, quality metrics. |
| `apps/` | Application workflows (StreamCast, StreamChat, etc.) that orchestrate model microservices. |
| `wrapper/` | Wrappers that exposes an HTTP endpoint for multi-modal models. |
| `deployment/` | All deployment artefacts (Docker, Kubernetes, Helm, Bicep, AKS, ACR, VM). |
| `deployment/set_properties.sh` | Central config with Azure subscription, resource group, region, ACR, HF token, K8s namespace. Source this before any deployment command. |
| `deployment/aks/` | AKS-specific deployment (Bicep template, pod YAMLs, service accounts, NVIDIA plugin, PV/PVC). |
| `deployment/helm/` | Helm chart for deploying GPU model microservices. |
| `deployment/acr/` | ACR creation and image mirroring docs. |
| `deployment/bicep/` | Shared Bicep modules (ACR role assignment, VM-based K8s, bastion, etc.). |
| `deployment/wrappers/` | Per-model Docker build contexts (Dockerfile + `setup_image.sh` for each model). |
| `streamwise/` | StreamWise cluster manager source code. |
| `simulator/` | Provisioning and scheduling simulator. |

## Key conventions

- **Namespace**: all K8s resources go in namespace `rtgen` (set via `$K8S_NAMESPACE`).
- **Docker images**: tagged as `<ACR_URL>/<model>:<tag>`, tags come from `services.json`.
- **Pod YAMLs** in `deployment/aks/` use shell variable placeholders (`${ACR_URL}`, `${LOAD_BALANCER_IP}`, `${RESOURCE_GROUP_NAME}`). Deploy with `envsubst < file.yaml | kubectl apply -f -`.
- **Helm chart** in `deployment/helm/` reads image tags from `services.json` via the `deploy.sh` script.
- **GPU spot node pools** start at 0 nodes to save cost; scale up explicitly before deploying GPU workloads.
- **ACR attachment**: prefer `az aks update --attach-acr` over `imagePullSecrets` when using AKS.

---

# AKS Cluster Deployment & Management

For full step-by-step instructions, read these files in order:

1. **`deployment/set_properties.sh`** – Fill in and source this first (`source deployment/set_properties.sh`). It defines all Azure and K8s environment variables.
2. **`deployment/README.md`** – Overview of all deployment options (AKS recommended, manual K8s, VM+Docker).
3. **`deployment/acr/README.md`** – ACR creation and image mirroring. Do this before AKS deployment.
4. **`deployment/aks/README.md`** – End-to-end AKS deployment: Bicep cluster creation, credentials, K8s prerequisites, StreamWise/StreamCast pod deployment, GPU node scaling, and troubleshooting.
5. **`deployment/helm/README.md`** – Helm-based deployment of GPU model microservices (alternative to StreamWise REST API).
6. **`deployment/aks/aks.bicep`** – Bicep template source; review for available parameters and GPU VM sizes.
