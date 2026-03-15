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
| `tests/` | pytest test suites: root-level utils, `tests/simulator/`, `tests/streamwise/`, `tests/streamwise_app/`. |

## Key conventions

- **Namespace**: all K8s resources go in namespace `rtgen` (set via `$K8S_NAMESPACE`).
- **Docker images**: tagged as `<ACR_URL>/<model>:<tag>`, tags come from `services.json`.
- **Pod YAMLs** in `deployment/aks/` use shell variable placeholders (`${ACR_URL}`, `${LOAD_BALANCER_IP}`, `${RESOURCE_GROUP_NAME}`). Deploy with `envsubst < file.yaml | kubectl apply -f -`.
- **Helm chart** in `deployment/helm/` reads image tags from `services.json` via the `deploy.sh` script.
- **GPU spot node pools** start at 0 nodes to save cost; scale up explicitly before deploying GPU workloads.
- **ACR attachment**: prefer `az aks update --attach-acr` over `imagePullSecrets` when using AKS.

---

# Development Guide

## Python coding standards

- **Python version**: 3.12.
- **Type annotations**: required on every function signature (`disallow_untyped_defs = True` in `mypy.ini`). Never use bare `Any` without justification.
- **Line length**: maximum 120 characters (enforced by flake8).
- **Async**: use `async`/`await` and `aiofiles` for all I/O-bound work. The web framework is Quart (async Flask).
- **Logging**: use the `colorlog`-based logger from `console_utils.py`; pass `extra={"markup": True}` where color markup is needed. Do not use f-strings in logging calls (W1203 is suppressed but plain `%`-style or structured logging is preferred).
- **Imports**: stdlib → third-party → local. Module-level imports not at the top of the file are allowed (E402 suppressed in `.flake8`).
- **No secrets in source**: never hard-code tokens, passwords, or subscription IDs. Use environment variables or Azure Key Vault references.

## Linting commands

Run these before opening a pull request:

```bash
# Python style
flake8 . tests

# Shell scripts
shellcheck $(git ls-files '*.sh')
shellcheck $(git ls-files '*.bash')

# Static type checking (install stubs on first run)
mypy --install-types --non-interactive --ignore-missing-imports .

# YAML
yamllint -c .yamllint.yml $(git ls-files '*.yml' '*.yaml')

# JSON
for f in $(git ls-files '*.json'); do python -m json.tool "$f" > /dev/null && echo "OK: $f"; done

# Jinja/HTML templates (|| true: j2lint exits non-zero on warnings; failures are informational)
j2lint streamwise/templates/ --extensions html || true
j2lint apps/*/templates/ --extensions html || true
```

## Testing

Tests use **pytest** with coverage. Run the relevant suite(s) after making changes:

```bash
# All root-level utility and wrapper tests (fastest, no GPU required)
pytest --ignore=tests/simulator --ignore=tests/streamwise --ignore=tests/streamwise_app -vv

# Simulator tests
pytest tests/simulator/ -vv

# StreamWise core tests
pytest tests/streamwise/ -vv

# Individual StreamWise app tests (e.g. StreamCast)
pytest tests/streamwise_app/test_streamcast.py -vv

# Full coverage report
coverage combine --keep .coverage.*
coverage xml -o coverage.xml --ignore-errors
```

Add new tests in the matching `tests/` sub-directory. Mirror the mock pattern used in `tests/` (e.g. `k8s_mock.py`, `torch_mock.py`) to avoid GPU/cloud dependencies in unit tests.

## Must / Never rules

- **Must**: add or update tests for every functional change.
- **Must**: keep `services.json` valid JSON; validate with `python -m json.tool services.json`.
- **Must**: run `flake8` and `mypy` before committing.
- **Never**: commit Azure credentials, HuggingFace tokens, or other secrets.
- **Never**: modify files under `deployment/` without reading `deployment/README.md` first.
- **Never**: change the K8s namespace away from `rtgen` without updating every reference.
- **Never**: remove or weaken existing tests.

---

# AKS Cluster Deployment & Management

For full step-by-step instructions, read these files in order:

1. **`deployment/set_properties.sh`** – Fill in and source this first (`source deployment/set_properties.sh`). It defines all Azure and K8s environment variables.
2. **`deployment/README.md`** – Overview of all deployment options (AKS recommended, manual K8s, VM+Docker).
3. **`deployment/acr/README.md`** – ACR creation and image mirroring. Do this before AKS deployment.
4. **`deployment/aks/README.md`** – End-to-end AKS deployment: Bicep cluster creation, credentials, K8s prerequisites, StreamWise/StreamCast pod deployment, GPU node scaling, and troubleshooting.
5. **`deployment/helm/README.md`** – Helm-based deployment of GPU model microservices (alternative to StreamWise REST API).
6. **`deployment/aks/aks.bicep`** – Bicep template source; review for available parameters and GPU VM sizes.
