# StreamWise Demo: End-to-End AKS Deployment with GPU Spot Probing

This document summarizes the full deployment walkthrough performed on 2026-06-05,
from capacity probing through to a running StreamWise instance on AKS with 32 H100 GPUs.

## Overview

| Step | Tool/Skill | Outcome |
|------|-----------|---------|
| 1. SKU Discovery | `az vm list-skus` | Found `Standard_ND96isrf_H100_v5` unrestricted in eastus2 and SwedenCentral |
| 2. Capacity Pre-Check | CCC Kusto query | SwedenCentral AZ03 has 6 allocable Spot VMs — enough for 4 nodes |
| 3. AKS Deployment | Bicep (`aks.bicep`) | Cluster + 4 H100 Spot nodes + networking provisioned in ~12 min |
| 4. K8s Setup | kubectl | Namespace, secrets, PV/PVC, NVIDIA device plugin |
| 5. StreamWise Deploy | kubectl + YAML templating | Pod running, web UI accessible at public IP:8081 |

## Step 1: GPU SKU Discovery (`azure-gpu-spot-probe` skill)

### What We Learned

The `azure-gpu-spot-probe` skill provides a structured approach to finding GPU Spot capacity:

1. **List available SKUs** with `az vm list-skus --size H100` to find all H100 variants and their restriction status.
2. **Cross-reference with CCC data** (Kusto query on `onecapacityfollower.centralus.kusto.windows.net`) to see actual allocable Spot VMs per region/zone.
3. **Key insight:** CCC data is fleet-wide, not per-subscription. A region may show capacity but still be `Location`-restricted for your subscription.

### H100 SKU Variants

| SKU | Key Difference |
|-----|---------------|
| `Standard_ND96isr_H100_v5` | 8× H100, InfiniBand |
| `Standard_ND96isrf_H100_v5` | 8× H100, InfiniBand (refresh/newer) |
| `Standard_ND96is_H100_v5` | 8× H100, no InfiniBand suffix |
| `Standard_ND96is_noIB_H100_v5` | 8× H100, explicitly no InfiniBand |

### Subscription Access Results

For `Standard_ND96isrf_H100_v5`:
- **Unrestricted:** eastus2 (zones 1,2), SwedenCentral (zones 1,2,3)
- **Location-blocked:** eastus, centralus, northcentralus

### CCC Capacity Data

```
Region          | SKU                          | AllocableSpotVMs
swedencentral   | Standard_ND96isrf_H100_v5    | 6
eastus          | Standard_ND96isr_H100_v5     | 16 (but sub is blocked)
centralus       | Standard_ND96isr_H100_v5     | 83 (but sub is blocked)
```

**Decision:** SwedenCentral — 6 allocable Spot VMs, subscription unrestricted, zones 1/2/3.

## Step 2: AKS Cluster Deployment

### Bicep Template (`deployment/aks/aks.bicep`)

The template provisions:
- **System node pool:** Standard_D16s_v5 (1 node) for StreamWise/StreamCast/system pods
- **GPU Spot node pool (`spoth100`):** Full-GPU nodes for heavy models
- **GPU MIG Spot node pool (`spoth100mig`):** For mixed-mode (7 full GPUs + 1 MIG-partitioned)
- **Networking:** Static public IP, NAT gateway, NSG (ports 8000–9000), VNet with disabled default outbound

### Deployment Command

```bash
az deployment group create \
  --name AKSDeployment \
  --resource-group hqiu-streamwise-aks-cluster \
  --template-file deployment/aks/aks.bicep \
  --parameters \
    clusterName=hqiu-streamwise-aks-cluster-cluster \
    gpuNodeVmSize=Standard_ND96isrf_H100_v5 \
    gpuNodePoolName=spoth100 \
    gpuMigNodePoolName=spoth100mig \
    gpuNodeCount=4 \
    acrName=inigogrtgen \
    acrResourceGroup=inigog-acr
```

### Gotcha: ACR Role Assignment Failure

The Bicep template includes a cross-resource-group ACR role assignment. If the role already exists
(e.g., from a prior deployment), it fails with `RoleAssignmentUpdateNotPermitted`. The cluster
itself still succeeds — just attach ACR manually:

```bash
az aks update -g <rg> -n <cluster> --attach-acr <acrName>
```

### Gotcha: ACR Login Server Name

ACR names like `inigogrtgen` may have a login server of `inigogrtgen-<hash>.azurecr.io`, NOT
`inigogrtgen.azurecr.io`. Always verify with:

```bash
az acr show --name <acr> --query loginServer -o tsv
```

## Step 3: Kubernetes Setup

```bash
kubectl create namespace rtgen
kubectl create secret generic hf-token -n rtgen --from-literal=token=$HF_TOKEN
kubectl apply -f deployment/k8s/local-pv.yaml
kubectl apply -f deployment/k8s/local-pvc.yaml -n rtgen
kubectl create namespace gpu-resources
kubectl apply -f deployment/k8s/nvidia-device-plugin-ds.yaml
```

## Step 4: StreamWise Deployment

The `streamwise-pod.yaml` uses shell variable placeholders (`${ACR_URL}`, `${LOAD_BALANCER_IP}`,
`${RESOURCE_GROUP_NAME}`). On Linux use `envsubst`; on Windows use PowerShell string replacement:

```powershell
$yaml = Get-Content "deployment/aks/streamwise-pod.yaml" -Raw
$yaml = $yaml -replace '\$\{ACR_URL\}', 'inigogrtgen-cjd9f3dydte2bzbb.azurecr.io'
$yaml = $yaml -replace '\$\{RESOURCE_GROUP_NAME\}', 'hqiu-streamwise-aks-cluster'
$yaml = $yaml -replace '\$\{LOAD_BALANCER_IP\}', '4.223.71.250'
$yaml | kubectl apply -f -
```

### First Pull Time

The StreamWise image is ~9 GB. First pull takes 5–10 minutes (pod shows `ContainerCreating`).

## Step 5: Verification

```bash
kubectl get pods -n rtgen       # Should show 1/1 Running
kubectl get svc -n rtgen        # Should show LoadBalancer with external IP
curl http://<IP>:8081/          # Should return HTTP 200
```

## Final Result

| Property | Value |
|----------|-------|
| Cluster | `hqiu-streamwise-aks-cluster-cluster` |
| Region | SwedenCentral |
| GPU Nodes | 4 × Standard_ND96isrf_H100_v5 (32 H100 GPUs) |
| Public IP | 4.223.71.250 |
| StreamWise URL | http://4.223.71.250:8081 |
| FQDN | http://streamwise-fnv3ci.swedencentral.cloudapp.azure.com:8081 |

## Auto-Deployment Feature

From the StreamWise web UI at port 8081, you can deploy all GPU model services with one click,
or via REST API:

```bash
# Deploy all services
curl -X POST "http://4.223.71.250:8081/api/service"

# Deploy individual services
curl -X POST "http://4.223.71.250:8081/api/pod" \
  -d "container_name=kokoro" -d "gpu=1" -d "memory=8" -d "cpu=2"

# List deployed services
curl "http://4.223.71.250:8081/api/services"
```

## Step 6: Rebuilding the Image for New Features

When the deployed image is stale (e.g., missing the auto-deploy feature from a newer branch),
rebuild and push from a local Docker install or use ACR Tasks.

### Local Docker Build (recommended on Windows)

```powershell
# Prepare build context (see deployment/setup_image.sh for the full script)
# Fix CRLF line endings in bash files before building on Windows:
$content = [System.IO.File]::ReadAllText("deployment\streamwise\docker_files\run_httpserver.bash")
$content = $content -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("deployment\streamwise\docker_files\run_httpserver.bash", $content, [System.Text.UTF8Encoding]::new($false))

# Build and push
docker buildx build --platform linux/amd64 `
  --build-arg DOCKER_REPO=inigogrtgen-cjd9f3dydte2bzbb.azurecr.io `
  --build-arg BASE_TAG=v0.5.0 `
  -t "inigogrtgen-cjd9f3dydte2bzbb.azurecr.io/streamwise:v0.6.2-autodeploy" `
  "deployment\streamwise\docker_files" --push
```

### ACR Cloud Build (broken on Windows due to Unicode encoding)

`az acr build` streams build logs through the Azure CLI, which crashes on Windows cp1252
terminals when pip outputs Unicode progress bars. Use local Docker build instead.

### Redeploy the Pod

```powershell
kubectl delete pod streamwise -n rtgen --force --grace-period=0
# Re-apply YAML with updated image tag
```

### Gotcha: CRLF Line Endings in Bash Scripts

When building Docker images on Windows, `COPY *.bash .` preserves CRLF line endings.
Linux containers then fail with `$'\r': command not found`. **Always convert to LF** before
building, or add a `.dockerignore`/`dos2unix` step.

### Gotcha: Simulator Data Path in Docker

The Dockerfile copies the simulator into `/streamwise/simulator/`, but `allocator_bridge.py`
originally resolved the data path as `os.path.dirname(__file__) + "/../simulator/data"` which
evaluates to `/simulator/data/` (doesn't exist). Fixed by changing to:

```python
default_path = os.path.join(os.path.dirname(__file__), "simulator", "data")
```

This works in Docker (where cwd = `/streamwise/`) and locally (where `__file__` is in `streamwise/`).

## Key Lessons Learned

1. **Always check CCC capacity before probing** — saves time and money on doomed create attempts.
2. **CCC ≠ subscription access** — fleet capacity doesn't mean your subscription can use it.
3. **ACR login servers may have hashed suffixes** — always `az acr show --query loginServer`.
4. **ACR role assignments are idempotent-ish** — redeployments fail but don't break the cluster.
5. **Spot VMs can be evicted** — plan for re-scaling and MIG reconfiguration after eviction.
6. **Image pulls are slow** — ~9 GB images take 5–10 min on first pull; be patient.
7. **Zone mapping is opaque** — CCC AZ03 doesn't necessarily mean subscription zone 3.
8. **CRLF kills Linux containers** — always fix line endings when building on Windows.
9. **ACR cloud build broken on Windows** — use local Docker build + push instead.
10. **Relative paths shift in Docker** — verify path assumptions match the container layout (`WORKDIR` + `COPY` targets).
