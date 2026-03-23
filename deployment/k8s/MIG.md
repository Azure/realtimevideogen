# NVIDIA Multi-Instance GPU (MIG) Setup

MIG partitions a single A100 or H100 GPU into smaller isolated slices, each with dedicated memory and compute resources.
This lets lightweight models such as **Kokoro** (TTS) and **YOLO** (image detection) share a physical GPU instead of occupying a whole one.

---

## Recommended layout: 7 full GPUs + 1 MIG-partitioned GPU

On an 8-GPU node the recommended setup is to keep **7 GPUs in standard (full) mode** for heavy models (Wan, Flux, Gemma, etc.) and configure **1 GPU in MIG mode** for lightweight services (Kokoro, YOLO, etc.).

### 1. Enable MIG on a single GPU

SSH into the GPU node (on AKS, use `kubectl node-shell <node-name>`) and enable MIG on GPU 7 only, leaving GPUs 0–6 as full GPUs:

```bash
# Enable MIG mode on GPU 7 only (requires a GPU reset or node reboot)
sudo nvidia-smi -i 7 -mig 1

# Verify MIG mode is on for GPU 7
nvidia-smi -L
```

### 2. Create MIG instances

Create MIG instances on GPU 7.
Choose the profile set that matches your GPU memory:

**A100/H100 80 GB** — 2 × `2g.20gb` + 3 × `1g.10gb` (uses all 7 slices: 2×2 + 3×1 = 7):
```bash
sudo nvidia-smi mig -cgi 2g.20gb,2g.20gb,1g.10gb,1g.10gb,1g.10gb -C -i 7
```

**A100 40 GB** — 2 × `2g.10gb` + 3 × `1g.5gb` (uses all 7 slices: 2×2 + 3×1 = 7):
```bash
sudo nvidia-smi mig -cgi 2g.10gb,2g.10gb,1g.5gb,1g.5gb,1g.5gb -C -i 7
```

> **Tip:** The profiles above fill the entire GPU (7/7 slices).
> Lightweight services like Kokoro (TTS) or YOLO (detection) fit in a `1g` slice.
> The two `2g` slices can run slightly heavier workloads, e.g. RealESRGAN upscaling.

### 3. Configure the NVIDIA device plugin for MIG

Apply the ConfigMap ([nvidia-plugin-mig-config.yaml](nvidia-plugin-mig-config.yaml)) and restart the device plugin:
```bash
kubectl apply -f deployment/k8s/nvidia-plugin-mig-config.yaml
# Restart the device plugin so it picks up the new config
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-resources
```

With `migStrategy = "mixed"`, the device plugin advertises **both** whole-GPU resources (`nvidia.com/gpu`) for the 7 standard GPUs and per-slice resources (`nvidia.com/mig-<profile>`) for the MIG GPU, so both types of workloads can coexist on the same node.

### 4. Deploy a service with a MIG slice

Using the StreamWise web UI, select a **MIG Profile** in the Resources section when adding a service.

Or via the REST API:
```bash
# Deploy Kokoro using a 1g.10gb MIG slice on an H100 node
curl -X POST "http://$LOAD_BALANCER_IP:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "mig_profile=1g.10gb" \
  -d "gpu_type=h100" \
  -d "memory=8" \
  -d "cpu=2"

# Deploy YOLO using a 1g.5gb MIG slice on an A100 40 GB node
curl -X POST "http://$LOAD_BALANCER_IP:8081/api/pod" \
  -d "container_name=yolo" \
  -d "gpu=1" \
  -d "mig_profile=1g.5gb" \
  -d "gpu_type=a100" \
  -d "memory=8" \
  -d "cpu=4"
```

> **Note:** When specifying a `mig_profile`, the pod requests `nvidia.com/mig-<profile>` instead of `nvidia.com/gpu`.
> The `gpu` parameter controls the number of MIG slices requested (usually `1`).
> Pods that request a MIG slice will only be scheduled on nodes where MIG mode is enabled and matching instances exist.

---

## AKS-specific: automatic MIG via node pool

For A100 v4 series VMs (e.g., `Standard_ND96amsr_A100_v4`), AKS can configure MIG partitioning automatically at node-pool creation time.
See the [Azure AKS GPU multi-instance documentation](https://learn.microsoft.com/en-us/azure/aks/gpu-multi-instance) for full details and supported VM sizes.

```bash
# Create a dedicated node pool where every A100 GPU is split into 1g.5gb slices
az aks nodepool add \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --name migpool \
  --node-count 1 \
  --node-vm-size Standard_ND96amsr_A100_v4 \
  --gpu-instance-profile MIG1g
```

| `--gpu-instance-profile` value | Profile | GPU fraction | Memory |
|-------------------------------|---------|-------------|--------|
| `MIG1g` | `1g.5gb` | 1/7 | 5 GB |
| `MIG2g` | `2g.10gb` | 2/7 | 10 GB |
| `MIG3g` | `3g.20gb` | 3/7 | 20 GB |
| `MIG4g` | `4g.20gb` | 4/7 | 20 GB |
| `MIG7g` | `7g.40gb` | 7/7 | 40 GB |

> **Note:** `--gpu-instance-profile` applies one uniform partition strategy across every GPU in the node pool.
> It is set at pool creation time and cannot be changed afterwards.
> Mix MIG and non-MIG workloads by creating separate node pools.

---

## MIG profile reference

### A100 80 GB / H100 80 GB

| Profile | GPU fraction | Memory |
|---------|-------------|--------|
| `1g.10gb` | 1/7 | 10 GB |
| `2g.20gb` | 2/7 | 20 GB |
| `3g.40gb` | 3/7 | 40 GB |
| `4g.40gb` | 4/7 | 40 GB |
| `7g.80gb` | 7/7 | 80 GB (full GPU) |

### A100 40 GB

| Profile | GPU fraction | Memory |
|---------|-------------|--------|
| `1g.5gb` | 1/7 | 5 GB |
| `2g.10gb` | 2/7 | 10 GB |
| `3g.20gb` | 3/7 | 20 GB |
| `4g.20gb` | 4/7 | 20 GB |
| `7g.40gb` | 7/7 | 40 GB (full GPU) |
