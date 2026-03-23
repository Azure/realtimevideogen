# NVIDIA Multi-Instance GPU (MIG) Setup

MIG partitions a single A100 or H100 GPU into smaller isolated slices, each with dedicated memory and compute resources.
This lets lightweight models such as **Kokoro** (TTS) and **YOLO** (image detection) share a physical GPU instead of occupying a whole one.

---

## Recommended layout: 7 full GPUs + 1 MIG-partitioned GPU

On an 8-GPU node the recommended setup is to keep **7 GPUs in standard (full) mode** for heavy models (Wan, Flux, Gemma, etc.) and configure **1 GPU in MIG mode** for lightweight services (Kokoro, YOLO, etc.).

### Prerequisites

Before starting MIG setup you will need:

- `kubectl` access to the cluster.
- The node must already be running (GPU spot node pool scaled up) and showing `Ready`.
- The NVIDIA device plugin DaemonSet deployed in the `gpu-resources` namespace
  (see [AKS GPU Setup](../aks/README.md#step-5-gpu-setup)).

Throughout this guide, replace `<gpu-node>` with the actual node name
(e.g. `aks-spota100-27824666-vmss000000`).

### 1. Create a privileged debug pod on the GPU node

You need host-level access to run `nvidia-smi` commands.
Create a privileged pod pinned to the target node:

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-debug
  namespace: default
spec:
  nodeName: <gpu-node>
  hostPID: true
  tolerations:
  - operator: Exists
  containers:
  - name: debug
    image: mcr.microsoft.com/cbl-mariner/base/core:2.0
    command: ["sleep", "3600"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: host
      mountPath: /host
  volumes:
  - name: host
    hostPath:
      path: /
  restartPolicy: Never
EOF
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=60s
```

All `nvidia-smi` commands below are run through this pod:
```bash
kubectl exec gpu-debug -- chroot /host <command>
```

### 2. Stop the NVIDIA device plugin on the target node

The device plugin holds open handles on the GPU devices.
It must be stopped before any GPU reset or MIG mode change.

Because the device plugin runs as a DaemonSet, deleting its pod is not enough — the DaemonSet controller will recreate it immediately.
Apply a `NoExecute` taint to the node so the DaemonSet pod is evicted and cannot be rescheduled:

```bash
kubectl taint node <gpu-node> mig-setup=true:NoExecute
```

Verify the device plugin pod on this node is gone:
```bash
kubectl -n gpu-resources get pods -o wide | grep <gpu-node>
# Should return nothing
```

> **Important:** This taint will also evict any other pods on the node that do not tolerate it.
> The debug pod tolerates all taints (`operator: Exists`), so it stays.

### 3. Enable MIG mode on GPU 7

```bash
kubectl exec gpu-debug -- chroot /host nvidia-smi -i 7 -mig 1
```

This puts MIG into **pending** state.
Verify:
```bash
kubectl exec gpu-debug -- chroot /host \
  nvidia-smi --query-gpu=index,mig.mode.current,mig.mode.pending --format=csv
```
GPU 7 should show `Disabled, Enabled` (current=Disabled, pending=Enabled).

### 4. Restart the VM to activate MIG

On **NVSwitch-based SXM4 systems** (A100-SXM4, H100-SXM4), all GPUs are interconnected via NVLink and must be reset together.
A simple `nvidia-smi -r` will fail because the Fabric Manager and kernel modules hold references.
**The most reliable way to activate MIG is a VM restart.**

For AKS, restart the VMSS instance:
```bash
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"
# Find the VMSS name for the GPU node pool
VMSS_NAME=$(az vmss list -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'spota100') || contains(name, 'spoth100')].name | [0]" -o tsv)

# The instance ID is the number at the end of the node name (e.g. vmss000000 → 0, vmss000004 → 4)
INSTANCE_ID=0
az vmss restart -g $MC_RESOURCE_GROUP --name $VMSS_NAME --instance-ids $INSTANCE_ID
```

Wait for the node to come back:
```bash
kubectl wait --for=condition=Ready node/<gpu-node> --timeout=300s
```

> **For PCIe-based systems** (e.g. `Standard_NC96ads_A100_v4`), GPUs are independent.
> You may be able to skip the VM restart by stopping the Fabric Manager (`systemctl stop nvidia-fabricmanager`) and resetting the GPU (`nvidia-smi -i 7 -r`).
> If that fails, fall back to a VM restart.

### 5. Verify MIG is active and create instances

Recreate the debug pod (it was destroyed by the VM restart):
```bash
# Re-apply the debug pod YAML from Step 1, then:
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=60s
```

Verify GPU 7 now shows MIG `Enabled, Enabled`:
```bash
kubectl exec gpu-debug -- chroot /host \
  nvidia-smi --query-gpu=index,mig.mode.current,mig.mode.pending --format=csv
```

Create MIG instances on GPU 7.
Choose the profile set that matches your GPU memory:

**A100/H100 80 GB** — 2 × `2g.20gb` + 3 × `1g.10gb` (uses all 7 slices: 2×2 + 3×1 = 7):
```bash
kubectl exec gpu-debug -- chroot /host \
  nvidia-smi mig -cgi 2g.20gb,2g.20gb,1g.10gb,1g.10gb,1g.10gb -C -i 7
```

**A100 40 GB** — 2 × `2g.10gb` + 3 × `1g.5gb` (uses all 7 slices: 2×2 + 3×1 = 7):
```bash
kubectl exec gpu-debug -- chroot /host \
  nvidia-smi mig -cgi 2g.10gb,2g.10gb,1g.5gb,1g.5gb,1g.5gb -C -i 7
```

Verify the MIG instances are created:
```bash
kubectl exec gpu-debug -- chroot /host nvidia-smi -L
```
GPU 7 should list the five MIG devices under it.

> **Tip:** The profiles above fill the entire GPU (7/7 slices).
> Lightweight services like Kokoro (TTS) or YOLO (detection) fit in a `1g` slice.
> The two `2g` slices can run slightly heavier workloads, e.g. RealESRGAN upscaling.

### 6. Configure the NVIDIA device plugin for MIG

Apply the ConfigMap ([nvidia-plugin-mig-config.yaml](nvidia-plugin-mig-config.yaml)):
```bash
kubectl apply -f deployment/k8s/nvidia-plugin-mig-config.yaml
```

The device plugin DaemonSet ([nvidia-device-plugin-ds.yaml](nvidia-device-plugin-ds.yaml)) must include:
- **`MIG_STRATEGY=mixed`** environment variable — tells the plugin to advertise both `nvidia.com/gpu` (for full GPUs) and `nvidia.com/mig-<profile>` (for MIG slices).
- **`privileged: true`** security context — required for the NVML calls that enumerate MIG devices. Without this the plugin fails with *"Insufficient Permissions"*.

These are already set in the checked-in YAML. Apply it:
```bash
kubectl apply -f deployment/k8s/nvidia-device-plugin-ds.yaml
```

### 7. Remove the taint and verify

Remove the setup taint so the device plugin (and other pods) can schedule on the node again:
```bash
kubectl taint node <gpu-node> mig-setup=true:NoExecute-
```

Wait ~15 seconds for the device plugin to start and register resources, then verify:
```bash
kubectl get node <gpu-node> -o json | python3 -c "
import json, sys
node = json.load(sys.stdin)
cap = node['status']['capacity']
alloc = node['status']['allocatable']
keys = sorted(set(k for k in list(cap) + list(alloc) if 'nvidia' in k))
print('Resource                        Capacity  Allocatable')
print('-' * 60)
for k in keys:
    print(f'{k:<32} {cap.get(k,\"-\"):>8}  {alloc.get(k,\"-\"):>11}')
"
```

Expected output for an 80 GB A100 node:
```
Resource                        Capacity  Allocatable
------------------------------------------------------------
nvidia.com/gpu                          7            7
nvidia.com/mig-1g.10gb                  3            3
nvidia.com/mig-2g.20gb                  2            2
```

### 8. Clean up

Delete the debug pod:
```bash
kubectl delete pod gpu-debug --ignore-not-found
```

### 9. Deploy a service with a MIG slice

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
