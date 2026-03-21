# Azure Kubernetes Service (AKS)

This guide provides step-by-step instructions for deploying StreamWise and StreamCast on Azure Kubernetes Service (AKS).

GitHub Copilot CLI can do the full deployment with a prompt like:
```bash
copilot -p "Deploy the full StreamWise stack on a new AKS cluster in swedencentral with 2 Spot H100 VMs. Use resource group aks-agent, ACR rtgen in acr RG, and HF token hf_XYZ. Reboot the GPU VMs after they come up." --allow-all
```

## Prerequisites

Before starting, ensure you have:
- Azure CLI installed and configured ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- kubectl installed (`az aks install-cli` or [Install Guide](https://kubernetes.io/docs/tasks/tools/))
- Azure Container Registry (ACR) created ([ACR Setup](../acr/README.md))
- Docker images built and pushed to ACR ([Build & Push Images](../README.md#building-and-pushing-docker-images))
- Hugging Face token ([Get Token](https://huggingface.co/settings/tokens))

Fill out configuration parameters in `set_properties.sh`
* Azure resource group and region
* ACR settings

To load the parameters to use them later:
```bash
source set_properties.sh
```

## Step 1: Deploy AKS Cluster

The Bicep template ([aks.bicep](aks.bicep)) provisions:
- An AKS cluster with a system node pool (for StreamWise, StreamCast, and system pods)
- A GPU spot node pool (starts at 0 nodes, scale up when needed)
- A static public IP (`aks-pods-public-ip`) for LoadBalancer services
- ACR attachment via role assignment

Review the Bicep parameters in [aks.bicep](aks.bicep) (cluster name, GPU VM size, ACR name), then deploy:

```bash
az login

# Pick a name for your Azure resource group and region
AZ_RESOURCE_GROUP="my-resource-group"
AZ_REGION="swedencentral"

az group create --name $AZ_RESOURCE_GROUP --location $AZ_REGION

az deployment group create \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --template-file deployment/aks/aks.bicep \
  --parameters \
    clusterName=my-cluster \
    gpuNodeVmSize=Standard_ND96isrf_H100_v5 \
    acrName=myacr \
    acrResourceGroup=my-acr-rg
```

Some available GPU VM sizes:
| VM Size | GPU |
|---------|-----|
| `Standard_NC96ads_A100_v4` | NVIDIA A100 |
| `Standard_ND96ams_A100_v4` | NVIDIA A100 (InfiniBand) |
| `Standard_ND96isrf_H100_v5` | NVIDIA H100 |

> **Note:** The cluster name defaults to `<resource-group>-cluster`.
> If the ACR role assignment fails (e.g. on redeployment), the cluster itself will still be created successfully.
> Attach the ACR manually with:
> `az aks update -g $AZ_RESOURCE_GROUP -n <cluster> --attach-acr <acrName>`

After deployment, retrieve the outputs and get cluster credentials:
```bash
AKS_CLUSTER=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.clusterName.value -o tsv)

IP_ADDRESS=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.publicIpAddress.value -o tsv)

echo "AKS cluster: $AKS_CLUSTER"
echo "Public IP:   $IP_ADDRESS"

az aks get-credentials --resource-group $AZ_RESOURCE_GROUP --name $AKS_CLUSTER
```

**Verify AKS deployment:**
```bash
kubectl get nodes
kubectl cluster-info
```

Expected output should show node(s) in `Ready` state.


## Step 2: Setup Kubernetes Prerequisites

### 2.1 Namespace
```bash
kubectl create namespace $K8S_NAMESPACE
```

### 2.2 Secrets
Configure the Hugging Face token (required for model access).
Make sure `HF_TOKEN` is set in [`set_properties.sh`](../set_properties.sh), then:
```bash
kubectl create secret generic hf-token -n $K8S_NAMESPACE --from-literal=token=$HF_TOKEN
```

> **Note:** When ACR is attached via the Bicep template or `az aks update --attach-acr`, images are pulled using the AKS managed identity.
> No `acr-secret` is needed.
> The `imagePullSecrets` entry in the pod YAMLs is a fallback for non-AKS clusters and can be safely ignored when ACR is attached.

### 2.3 Storage
Deploy persistent volumes and claims (used by GPU model services for caching):
```bash
kubectl apply -f deployment/aks/local-pv.yaml
kubectl apply -f deployment/aks/local-pvc.yaml -n $K8S_NAMESPACE
```

## Step 3: Deploy StreamWise (Cluster Manager)

Set the environment variables needed by the YAML templates, then deploy.

> **First pull:** The StreamWise and StreamCast images are ~9 GB each.
> The first pull may take 5–10 minutes.
> The pod will show `ContainerCreating` during this time.

```bash
source ../set_properties.sh  # provides ACR_URL, AZ_RESOURCE_GROUP, etc.
export LOAD_BALANCER_IP=$IP_ADDRESS
export RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP

cd deployment/aks

kubectl apply -f streamwise-service-account.yaml
envsubst < streamwise-pod.yaml | kubectl apply -f -
```

For Windows (PowerShell), you can use `Get-Content` and `-replace` instead of `envsubst`.

Verify StreamWise is running:
```bash
kubectl get pods -n rtgen
kubectl get svc -n rtgen
```

Get logs and log into the container:
```bash
kubectl exec -n rtgen streamwise -- cat /tmp/streamwise.log
kubectl exec -it -n rtgen streamwise -- /bin/bash
```

Open the web UI at: `http://$IP_ADDRESS:8081`

### Remove StreamWise
```bash
kubectl delete -f streamwise-pod.yaml
kubectl delete -f streamwise-service-account.yaml
```

## Step 4: Deploy StreamCast

Deploy using the same variable substitution approach as Step 3:

```bash
kubectl apply -f streamwiseapp-service-account.yaml
envsubst < streamcast-pod.yaml | kubectl apply -f -
```

Verify StreamCast is running:
```bash
kubectl get pods -n rtgen
kubectl get svc -n rtgen
```

Open the StreamCast UI at: `http://$IP_ADDRESS:8080`

### Remove StreamCast
```bash
kubectl delete -f streamcast-pod.yaml
kubectl delete -f streamwiseapp-service-account.yaml
```

## Step 5: GPU Setup

Install the NVIDIA device plugin so Kubernetes can schedule GPU workloads:
```bash
kubectl create namespace gpu-resources
kubectl apply -f nvidia-device-plugin-ds.yaml
```

Scale the GPU spot node pool up (it starts at 0 nodes):
```bash
az aks nodepool scale \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --name spoth100 \
  --node-count 1
```

If these fails, scale the VMSS directly:
```bash
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"
NODEPOOL_NAME=$(az aks nodepool list \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --query "[?starts_with(vmSize, 'Standard_N')].name | [0]" -o tsv)
VMSS_NAME=$(az vmss list \
  -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-${NODEPOOL_NAME}-')].name | [0]" -o tsv)
az vmss scale -g $MC_RESOURCE_GROUP -n $VMSS_NAME --new-capacity 1
```

Log into a node using `node-shell`:
```bash
kubectl get nodes
kubectl node-shell <node-name>
```

Sometimes the NVIDIA setup with torch is not correct and we need to restart the VM:
```bash
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"
NODEPOOL_NAME=$(az aks nodepool list \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --query "[?starts_with(vmSize, 'Standard_N')].name | [0]" -o tsv)
VMSS_NAME=$(az vmss list \
  -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-${NODEPOOL_NAME}-')].name | [0]" -o tsv)
INSTANCE_ID=0
az vmss restart -g $MC_RESOURCE_GROUP --name $VMSS_NAME --instance-ids $INSTANCE_ID
```

### 5.1 Partial GPU Support (MIG)

NVIDIA Multi-Instance GPU (MIG) partitions a single A100 or H100 GPU into smaller isolated slices, each with dedicated memory and compute resources. This lets lightweight models such as **Kokoro** (TTS) and **YOLO** (detection) share a physical GPU instead of occupying a whole one.

#### Enable MIG on the node

SSH into the GPU node (via `kubectl node-shell`) and run:
```bash
# Enable MIG mode on all GPUs (requires a node reboot or driver restart)
sudo nvidia-smi -i 0 -mig 1

# Verify MIG mode is on
nvidia-smi -L
```

#### Create MIG instances

Choose a profile that matches your workload.  Common profiles for A100 80 GB / H100 80 GB:

| Profile | GPU fraction | Memory |
|---------|-------------|--------|
| `1g.10gb` | 1/7 | 10 GB |
| `2g.20gb` | 2/7 | 20 GB |
| `3g.40gb` | 3/7 | 40 GB |
| `4g.40gb` | 4/7 | 40 GB |
| `7g.80gb` | 7/7 | 80 GB |

For A100 40 GB:

| Profile | GPU fraction | Memory |
|---------|-------------|--------|
| `1g.5gb` | 1/7 | 5 GB |
| `2g.10gb` | 2/7 | 10 GB |
| `3g.20gb` | 3/7 | 20 GB |
| `4g.20gb` | 4/7 | 20 GB |
| `7g.40gb` | 7/7 | 40 GB |

Create instances on the first GPU:
```bash
# Example: partition GPU 0 into 2× 3g.40gb + 1× 1g.10gb on an H100 80 GB
sudo nvidia-smi mig -cgi 3g.40gb,3g.40gb,1g.10gb -C -i 0
```

> **Tip:** A100/H100 also support a *mixed* GPU configuration where some GPUs on a node run in MIG mode and others run in standard (whole-GPU) mode.  Setting `migStrategy = "mixed"` in the device plugin (see below) tells Kubernetes to expose both whole-GPU (`nvidia.com/gpu`) and MIG-slice (`nvidia.com/mig-*`) resources from the same node, so MIG and non-MIG workloads can coexist in the cluster.

#### Configure the NVIDIA device plugin for MIG

Patch the device plugin ConfigMap to expose MIG instances as Kubernetes resources:

```bash
kubectl patch configmap nvidia-plugin-config -n gpu-resources --type merge -p '
{
  "data": {
    "config.toml": "[flags]\n  migStrategy = \"mixed\"\n"
  }
}'
# Restart the device plugin daemon set so it picks up the new config
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-resources
```

With `migStrategy = "mixed"`, the device plugin advertises each MIG instance as a separate Kubernetes resource, e.g. `nvidia.com/mig-1g.10gb`.

#### Deploy a service with a MIG slice

Using the StreamWise web UI, select a **MIG Profile** in the Resources section when adding a service.

Or via the REST API:
```bash
# Deploy Kokoro using a 1g.10gb MIG slice on an H100 node
curl -X POST "http://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "mig_profile=1g.10gb" \
  -d "gpu_type=h100" \
  -d "memory=8" \
  -d "cpu=2"

# Deploy YOLO using a 1g.5gb MIG slice on an A100 40 GB node
curl -X POST "http://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=yolo" \
  -d "gpu=1" \
  -d "mig_profile=1g.5gb" \
  -d "gpu_type=a100" \
  -d "memory=8" \
  -d "cpu=4"
```

> **Note:** When a `mig_profile` is specified the pod requests `nvidia.com/mig-<profile>` instead of `nvidia.com/gpu`.
> The `gpu` parameter then controls the number of MIG slices requested (usually `1`).
> Pods that request a MIG slice will only be scheduled on nodes where MIG mode is enabled and matching instances exist.

## Step 6: Deploy GPU Microservices

Deploy model services through the StreamWise web UI or REST API.

**GPU requirements per service:**
| Service | GPUs | MIG Profile (recommended) | Purpose |
|---------|------|--------------------------|---------|
| `gemma` | 2–4 | — (full GPUs) | LLM (screenplay generation) |
| `flux` | 2 | — (full GPUs) | Text-to-image |
| `hunyuanframepackf1` | 2 | — (full GPUs) | Image-to-video |
| `fantasytalking` | 2 | — (full GPUs) | Audio-driven video |
| `kokoro` | 1 | `1g.10gb` (H100) or `1g.5gb` (A100) | Text-to-speech |
| `yolo` | 1 | `1g.10gb` (H100) or `1g.5gb` (A100) | Character extraction |
| `realesrgan` | 1 | — | Video upscaling |
| `podcasttranscript` | 0 | — | Transcript orchestration (CPU-only) |

> **Capacity planning:** A single `Standard_ND96isrf_H100_v5` node provides 8 GPUs.
> A minimal StreamCast pipeline (gemma + kokoro + flux + yolo + hunyuanframepackf1) requires 8 GPUs.
> With MIG enabled, kokoro and yolo each consume only a 1/7 GPU slice, freeing whole GPUs for heavier models.
> For parallel execution of all services, add a second GPU node.

The Web UI is available at `http://$IP_ADDRESS:8081` to manage services.
Use the REST API to deploy all services at once:
```bash
curl -X POST "http://$IP_ADDRESS:8081/api/service"
```

Or deploy individual services with specific resource allocations:
```bash
# Deploy a single service (whole GPU)
curl -X POST "http://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "memory=8" \
  -d "cpu=2"

# Deploy with a MIG slice (partial GPU — requires MIG configured on the node; see Step 5.1)
curl -X POST "http://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "mig_profile=1g.10gb" \
  -d "gpu_type=h100" \
  -d "memory=8" \
  -d "cpu=2"

# Verify deployed services
curl "http://$IP_ADDRESS:8081/api/services"
```


## Troubleshooting

If pods fail to start:
```bash
kubectl describe pod <POD_NAME> -n rtgen
kubectl logs <POD_NAME> -n rtgen
kubectl get events -n rtgen --sort-by='.lastTimestamp'
```

Common issues:
- **Image pull errors**: Verify ACR is attached to AKS (`az aks check-acr -g $AZ_RESOURCE_GROUP -n $AKS_CLUSTER --acr <acrName>`)
- **Pods stuck in Pending (Insufficient cpu)**: The system node pool doesn't have enough CPU. Scale up with `az aks nodepool scale` or use a larger VM size (see Sizing note in Step 1)
- **GPU not available**: Ensure the GPU node pool is scaled up and the NVIDIA device plugin is running
- **LoadBalancer stuck in Pending**: Verify the public IP exists (`az network public-ip show -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip`) and the AKS identity has Network Contributor role on the resource group
- **Secret errors**: Verify HF token is correctly configured with `kubectl get secret hf-token -n rtgen`
- **ACR role assignment fails on redeployment**: The role already exists. Attach ACR manually: `az aks update -g $AZ_RESOURCE_GROUP -n $AKS_CLUSTER --attach-acr <acrName>`

## Cleanup

```bash
# Delete pods and services
kubectl delete -f streamcast-pod.yaml
kubectl delete -f streamwiseapp-service-account.yaml
kubectl delete -f streamwise-pod.yaml
kubectl delete -f streamwise-service-account.yaml

# Delete storage and namespace
kubectl delete pvc local-pvc -n rtgen
kubectl delete pv local-pv
kubectl delete namespace rtgen

# Delete entire resource group (includes AKS cluster and all resources)
az group delete --name $AZ_RESOURCE_GROUP --yes --no-wait
```