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

```bash
# Create your personal config from the template (only needed once)
cp deployment/set_properties.sh.template deployment/set_properties.sh
# Edit the file and fill in your values (AZ_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP, ACR_NAME, HF_TOKEN, …)
```

To load the parameters to use them later:
```bash
source deployment/set_properties.sh
```

## Step 1: Deploy AKS Cluster

The Bicep template ([aks.bicep](aks.bicep)) provisions:
- An AKS cluster with a system node pool (for StreamWise, StreamCast, and system pods)
- A **full-GPU spot node pool** (e.g., `spoth100`): all GPUs in standard mode; for heavy models (Wan, Flux, Gemma, etc.) — starts at 0 nodes
- A **MIG spot node pool** (e.g., `spoth100mig`): same VM size but designated for mixed-mode use — 7 standard GPUs + 1 MIG-partitioned GPU for lightweight services — starts at 0 nodes
- A static public IP (`aks-pods-public-ip`) for LoadBalancer services
- A Network Security Group (`aks-node-subnet-nsg`) allowing inbound TCP on ports 8000–9000, attached to the node subnet so that LoadBalancer services are reachable from the Internet
- ACR attachment via role assignment

Separate node pools are needed because MIG mode is configured per node: putting the MIG node in its own pool prevents MIG changes on one VMSS instance from affecting the full-GPU instances.

Review the Bicep parameters in [aks.bicep](aks.bicep) (cluster name, GPU VM size, ACR name), then deploy:

```bash
az login

# If you haven't already, source your config (provides AZ_RESOURCE_GROUP, AZ_REGION, ACR_NAME, etc.)
source deployment/set_properties.sh

az group create --name $AZ_RESOURCE_GROUP --location $AZ_REGION

az deployment group create \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --template-file deployment/aks/aks.bicep \
  --parameters \
    clusterName=my-cluster \
    gpuNodeVmSize=Standard_ND96isrf_H100_v5 \
    gpuNodePoolName=spoth100 \
    gpuMigNodePoolName=spoth100mig \
    acrName=$ACR_NAME \
    acrResourceGroup=$AZ_RESOURCE_GROUP  # set to ACR resource group if different
```

Some available GPU VM sizes:

| VM Size | GPU |
|---------|-----|
| `Standard_NC96ads_A100_v4` | NVIDIA A100 |
| `Standard_ND96ams_A100_v4` | NVIDIA A100 (InfiniBand) |
| `Standard_ND96isrf_H100_v5` | NVIDIA H100 |
| `Standard_ND96isr_H200_v5` | NVIDIA H200 |
| `Standard_ND128isr_NDR_GB200_v6`| NVIDIA GB200 |
| `Standard_ND128isr_GB300_v6`| NVIDIA GB300 |

> **Note:** The cluster name defaults to `<resource-group>-cluster`.
> If the ACR role assignment fails (e.g. on redeployment), the cluster itself will still be created successfully.
> Attach the ACR manually with:
> `az aks update -g $AZ_RESOURCE_GROUP -n <cluster> --attach-acr <acrName>`
>
> **Note:** HTTPS/TLS is **disabled by default**. To enable Key Vault, the self-signed TLS certificate, OIDC issuer, workload identity, and federated identity credentials, add `enableSecureSetup=true` to the deployment parameters:
> ```bash
> az deployment group create \
>   --name AKSDeployment \
>   --resource-group $AZ_RESOURCE_GROUP \
>   --template-file deployment/aks/aks.bicep \
>   --parameters \
>     clusterName=my-cluster \
>     acrName=$ACR_NAME \
>     enableSecureSetup=true
> ```
> See [deployment/k8s/certs.md](../k8s/certs.md) for the follow-up steps to deploy the `SecretProviderClass` and verify HTTPS.

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

KEY_VAULT_NAME=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.keyVaultName.value -o tsv)

CSI_ADDON_CLIENT_ID=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.csiAddonClientId.value -o tsv)

AZ_TENANT_ID=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.tenantId.value -o tsv)

TLS_CERT_NAME=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.tlsCertificateName.value -o tsv)

echo "AKS cluster:           $AKS_CLUSTER"
echo "Public IP:             $IP_ADDRESS"
echo "Key Vault:             $KEY_VAULT_NAME"
echo "CSI addon client ID:   $CSI_ADDON_CLIENT_ID"
echo "Tenant ID:             $AZ_TENANT_ID"
echo "TLS cert name:         $TLS_CERT_NAME"

az aks get-credentials --resource-group $AZ_RESOURCE_GROUP --name $AKS_CLUSTER
```

**Verify AKS deployment:**
```bash
kubectl get nodes
kubectl cluster-info
```

Expected output should show node(s) in `Ready` state.


## Step 2: Setup Kubernetes Prerequisites

For the generic Kubernetes setup steps (namespace, storage, secrets), see the [Generic Kubernetes Setup guide](../k8s/README.md).

Quick reference for AKS:

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
kubectl apply -f deployment/k8s/local-pv.yaml
kubectl apply -f deployment/k8s/local-pvc.yaml -n $K8S_NAMESPACE
```

### 2.4 HTTPS / TLS Certificates

The Bicep template provisions an Azure Key Vault and generates a self-signed TLS certificate.
See [HTTPS / TLS Certificates](../k8s/certs.md) for the full setup guide.

## Step 3: Deploy StreamWise (Cluster Manager)

Set the environment variables needed by the YAML templates, then deploy.

> **First pull:** The StreamWise and StreamCast images are ~9 GB each.
> The first pull may take 5–10 minutes.
> The pod will show `ContainerCreating` during this time.

```bash
# If not already sourced (provides ACR_URL, AZ_RESOURCE_GROUP, etc.)
source deployment/set_properties.sh
export LOAD_BALANCER_IP=$IP_ADDRESS
export RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP

kubectl apply -f deployment/k8s/streamwise-service-account.yaml
envsubst < deployment/aks/streamwise-pod.yaml | kubectl apply -f -
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

Open the web UI at: `https://$IP_ADDRESS:8081`

### Remove StreamWise
```bash
kubectl delete -f deployment/aks/streamwise-pod.yaml
kubectl delete -f deployment/k8s/streamwise-service-account.yaml
```

## Step 4: Deploy StreamCast

Deploy using the same variable substitution approach as Step 3:


```bash
kubectl apply -f deployment/k8s/streamwiseapp-service-account.yaml
envsubst < deployment/aks/streamcast-pod.yaml | kubectl apply -f -
```

Verify StreamCast is running:
```bash
kubectl get pods -n rtgen
kubectl get svc -n rtgen
```

Open the StreamCast UI at: `https://$IP_ADDRESS:8080`

### Remove StreamCast
```bash
kubectl delete -f deployment/aks/streamcast-pod.yaml
kubectl delete -f deployment/k8s/streamwiseapp-service-account.yaml
```

## Step 5: GPU Setup

Install the NVIDIA device plugin so Kubernetes can schedule GPU workloads.
See the [Generic Kubernetes GPU Setup](../k8s/README.md#gpu-setup) for the generic installation steps.

For AKS, apply the local DaemonSet manifest:
```bash
kubectl create namespace gpu-resources
kubectl apply -f deployment/k8s/nvidia-device-plugin-ds.yaml
```

Scale the GPU spot node pool up (it starts at 0 nodes):

```bash
# Scale the full-GPU node pool (no MIG)
az aks nodepool scale \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --name spoth100 \
  --node-count 1

# Scale the MIG node pool
az aks nodepool scale \
  --resource-group $AZ_RESOURCE_GROUP \
  --cluster-name $AKS_CLUSTER \
  --name spoth100mig \
  --node-count 1
```

If these fails, scale the VMSS directly:
```bash
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"

# Scale the full-GPU VMSS
VMSS_FULL=$(az vmss list -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-spoth100-') && !contains(name, 'mig')].name | [0]" -o tsv)
az vmss scale -g $MC_RESOURCE_GROUP -n $VMSS_FULL --new-capacity 1

# Scale the MIG VMSS
VMSS_MIG=$(az vmss list -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-spoth100mig-')].name | [0]" -o tsv)
az vmss scale -g $MC_RESOURCE_GROUP -n $VMSS_MIG --new-capacity 1
```

> **Spot VM eviction:**
> Because these are Spot VMs, Azure may evict them at any time.
> If a node disappears or the VMSS shows 0 instances after scaling, re-run the scale command.
> Check VMSS capacity with:
> `az vmss show -g $MC_RESOURCE_GROUP -n <vmss-name> --query sku.capacity`

Log into a node using `node-shell`:
```bash
kubectl get nodes
kubectl node-shell <node-name>
```

Sometimes the NVIDIA setup with torch is not correct and we need to restart the VM:
```bash
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"

# Restart a specific instance in the full-GPU pool
VMSS_FULL=$(az vmss list -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-spoth100-') && !contains(name, 'mig')].name | [0]" -o tsv)
INSTANCE_ID=0
az vmss restart -g $MC_RESOURCE_GROUP --name $VMSS_FULL --instance-ids $INSTANCE_ID

# Restart a specific instance in the MIG pool
VMSS_MIG=$(az vmss list -g $MC_RESOURCE_GROUP \
  --query "[?contains(name, 'aks-spoth100mig-')].name | [0]" -o tsv)
INSTANCE_ID=0
az vmss restart -g $MC_RESOURCE_GROUP --name $VMSS_MIG --instance-ids $INSTANCE_ID
```

### 5.1 Partial GPU Support (MIG)

The **MIG node pool** (e.g., `spoth100mig`) runs the same VM size as the full-GPU pool but is designed for **mixed-mode** use:
- Most GPUs remain in standard mode; available as `nvidia.com/gpu` for heavy models
- **1 GPU** (the last one) is manually put in MIG mode after the node comes up, creating smaller isolated slices for lightweight services (e.g., Kokoro and YOLO)

The Bicep template labels MIG nodes with `gpu-config=mig`.
The device plugin DaemonSet uses this label to apply `MIG_STRATEGY=mixed` on those nodes and `MIG_STRATEGY=none` on full-GPU nodes, so MIG slices only appear where they are configured.

> **⚠️ MIG must be configured before GPU services can be scheduled.**
> Until MIG mode is enabled and instances are created, the `mixed`-strategy device plugin will report **0 GPUs** on the MIG node (it cannot enumerate devices without active MIG instances).
> After scaling up the MIG node pool (Step 5 above), follow the **[guide to enable MIG](../k8s/MIG.md)**, create instances, and verify the setup.
> The full-GPU node pool (`spoth100`) works without additional configuration.

## Step 6: Deploy GPU Microservices

Deploy model services through the StreamWise web UI or REST API.

**GPU requirements per service:**

| Service | GPUs | MIG Profile (recommended) | Purpose |
|---------|------|--------------------------|---------|
| `gemma` | 2–4 | — (full GPUs) | LLM (screenplay generation) |
| `flux` | 2 | — (full GPUs) | Text-to-image |
| `hunyuanframepackf1` | 2 | — (full GPUs) | Image-to-video |
| `fantasytalking` | 2 | — (full GPUs) | Audio-driven video |
| `kokoro` | 1 | `1g.10gb` (80 GB) or `1g.5gb` (40 GB) | Text-to-speech |
| `yolo` | 1 | `1g.10gb` (80 GB) or `1g.5gb` (40 GB) | Character extraction |
| `realesrgan` | 1 | `2g.20gb` (80 GB) or `2g.10gb` (40 GB) | Video upscaling |
| `podcasttranscript` | 0 | — | Transcript orchestration (CPU-only) |

> **Capacity planning:**
> With the [recommended MIG layout](../k8s/MIG.md) (2 × `2g.20gb` + 3 × `1g.10gb` on an 80 GB GPU), Kokoro, YOLO, and similar services each consume only a single MIG slice on the MIG pool, leaving the full-GPU pool for heavy models.
> For parallel execution of all services, add more GPU nodes to either pool.

The Web UI is available at `https://$IP_ADDRESS:8081` to manage services.
Use the REST API to deploy all services at once:
```bash
curl -k -X POST "https://$IP_ADDRESS:8081/api/service"
```

Or deploy individual services with specific resource allocations:
```bash
# Deploy a single service (whole GPU)
curl -k -X POST "https://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "memory=8" \
  -d "cpu=2"

# Deploy with a MIG slice (partial GPU — requires MIG configured on the node; see Step 5.1)
curl -k -X POST "https://$IP_ADDRESS:8081/api/pod" \
  -d "container_name=kokoro" \
  -d "gpu=1" \
  -d "mig_profile=1g.10gb" \
  -d "gpu_type=h100" \
  -d "memory=8" \
  -d "cpu=2"

# Verify deployed services
curl -k "https://$IP_ADDRESS:8081/api/services"
```

> **Note:** `-k` skips TLS certificate verification for the self-signed certificate.
> Replace with `--cacert /path/to/cert.pem` to verify against the certificate, or omit `-k` entirely when using a CA-signed certificate.


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
- **MIG node reports 0 GPUs**: The `mixed`-strategy device plugin cannot enumerate devices until MIG mode is enabled and MIG instances are created. Complete the full [MIG Setup Guide](../k8s/MIG.md) — once instances exist the plugin will register `nvidia.com/gpu` (full GPUs) and `nvidia.com/mig-<profile>` (MIG slices) within ~30–60 seconds
- **Spot VM evicted**: Spot VMs may be evicted at any time. Re-run `az aks nodepool scale` to restore the node. After re-scaling a MIG node you must repeat the [MIG Setup Guide](../k8s/MIG.md) since MIG state does not persist across evictions
- **LoadBalancer stuck in Pending**: Verify the public IP exists (`az network public-ip show -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip`) and the AKS identity has Network Contributor role on the resource group
- **Cannot reach LoadBalancer services (e.g. StreamWise at :8081)**: When `disableDefaultOutboundAccess` is true, the Bicep template creates an NSG (`aks-node-subnet-nsg`) allowing inbound TCP on the Kubernetes NodePort range. For `type: LoadBalancer` Services, the subnet NSG evaluates traffic destined to the node private IPs and the Service’s NodePort(s), not the public Load Balancer IP. If a corporate policy replaces or overrides this NSG on the subnet, add an inbound allow rule on the node subnet NSG such as: `az network nsg rule create -g $AZ_RESOURCE_GROUP --nsg-name <subnet-nsg> --name AllowK8sNodePorts --priority 100 --direction Inbound --access Allow --protocol Tcp --source-address-prefixes Internet --destination-address-prefixes VirtualNetwork --destination-port-ranges 30000-32767`. For tighter control, you can set explicit `nodePort` values on your Services and open only those ports instead of the full range. Azure evaluates both the subnet NSG and the NIC-level NSG (in the MC_ resource group) — traffic must be allowed by **both**.
- **Secret errors**: Verify HF token is correctly configured with `kubectl get secret hf-token -n rtgen`
- **ACR role assignment fails on redeployment**: The role already exists. Attach ACR manually: `az aks update -g $AZ_RESOURCE_GROUP -n $AKS_CLUSTER --attach-acr <acrName>`

## Cleanup

```bash
# Delete pods and services
kubectl delete -f deployment/aks/streamcast-pod.yaml
kubectl delete -f deployment/k8s/streamwiseapp-service-account.yaml
kubectl delete -f deployment/aks/streamwise-pod.yaml
kubectl delete -f deployment/k8s/streamwise-service-account.yaml

# Delete storage and namespace
kubectl delete pvc local-pvc -n rtgen
kubectl delete pv local-pv
kubectl delete namespace rtgen

# Delete entire resource group (includes AKS cluster and all resources)
az group delete --name $AZ_RESOURCE_GROUP --yes --no-wait
```
