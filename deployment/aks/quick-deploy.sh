#!/usr/bin/env bash
# quick-deploy.sh — End-to-end AKS deployment script.
# Edit the configuration variables below, then run the script from the repository root.
set -euo pipefail

# ── Configuration (edit these) ──────────────────────────────────────────
export AZ_RESOURCE_GROUP="my-rg"
export AZ_REGION="eastus2"
export ACR_NAME="myacr"
export ACR_RG="$AZ_RESOURCE_GROUP"            # set to ACR resource group if different
export HF_TOKEN="hf_..."
export K8S_NAMESPACE="rtgen"
export GPU_VM_SIZE="Standard_NC96ads_A100_v4"  # see VM sizes table in README.md
export SYSTEM_VM_SIZE="Standard_D16ds_v5"      # override if unavailable in your region
export GPU_POOL="spota100"
export MIG_POOL="spota100mig"
# ────────────────────────────────────────────────────────────────────────

# shellcheck source=deployment/set_properties.sh.template
source deployment/set_properties.sh  # loads ACR_URL etc.

# 1. Create RG & deploy AKS cluster
az group create --name $AZ_RESOURCE_GROUP --location $AZ_REGION
az deployment group create \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --template-file deployment/aks/aks.bicep \
  --parameters \
    systemNodeVmSize=$SYSTEM_VM_SIZE \
    gpuNodeVmSize=$GPU_VM_SIZE \
    gpuNodePoolName=$GPU_POOL \
    gpuMigNodePoolName=$MIG_POOL \
    gpuNodeCount=1 \
    gpuMigNodeCount=1 \
    acrName=$ACR_NAME \
    acrResourceGroup=$ACR_RG

# If ACR role assignment failed (redeployment), attach manually:
az aks update -g $AZ_RESOURCE_GROUP -n "$(az aks list -g $AZ_RESOURCE_GROUP --query "[0].name" -o tsv)" \
  --attach-acr $ACR_NAME || true

# 2. Retrieve outputs
AKS_CLUSTER=$(az deployment group show --name AKSDeployment -g $AZ_RESOURCE_GROUP \
  --query properties.outputs.clusterName.value -o tsv)
IP_ADDRESS=$(az network public-ip show -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip --query ipAddress -o tsv)
PUBLIC_FQDN=$(az deployment group show --name AKSDeployment -g $AZ_RESOURCE_GROUP \
  --query properties.outputs.publicFqdn.value -o tsv)
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"

az aks get-credentials -g $AZ_RESOURCE_GROUP -n "$AKS_CLUSTER" --overwrite-existing

# 3. K8s prerequisites
kubectl create namespace $K8S_NAMESPACE
kubectl create secret generic hf-token -n $K8S_NAMESPACE --from-literal=token="$HF_TOKEN"
kubectl apply -f deployment/k8s/local-pv.yaml
kubectl apply -f deployment/k8s/local-pvc.yaml -n $K8S_NAMESPACE
kubectl apply -f deployment/k8s/streamwise-service-account.yaml
kubectl apply -f deployment/k8s/streamwiseapp-service-account.yaml

# 4. NVIDIA device plugin
kubectl create namespace gpu-resources
kubectl apply -f deployment/k8s/nvidia-device-plugin-ds.yaml

# 5. Deploy StreamWise + StreamCast
envsubst < deployment/aks/streamwise-pod.yaml | kubectl apply -f -
envsubst < deployment/aks/streamcast-pod.yaml | kubectl apply -f -

# 6. Reboot GPU VMs (fixes NVIDIA driver/torch issues on first boot)
VMSS_FULL=$(az vmss list -g "$MC_RESOURCE_GROUP" \
  --query "[?contains(name, 'aks-${GPU_POOL}-') && !contains(name, 'mig')].name | [0]" -o tsv)
VMSS_MIG=$(az vmss list -g "$MC_RESOURCE_GROUP" \
  --query "[?contains(name, 'aks-${MIG_POOL}-')].name | [0]" -o tsv)
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_FULL" --instance-ids 0
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_MIG" --instance-ids 0

# Wait for nodes to come back
kubectl wait --for=condition=Ready node -l kubernetes.azure.com/scalesetpriority=spot --timeout=300s

# 7. MIG setup (see deployment/k8s/MIG.md for detailed guide)
GPU_NODE=$(kubectl get nodes -l gpu-config=mig -o jsonpath='{.items[0].metadata.name}')
# Determine GPU_INDEX from VM size (last GPU, 0-indexed)
# Standard_NC96ads_A100_v4 → 4 GPUs → GPU_INDEX=3
# Standard_ND96ams_A100_v4 / H100 → 8 GPUs → GPU_INDEX=7
export GPU_NODE
envsubst < deployment/k8s/gpu-debug-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=120s
kubectl taint node "$GPU_NODE" mig-setup=true:NoExecute
sleep 5
kubectl exec gpu-debug -- chroot /host nvidia-smi -i "$GPU_INDEX" -mig 1
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_MIG" --instance-ids 0
kubectl wait --for=condition=Ready "node/$GPU_NODE" --timeout=300s
kubectl delete pod gpu-debug --ignore-not-found
envsubst < deployment/k8s/gpu-debug-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=120s
# For 80 GB GPUs (A100/H100): 2×2g.20gb + 3×1g.10gb
kubectl exec gpu-debug -- chroot /host nvidia-smi mig -cgi 2g.20gb,2g.20gb,1g.10gb,1g.10gb,1g.10gb -C -i "$GPU_INDEX"
kubectl taint node "$GPU_NODE" mig-setup=true:NoExecute-
kubectl delete pod gpu-debug --ignore-not-found

echo "==========================================="
echo "Deployment complete!"
echo "StreamWise: http://$IP_ADDRESS:8081"
echo "StreamCast: http://$IP_ADDRESS:8080"
echo "Public IP:  $IP_ADDRESS"
echo "Public FQDN: $PUBLIC_FQDN"
echo "==========================================="
