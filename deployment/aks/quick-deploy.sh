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
# HTTPS/TLS: set ENABLE_SECURE=true to provision Key Vault and TLS certificates.
# By default the cluster is deployed without TLS (HTTP only).
export ENABLE_SECURE="false"
# CA-signed cert: set LETSENCRYPT_EMAIL when ENABLE_SECURE=true to obtain a
# browser-trusted Let's Encrypt certificate automatically.
# NOTE: Let's Encrypt requires port 80 reachable from the public Internet.
# Corporate Azure subscriptions with NRMS rules may block this — the script
# will fall back to a self-signed certificate automatically.
export LETSENCRYPT_EMAIL=""                    # e.g. "team@example.com"
# ────────────────────────────────────────────────────────────────────────

# shellcheck source=deployment/set_properties.sh.template
source deployment/set_properties.sh  # loads ACR_URL etc.

# Pre-flight: check required tools
for cmd in az kubectl envsubst openssl; do
  command -v "$cmd" &>/dev/null || { echo "ERROR: '$cmd' not found. Install it first."; exit 1; }
done
if [ "$ENABLE_SECURE" = "true" ] && [ -n "${LETSENCRYPT_EMAIL:-}" ]; then
  command -v helm &>/dev/null || { echo "ERROR: 'helm' required for Let's Encrypt. Install: https://helm.sh/docs/intro/install/"; exit 1; }
fi

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
    acrResourceGroup=$ACR_RG \
    enableSecureSetup=$ENABLE_SECURE \
  || {
    # Known non-fatal failures:
    #   - ACR role assignment: "RoleAssignmentUpdateNotPermitted" (redeployment)
    #   - Federated creds: "ConcurrentFederatedIdentityCredentialsWrites" (race)
    # The AKS cluster itself is created successfully in both cases.
    echo ">>> Bicep reported errors (cluster may still be OK — checking...)"
    AKS_CHECK=$(az aks list -g $AZ_RESOURCE_GROUP --query "[0].name" -o tsv 2>/dev/null || true)
    if [ -z "$AKS_CHECK" ]; then
      echo "ERROR: AKS cluster was not created. Check the deployment errors above."
      exit 1
    fi
    echo ">>> AKS cluster exists — continuing with manual fixups."
  }

# If ACR role assignment failed (redeployment), attach manually.
# Retry in a loop in case an AKS operation is still in progress (node pool scaling).
AKS_ATTACH_CLUSTER_NAME="$(az aks list -g "$AZ_RESOURCE_GROUP" --query "[0].name" -o tsv)"
if [ -z "$AKS_ATTACH_CLUSTER_NAME" ]; then
  echo "ERROR: Could not determine AKS cluster name for ACR attachment." >&2
  exit 1
fi
acr_attach_succeeded=false
for attempt in 1 2 3 4 5; do
  if az aks update -g "$AZ_RESOURCE_GROUP" -n "$AKS_ATTACH_CLUSTER_NAME" \
    --attach-acr "$ACR_NAME" 2>/dev/null; then
    acr_attach_succeeded=true
    break
  fi
  echo ">>> ACR attach attempt $attempt failed (likely in-progress operation) — retrying in 30s..."
  sleep 30
done

if [ "$acr_attach_succeeded" != "true" ]; then
  echo "ERROR: Failed to attach ACR '$ACR_NAME' to AKS cluster '$AKS_ATTACH_CLUSTER_NAME' after 5 attempts." >&2
  echo "ERROR: Image pulls may fail until this is fixed." >&2
  echo "ERROR: Run the following command after any in-progress AKS operation completes:" >&2
  echo "  az aks update -g $AZ_RESOURCE_GROUP -n \"$AKS_ATTACH_CLUSTER_NAME\" --attach-acr $ACR_NAME" >&2
  exit 1
fi

# 2. Retrieve outputs
AKS_CLUSTER=$(az deployment group show --name AKSDeployment -g $AZ_RESOURCE_GROUP \
  --query properties.outputs.clusterName.value -o tsv)
IP_ADDRESS=$(az network public-ip show -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip --query ipAddress -o tsv)
MC_RESOURCE_GROUP="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"

if [ "$ENABLE_SECURE" = "true" ]; then
  PUBLIC_FQDN=$(az network public-ip show -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip --query dnsSettings.fqdn -o tsv)
  KEY_VAULT_NAME=$(az keyvault list -g $AZ_RESOURCE_GROUP --query "[0].name" -o tsv)
fi

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

# 5. TLS certificate setup (only when ENABLE_SECURE=true)
export LOAD_BALANCER_IP=$IP_ADDRESS RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP

if [ "$ENABLE_SECURE" = "true" ]; then
  if [ -n "${LETSENCRYPT_EMAIL:-}" ]; then
    echo ">>> Bootstrapping with self-signed cert while Let's Encrypt provisions..."
  else
    echo ">>> Using self-signed certificate from Key Vault (set LETSENCRYPT_EMAIL for CA-signed)."
  fi
  # Try Key Vault first; fall back to openssl-generated self-signed cert if the KV
  # cert was not created (e.g. partial Bicep failure or concurrent write race).
  # The KV cert may still be generating — wait up to 2 minutes.
  KV_CERT_READY="false"
  for i in $(seq 1 24); do
    if az keyvault certificate show --vault-name "$KEY_VAULT_NAME" --name streamwise-tls --query attributes.enabled -o tsv 2>/dev/null | grep -q true; then
      KV_CERT_READY="true"
      break
    fi
    echo ">>> Waiting for KV cert (attempt $i/24)..."
    sleep 5
  done

  if [ "$KV_CERT_READY" = "true" ]; then
    echo ">>> Downloading TLS certificate from Key Vault..."
    az keyvault certificate download --vault-name "$KEY_VAULT_NAME" --name streamwise-tls --encoding PEM -f /tmp/tls.crt
    az keyvault secret download --vault-name "$KEY_VAULT_NAME" --name streamwise-tls -f /tmp/bundle.pem
    openssl pkey -in /tmp/bundle.pem -out /tmp/tls.key
  else
    echo ">>> Key Vault cert not ready — generating self-signed cert with openssl."
    CERT_CN="${PUBLIC_FQDN:-streamwise}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout /tmp/tls.key -out /tmp/tls.crt \
      -subj "/CN=$CERT_CN" -addext "subjectAltName=DNS:$CERT_CN"
  fi
  kubectl create secret tls streamwise-tls-secret -n "$K8S_NAMESPACE" --cert=/tmp/tls.crt --key=/tmp/tls.key \
    --dry-run=client -o yaml | kubectl apply -f -
  rm -f /tmp/tls.crt /tmp/bundle.pem /tmp/tls.key
fi

# 6. Deploy StreamWise + StreamCast
envsubst < deployment/aks/streamwise-pod.yaml | kubectl apply -f -
envsubst < deployment/aks/streamcast-pod.yaml | kubectl apply -f -

# 7. Reboot GPU VMs (fixes NVIDIA driver/torch issues on first boot)
VMSS_FULL=$(az vmss list -g "$MC_RESOURCE_GROUP" \
  --query "[?contains(name, 'aks-${GPU_POOL}-') && !contains(name, 'mig')].name | [0]" -o tsv)
VMSS_MIG=$(az vmss list -g "$MC_RESOURCE_GROUP" \
  --query "[?contains(name, 'aks-${MIG_POOL}-')].name | [0]" -o tsv)
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_FULL" --instance-ids \
  "$(az vmss list-instances -g "$MC_RESOURCE_GROUP" -n "$VMSS_FULL" --query '[0].instanceId' -o tsv)"
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_MIG" --instance-ids \
  "$(az vmss list-instances -g "$MC_RESOURCE_GROUP" -n "$VMSS_MIG" --query '[0].instanceId' -o tsv)"

# Wait for nodes to come back; Azure CNS needs ~30-60s after reboot to accept pods.
kubectl wait --for=condition=Ready node -l kubernetes.azure.com/scalesetpriority=spot --timeout=300s
echo ">>> Waiting for Azure CNS to stabilize after reboot..."
sleep 45

# 8. MIG setup (see deployment/k8s/MIG.md for detailed guide)
GPU_NODE=$(kubectl get nodes -l gpu-config=mig -o jsonpath='{.items[0].metadata.name}')
# Determine GPU_INDEX from VM size (last GPU, 0-indexed)
case "$GPU_VM_SIZE" in
  Standard_NC96ads_A100_v4) GPU_INDEX=3 ;;  # 4 GPUs
  Standard_ND96ams_A100_v4|Standard_ND96isrf_H100_v5|Standard_ND96isr_H200_v5) GPU_INDEX=7 ;;  # 8 GPUs
  *) GPU_INDEX=7 ; echo "WARNING: Unknown VM size '$GPU_VM_SIZE' — defaulting to GPU_INDEX=7" ;;
esac
echo ">>> MIG setup: GPU_INDEX=$GPU_INDEX on node $GPU_NODE"
export GPU_NODE
envsubst < deployment/k8s/gpu-debug-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=180s
kubectl taint node "$GPU_NODE" mig-setup=true:NoExecute
sleep 5
kubectl exec gpu-debug -- chroot /host nvidia-smi -i "$GPU_INDEX" -mig 1
# Reboot required to activate MIG mode
MIG_INSTANCE_ID=$(az vmss list-instances -g "$MC_RESOURCE_GROUP" -n "$VMSS_MIG" --query "[0].instanceId" -o tsv)
az vmss restart -g "$MC_RESOURCE_GROUP" --name "$VMSS_MIG" --instance-ids "$MIG_INSTANCE_ID"
kubectl wait --for=condition=Ready "node/$GPU_NODE" --timeout=300s
sleep 45  # wait for Azure CNS
kubectl delete pod gpu-debug --ignore-not-found
envsubst < deployment/k8s/gpu-debug-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=180s
# For 80 GB GPUs (A100/H100): 2×2g.20gb + 3×1g.10gb
kubectl exec gpu-debug -- chroot /host nvidia-smi mig -cgi 2g.20gb,2g.20gb,1g.10gb,1g.10gb,1g.10gb -C -i "$GPU_INDEX"
kubectl taint node "$GPU_NODE" mig-setup=true:NoExecute-
kubectl delete pod gpu-debug --ignore-not-found

# 9. CA-signed certificate (only when ENABLE_SECURE=true and LETSENCRYPT_EMAIL is set)
if [ "$ENABLE_SECURE" = "true" ] && [ -n "${LETSENCRYPT_EMAIL:-}" ]; then
  echo ">>> Setting up Let's Encrypt CA-signed certificate..."
  if bash deployment/aks/setup-letsencrypt.sh; then
    echo ">>> Let's Encrypt setup succeeded."
    # Redeploy pods to pick up the new cert (setup-letsencrypt.sh deletes them)
    envsubst < deployment/aks/streamwise-pod.yaml | kubectl apply -f -
    envsubst < deployment/aks/streamcast-pod.yaml | kubectl apply -f -
    kubectl wait --for=condition=Ready pod/streamwise pod/streamcast -n "$K8S_NAMESPACE" --timeout=300s
  else
    echo ">>> Let's Encrypt failed (common on corporate subscriptions with NRMS rules"
    echo "    that block port 80 from the Internet). Continuing with self-signed cert."
    echo "    To retry later: bash deployment/aks/setup-letsencrypt.sh"
  fi
fi

echo "==========================================="
echo "Deployment complete!"
if [ "$ENABLE_SECURE" = "true" ]; then
  echo "StreamWise: https://$PUBLIC_FQDN:8081"
  echo "StreamCast: https://$PUBLIC_FQDN:8080"
  # Check if the cert is CA-signed or self-signed
  CERT_ISSUER=$(kubectl get secret streamwise-tls-secret -n "$K8S_NAMESPACE" \
    -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -issuer 2>/dev/null || true)
  if echo "$CERT_ISSUER" | grep -qi "let's encrypt\|R[0-9]\|E[0-9]"; then
    echo "TLS:        CA-signed (Let's Encrypt) — browser-trusted"
  else
    echo "TLS:        Self-signed — accept the warning in your browser or use curl -k"
    echo "            To get a CA-signed cert: bash deployment/aks/setup-letsencrypt.sh"
  fi
else
  echo "StreamWise: http://$IP_ADDRESS:8081"
  echo "StreamCast: http://$IP_ADDRESS:8080"
  echo "TLS:        Disabled (set ENABLE_SECURE=true for HTTPS)"
fi
echo "Public IP:  $IP_ADDRESS"
echo "==========================================="
