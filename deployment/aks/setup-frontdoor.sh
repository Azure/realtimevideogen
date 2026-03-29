#!/usr/bin/env bash
# setup-frontdoor.sh — Create Azure Front Door Premium with Private Link
# to the AKS internal load balancer.
#
# Prerequisites:
#   - AKS cluster deployed with aks-frontdoor.bicep
#   - kubectl configured to talk to the cluster
#   - K8s namespace, HF secret, and service accounts created
#
# Required environment variables:
#   AZ_RESOURCE_GROUP   - Azure resource group
#   AZ_REGION           - Azure region (e.g. eastus2)
#   K8S_NAMESPACE       - Kubernetes namespace (default: rtgen)
#   ACR_URL             - ACR URL (e.g. myacr.azurecr.io)
#
# Usage:
#   export AZ_RESOURCE_GROUP=my-rg AZ_REGION=eastus2 ACR_URL=myacr.azurecr.io
#   bash deployment/aks/setup-frontdoor.sh
set -euo pipefail

: "${AZ_RESOURCE_GROUP:?AZ_RESOURCE_GROUP must be set}"
: "${AZ_REGION:?AZ_REGION must be set}"
K8S_NAMESPACE="${K8S_NAMESPACE:-rtgen}"
: "${ACR_URL:?ACR_URL must be set}"

AKS_CLUSTER=$(az aks list -g "$AZ_RESOURCE_GROUP" --query "[0].name" -o tsv)
MC_RG="MC_${AZ_RESOURCE_GROUP}_${AKS_CLUSTER}_${AZ_REGION}"
AFD_NAME="afd-${AZ_RESOURCE_GROUP}"

echo "=== Front Door Premium + Private Link Setup ==="
echo "RG:        $AZ_RESOURCE_GROUP"
echo "Region:    $AZ_REGION"
echo "Cluster:   $AKS_CLUSTER"
echo "MC RG:     $MC_RG"
echo "Namespace: $K8S_NAMESPACE"
echo ""

# -- 1. Deploy pods with internal LB services --------------------------------
echo ">>> Deploying StreamWise and StreamCast with internal LoadBalancer..."

# StreamWise internal LB service
cat <<'EOF' | RESOURCE_GROUP_NAME="$AZ_RESOURCE_GROUP" LOAD_BALANCER_IP="" ACR_URL="$ACR_URL" envsubst | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: streamwise
  namespace: rtgen
  labels:
    app: streamwise
spec:
  containers:
  - name: streamwise
    image: ${ACR_URL}/streamwise:v0.5.0
    ports:
    - containerPort: 18181
      protocol: TCP
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
    - name: NVIDIA_VISIBLE_DEVICES
      value: none
    - name: LB_RESOURCE_GROUP
      value: ${RESOURCE_GROUP_NAME}
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
        ephemeral-storage: "16Gi"
      limits:
        cpu: "4"
        memory: "8Gi"
        ephemeral-storage: "32Gi"
  nodeSelector:
    kubernetes.io/os: linux
    kubernetes.io/arch: amd64
  serviceAccountName: streamwise-service-account
---
apiVersion: v1
kind: Service
metadata:
  name: streamwise-svc
  namespace: rtgen
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  selector:
    app: streamwise
  ports:
  - port: 8081
    targetPort: 18181
    protocol: TCP
EOF

# StreamCast internal LB service
cat <<'EOF' | ACR_URL="$ACR_URL" envsubst | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: streamcast
  namespace: rtgen
  labels:
    app: streamcast
spec:
  containers:
  - name: streamcast
    image: ${ACR_URL}/streamcast:v0.5.0
    ports:
    - containerPort: 18080
      protocol: TCP
    env:
    - name: HUGGING_FACE_HUB_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: token
    - name: NVIDIA_VISIBLE_DEVICES
      value: none
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
        ephemeral-storage: "16Gi"
      limits:
        cpu: "4"
        memory: "8Gi"
        ephemeral-storage: "32Gi"
  nodeSelector:
    kubernetes.io/os: linux
    kubernetes.io/arch: amd64
  serviceAccountName: streamwiseapp-service-account
---
apiVersion: v1
kind: Service
metadata:
  name: streamcast-svc
  namespace: rtgen
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  selector:
    app: streamcast
  ports:
  - port: 8080
    targetPort: 18080
    protocol: TCP
EOF

echo ">>> Waiting for pods and internal LB IPs..."
kubectl wait --for=condition=Ready pod/streamwise pod/streamcast -n "$K8S_NAMESPACE" --timeout=300s

# Wait for internal LB IPs to be assigned
for i in $(seq 1 30); do
  SW_IP=$(kubectl get svc streamwise-svc -n "$K8S_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  SC_IP=$(kubectl get svc streamcast-svc -n "$K8S_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  if [ -n "$SW_IP" ] && [ -n "$SC_IP" ]; then
    break
  fi
  sleep 10
done
echo "    StreamWise ILB IP: $SW_IP"
echo "    StreamCast ILB IP: $SC_IP"

# -- 2. Find the internal LB resource ----------------------------------------
echo ">>> Finding internal load balancer..."
ILB_ID=$(az network lb list -g "$MC_RG" \
  --query "[?contains(name, 'kubernetes-internal')].id | [0]" -o tsv)
ILB_NAME=$(az network lb list -g "$MC_RG" \
  --query "[?contains(name, 'kubernetes-internal')].name | [0]" -o tsv)
ILB_FRONTEND_IP_ID=$(az network lb frontend-ip list -g "$MC_RG" --lb-name "$ILB_NAME" \
  --query "[0].id" -o tsv)
echo "    ILB: $ILB_NAME"

# -- 3. Create Private Link Service ------------------------------------------
echo ">>> Creating Private Link Service..."
PLS_SUBNET_ID=$(az network vnet subnet show -g "$AZ_RESOURCE_GROUP" \
  --vnet-name aks-vnet --name pls-subnet --query id -o tsv)

az network private-link-service create \
  --name streamwise-pls \
  --resource-group "$AZ_RESOURCE_GROUP" \
  --lb-frontend-ip-configs "$ILB_FRONTEND_IP_ID" \
  --subnet "$PLS_SUBNET_ID" \
  --auto-approval "*" \
  --visibility "*" \
  --location "$AZ_REGION" \
  --only-show-errors -o none

PLS_ID=$(az network private-link-service show \
  --name streamwise-pls -g "$AZ_RESOURCE_GROUP" --query id -o tsv)
echo "    PLS ID: $PLS_ID"

# -- 4. Create Front Door Premium ---------------------------------------------
echo ">>> Creating Front Door Premium profile..."
az afd profile create \
  --profile-name "$AFD_NAME" \
  --resource-group "$AZ_RESOURCE_GROUP" \
  --sku Premium_AzureFrontDoor \
  --only-show-errors -o none

# -- StreamWise endpoint
echo ">>> Creating StreamWise endpoint..."
az afd endpoint create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamwise --only-show-errors -o none

az afd origin-group create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --origin-group-name streamwise-og \
  --probe-path "/health" --probe-protocol Http --probe-request-type GET \
  --probe-interval-in-seconds 30 \
  --sample-size 4 --successful-samples-required 3 \
  --additional-latency-in-milliseconds 50 \
  --only-show-errors -o none

az afd origin create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --origin-group-name streamwise-og \
  --origin-name aks-streamwise \
  --host-name "$SW_IP" \
  --origin-host-header "$SW_IP" \
  --http-port 8081 \
  --priority 1 --weight 1000 \
  --enable-private-link true \
  --private-link-resource "$PLS_ID" \
  --private-link-location "$AZ_REGION" \
  --private-link-request-message "Front Door" \
  --enforce-certificate-name-check false \
  --only-show-errors -o none

az afd route create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamwise --route-name streamwise-route \
  --origin-group streamwise-og \
  --supported-protocols Http Https \
  --forwarding-protocol HttpOnly \
  --https-redirect Enabled \
  --link-to-default-domain Enabled \
  --patterns-to-match "/*" \
  --only-show-errors -o none

# -- StreamCast endpoint
echo ">>> Creating StreamCast endpoint..."
az afd endpoint create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamcast --only-show-errors -o none

az afd origin-group create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --origin-group-name streamcast-og \
  --probe-path "/health" --probe-protocol Http --probe-request-type GET \
  --probe-interval-in-seconds 30 \
  --sample-size 4 --successful-samples-required 3 \
  --additional-latency-in-milliseconds 50 \
  --only-show-errors -o none

az afd origin create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --origin-group-name streamcast-og \
  --origin-name aks-streamcast \
  --host-name "$SC_IP" \
  --origin-host-header "$SC_IP" \
  --http-port 8080 \
  --priority 1 --weight 1000 \
  --enable-private-link true \
  --private-link-resource "$PLS_ID" \
  --private-link-location "$AZ_REGION" \
  --private-link-request-message "Front Door" \
  --enforce-certificate-name-check false \
  --only-show-errors -o none

az afd route create --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamcast --route-name streamcast-route \
  --origin-group streamcast-og \
  --supported-protocols Http Https \
  --forwarding-protocol HttpOnly \
  --https-redirect Enabled \
  --link-to-default-domain Enabled \
  --patterns-to-match "/*" \
  --only-show-errors -o none

# -- 5. Get the endpoint URLs -------------------------------------------------
SW_URL=$(az afd endpoint show --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamwise --query hostName -o tsv)
SC_URL=$(az afd endpoint show --profile-name "$AFD_NAME" -g "$AZ_RESOURCE_GROUP" \
  --endpoint-name streamcast --query hostName -o tsv)

echo ""
echo "==========================================="
echo "Front Door Premium deployment complete!"
echo ""
echo "StreamWise: https://$SW_URL"
echo "StreamCast: https://$SC_URL"
echo ""
echo "Browser-trusted HTTPS — no certificate warnings."
echo "Front Door uses Private Link to reach AKS (bypasses NRMS)."
echo ""
echo "NOTE: Front Door propagation takes 5-15 minutes."
echo "If you see 404, wait and retry."
echo "==========================================="
