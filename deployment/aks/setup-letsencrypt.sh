#!/usr/bin/env bash
# setup-letsencrypt.sh - Automate CA-signed TLS certificate provisioning via
# cert-manager + Let's Encrypt.
#
# This script replaces the self-signed fallback certificate (created by
# aks.bicep) with a browser-trusted CA-signed certificate.  It installs
# cert-manager, the nginx-ingress controller, creates the Let's Encrypt
# ClusterIssuers, and requests the certificate.
#
# Prerequisites:
#   - AKS cluster deployed with enableSecureSetup=true (provides DNS label,
#     port 80 NSG rule, and workload identity)
#   - kubectl configured to talk to the cluster
#   - helm installed (https://helm.sh/docs/intro/install/)
#
# Required environment variables:
#   PUBLIC_FQDN          - e.g. streamwise-abc123.eastus2.cloudapp.azure.com
#   LOAD_BALANCER_IP     - static public IP address (used by nginx-ingress)
#   RESOURCE_GROUP_NAME  - Azure resource group containing the AKS cluster
#   K8S_NAMESPACE        - Kubernetes namespace (default: rtgen)
#   LETSENCRYPT_EMAIL    - email for Let's Encrypt expiry notices
#
# Usage:
#   export PUBLIC_FQDN=... LOAD_BALANCER_IP=... RESOURCE_GROUP_NAME=...
#   export K8S_NAMESPACE=rtgen LETSENCRYPT_EMAIL=your@email.com
#   bash deployment/aks/setup-letsencrypt.sh
#
# After this script completes, the Kubernetes TLS secret
# "streamwise-tls-secret" in $K8S_NAMESPACE will contain the CA-signed
# certificate.  Pods mounting that secret at /certs/ will serve trusted HTTPS.
set -euo pipefail

: "${PUBLIC_FQDN:?PUBLIC_FQDN must be set (e.g. streamwise-abc123.eastus2.cloudapp.azure.com)}"
: "${LOAD_BALANCER_IP:?LOAD_BALANCER_IP must be set (static public IP address)}"
: "${RESOURCE_GROUP_NAME:?RESOURCE_GROUP_NAME must be set (Azure resource group)}"
: "${LETSENCRYPT_EMAIL:?LETSENCRYPT_EMAIL must be set (email for Lets Encrypt notifications)}"
K8S_NAMESPACE="${K8S_NAMESPACE:-rtgen}"
export PUBLIC_FQDN LOAD_BALANCER_IP RESOURCE_GROUP_NAME K8S_NAMESPACE LETSENCRYPT_EMAIL

echo "=== Let's Encrypt TLS setup ==="
echo "FQDN:        $PUBLIC_FQDN"
echo "IP:          $LOAD_BALANCER_IP"
echo "RG:          $RESOURCE_GROUP_NAME"
echo "Namespace:   $K8S_NAMESPACE"
echo "Email:       $LETSENCRYPT_EMAIL"
echo ""

# -- 1. Install cert-manager --------------------------------------------
echo ">>> Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

echo ">>> Waiting for cert-manager pods to be ready..."
kubectl wait --namespace cert-manager \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/instance=cert-manager \
  --timeout=120s

# -- 2. Install nginx-ingress controller --------------------------------
echo ">>> Installing nginx-ingress controller..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>/dev/null || true
helm repo update ingress-nginx

if helm status ingress-nginx --namespace ingress-nginx &>/dev/null; then
  echo "    nginx-ingress already installed, upgrading..."
  helm upgrade ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --set controller.service.loadBalancerIP="$LOAD_BALANCER_IP" \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-resource-group"="$RESOURCE_GROUP_NAME" \
    --set controller.service.externalTrafficPolicy=Local
else
  helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx --create-namespace \
    --set controller.service.loadBalancerIP="$LOAD_BALANCER_IP" \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-resource-group"="$RESOURCE_GROUP_NAME" \
    --set controller.service.externalTrafficPolicy=Local
fi

echo ">>> Waiting for nginx-ingress controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# -- 3. Create ClusterIssuers ------------------------------------------
echo ">>> Creating Let's Encrypt ClusterIssuers..."
envsubst < deployment/k8s/cert-manager-issuer.yaml | kubectl apply -f -

echo ">>> Waiting for issuers to be ready..."
for i in $(seq 1 30); do
  STAGING_READY=$(kubectl get clusterissuer letsencrypt-staging -o jsonpath='{.status.conditions[0].status}' 2>/dev/null || true)
  PROD_READY=$(kubectl get clusterissuer letsencrypt-prod -o jsonpath='{.status.conditions[0].status}' 2>/dev/null || true)
  if [ "$STAGING_READY" = "True" ] && [ "$PROD_READY" = "True" ]; then
    echo "    Both issuers are ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "WARNING: Issuers not ready after 60s - continuing anyway."
  fi
  sleep 2
done

# -- 4. Request the CA-signed certificate -------------------------------
echo ">>> Requesting CA-signed certificate for $PUBLIC_FQDN..."
envsubst < deployment/k8s/streamwise-certificate.yaml | kubectl apply -f -

echo ">>> Waiting for certificate to be issued (this may take 1-3 minutes)..."
for i in $(seq 1 90); do
  CERT_READY=$(kubectl get certificate streamwise-tls -n "$K8S_NAMESPACE" \
    -o jsonpath='{.status.conditions[0].status}' 2>/dev/null || true)
  if [ "$CERT_READY" = "True" ]; then
    echo "    Certificate issued successfully!"
    break
  fi
  if [ "$i" -eq 90 ]; then
    echo "WARNING: Certificate not ready after 3 minutes."
    echo "  Check status:  kubectl describe certificate streamwise-tls -n $K8S_NAMESPACE"
    echo "  Check order:   kubectl describe order -n $K8S_NAMESPACE"
    echo "  Check solver:  kubectl get pods -A | grep cm-acme"
    echo "  The certificate may still be issued - pods will pick it up automatically."
  fi
  sleep 2
done

# -- 5. Restart pods to pick up the new certificate ---------------------
echo ">>> Restarting StreamWise and StreamCast pods to pick up the CA-signed certificate..."
kubectl delete pod streamwise -n "$K8S_NAMESPACE" --ignore-not-found
kubectl delete pod streamcast -n "$K8S_NAMESPACE" --ignore-not-found

echo ""
echo "==========================================="
echo "Let's Encrypt TLS setup complete!"
echo ""
echo "Once pods restart, verify with:"
echo "  curl https://$PUBLIC_FQDN:8081/health"
echo "  curl https://$PUBLIC_FQDN:8080/health"
echo ""
echo "No -k flag needed - the certificate is now browser-trusted."
echo "==========================================="
