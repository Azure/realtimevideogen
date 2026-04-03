#!/usr/bin/env bash
# issue-onecert.sh - Issue a publicly trusted certificate via OneCertV2-PublicCA
#
# Prerequisites: Register the domain at https://aka.ms/onecert first!
#
# Usage: bash deployment/aks/issue-onecert.sh
set -euo pipefail

VAULT_NAME="kv-6rqfam6evyy5y"
CERT_NAME="streamwise-tls"
FQDN="streamwise-6rqfam.eastus2.cloudapp.azure.com"
K8S_NAMESPACE="${K8S_NAMESPACE:-rtgen}"
ISSUER_NAME="${ONECERT_ISSUER:-Testpublic}"  # Use "Test" for PrivateCA

echo "=== OneCertV2 Certificate Issuance ==="
echo "Vault:   $VAULT_NAME"
echo "Cert:    $CERT_NAME"
echo "FQDN:    $FQDN"
echo "Issuer:  $ISSUER_NAME"
echo ""

# -- 1. Create the certificate in Key Vault ---------------------------------
echo ">>> Creating certificate in Key Vault..."
POLICY=$(cat <<EOF
{
  "issuerParameters": { "name": "$ISSUER_NAME" },
  "keyProperties": { "exportable": true, "keyType": "RSA", "keySize": 2048, "reuseKey": false },
  "secretProperties": { "contentType": "application/x-pem-file" },
  "x509CertificateProperties": {
    "subject": "CN=$FQDN",
    "subjectAlternativeNames": { "dnsNames": ["$FQDN"] },
    "validityInMonths": 12,
    "keyUsage": ["digitalSignature", "keyEncipherment"],
    "ekus": ["1.3.6.1.5.5.7.3.1", "1.3.6.1.5.5.7.3.2"]
  },
  "lifetimeActions": [{ "trigger": { "daysBeforeExpiry": 30 }, "action": { "actionType": "AutoRenew" } }]
}
EOF
)

# Write policy to temp file
TMPFILE=$(mktemp /tmp/cert-policy-XXXXXX.json)
echo "$POLICY" > "$TMPFILE"

az keyvault certificate create \
  --vault-name "$VAULT_NAME" \
  --name "$CERT_NAME" \
  --policy "@$TMPFILE" \
  --only-show-errors || true

rm -f "$TMPFILE"

# -- 2. Wait for the certificate to be issued --------------------------------
echo ">>> Waiting for certificate to be issued (may take 1-5 minutes)..."
for i in $(seq 1 60); do
  STATUS=$(az keyvault certificate pending show \
    --vault-name "$VAULT_NAME" --name "$CERT_NAME" \
    --query status -o tsv 2>/dev/null || echo "unknown")

  if [ "$STATUS" = "completed" ]; then
    echo "    Certificate issued successfully!"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "ERROR: Certificate issuance failed!"
    az keyvault certificate pending show \
      --vault-name "$VAULT_NAME" --name "$CERT_NAME" \
      --query error -o json 2>/dev/null
    echo ""
    echo "Make sure the domain is registered at https://aka.ms/onecert"
    exit 1
  fi

  if [ "$i" -eq 60 ]; then
    echo "WARNING: Still waiting after 5 minutes. Check with:"
    echo "  az keyvault certificate pending show --vault-name $VAULT_NAME --name $CERT_NAME"
    exit 1
  fi
  printf "    Status: %s (attempt %d/60)\r" "$STATUS" "$i"
  sleep 5
done

# -- 3. Download cert and key from Key Vault ---------------------------------
echo ">>> Downloading certificate from Key Vault..."
CERT_PEM=$(az keyvault certificate show \
  --vault-name "$VAULT_NAME" --name "$CERT_NAME" \
  --query cer -o tsv | base64 -d | openssl x509 -inform der -outform pem 2>/dev/null)

SECRET_BUNDLE=$(az keyvault secret show \
  --vault-name "$VAULT_NAME" --name "$CERT_NAME" \
  --query value -o tsv)

# Extract the private key from the PEM bundle
KEY_PEM=$(echo "$SECRET_BUNDLE" | openssl pkey 2>/dev/null)
# Extract the full certificate chain
CHAIN_PEM=$(echo "$SECRET_BUNDLE" | openssl crl2pkcs7 -nocrl -certfile /dev/stdin 2>/dev/null | \
  openssl pkcs7 -print_certs 2>/dev/null || echo "$CERT_PEM")

echo "    Subject: $(echo "$CERT_PEM" | openssl x509 -noout -subject 2>/dev/null)"
echo "    Issuer:  $(echo "$CERT_PEM" | openssl x509 -noout -issuer 2>/dev/null)"

# -- 4. Update the Kubernetes TLS secret ------------------------------------
echo ">>> Updating Kubernetes TLS secret..."
TLS_CRT_B64=$(echo "$CHAIN_PEM" | base64 -w0)
TLS_KEY_B64=$(echo "$KEY_PEM" | base64 -w0)

kubectl create secret tls streamwise-tls-secret \
  --cert=<(echo "$CHAIN_PEM") \
  --key=<(echo "$KEY_PEM") \
  --namespace "$K8S_NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

# -- 5. Restart pods to pick up the new certificate -------------------------
echo ">>> Restarting pods..."
kubectl delete pod streamwise -n "$K8S_NAMESPACE" --ignore-not-found
kubectl delete pod streamcast -n "$K8S_NAMESPACE" --ignore-not-found

echo ""
echo "==========================================="
echo "OneCert TLS certificate deployed!"
echo ""
echo "Verify with:"
echo "  curl https://$FQDN:8081/health"
echo "  openssl s_client -connect $FQDN:8081 </dev/null 2>&1 | grep -E 'subject|issuer'"
echo ""
echo "No -k flag needed - the certificate is publicly trusted."
echo "Auto-renewal is handled by Key Vault (30 days before expiry)."
echo "==========================================="