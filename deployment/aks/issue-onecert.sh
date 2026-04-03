#!/usr/bin/env bash
# issue-onecert.sh - Issue a publicly trusted certificate via OneCertV2-PublicCA
#
# Prerequisites: Register the domain at https://aka.ms/onecert first!
#
# Usage:
#   bash deployment/aks/issue-onecert.sh --vault-name <vault-name> --fqdn <fqdn>
#   or set VAULT_NAME and FQDN environment variables before running.
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash deployment/aks/issue-onecert.sh --vault-name <vault-name> --fqdn <fqdn> \\
    [--cert-name <cert-name>] [--issuer-name <issuer-name>]

Required:
  --vault-name <vault-name>   Azure Key Vault name
  --fqdn <fqdn>               Fully qualified domain name for the certificate

Optional:
  --cert-name <cert-name>     Certificate name in Key Vault (default: streamwise-tls)
  --issuer-name <issuer-name> OneCert issuer name (default: Testpublic)
  --help                      Show this help message

Environment variable alternatives:
  VAULT_NAME, FQDN, CERT_NAME, ONECERT_ISSUER, K8S_NAMESPACE
EOF
}

VAULT_NAME="${VAULT_NAME:-}"
FQDN="${FQDN:-}"
CERT_NAME="${CERT_NAME:-streamwise-tls}"
K8S_NAMESPACE="${K8S_NAMESPACE:-rtgen}"
ISSUER_NAME="${ONECERT_ISSUER:-Testpublic}"  # Use "Test" for PrivateCA

while [ "$#" -gt 0 ]; do
  case "$1" in
    --vault-name)
      VAULT_NAME="$2"; shift 2 ;;
    --fqdn)
      FQDN="$2"; shift 2 ;;
    --cert-name)
      CERT_NAME="$2"; shift 2 ;;
    --issuer-name)
      ISSUER_NAME="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$VAULT_NAME" ]; then
  echo "ERROR: VAULT_NAME must be provided via --vault-name or the VAULT_NAME environment variable." >&2
  usage; exit 1
fi
if [ -z "$FQDN" ]; then
  echo "ERROR: FQDN must be provided via --fqdn or the FQDN environment variable." >&2
  usage; exit 1
fi

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

if az keyvault certificate show \
    --vault-name "$VAULT_NAME" --name "$CERT_NAME" \
    --only-show-errors >/dev/null 2>&1; then
  echo ">>> Certificate already exists in Key Vault; skipping creation."
else
  az keyvault certificate create \
    --vault-name "$VAULT_NAME" \
    --name "$CERT_NAME" \
    --policy "@$TMPFILE" \
    --only-show-errors
fi

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