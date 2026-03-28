# HTTPS / TLS Certificates

The Bicep template automatically provisions an Azure Key Vault, generates a self-signed TLS certificate inside it, and configures the Secrets Store CSI Driver addon to authenticate via workload identity.
This includes:
- Key Vault + self-signed certificate
- OIDC issuer and workload identity enabled on the cluster
- Key Vault RBAC for the CSI addon identity (Secrets User + Certificate User)
- Federated identity credentials on the CSI addon identity for each pod service account, so the CSI Driver can exchange projected pod tokens for Azure AD tokens

The Secrets Store CSI Driver mounts the certificate from Key Vault directly into each pod at `/certs/`, where the entrypoint scripts auto-detect it and enable HTTPS.

---

## Step 1 — Wait for the certificate to finish generating

Key Vault certificate generation is asynchronous.
Before deploying pods, confirm the certificate is ready:

```bash
az keyvault certificate show \
  --vault-name $KEY_VAULT_NAME --name $TLS_CERT_NAME \
  --query attributes.enabled -o tsv
# must return "true" before continuing
```

## Step 2 — Deploy the SecretProviderClass

The `SecretProviderClass` tells the CSI driver which Key Vault and certificate to use.
It also creates a K8s TLS Secret (`streamwise-tls-secret`) that is automatically rotated whenever the certificate is renewed in Key Vault.

```bash
# KEY_VAULT_NAME, CSI_ADDON_CLIENT_ID, AZ_TENANT_ID, and TLS_CERT_NAME were
# captured from the Bicep deployment outputs (see the output capture commands
# after the az deployment group create command in the AKS README).
envsubst < deployment/k8s/tls-secret-provider.yaml | kubectl apply -f -
```

Verify the SecretProviderClass was created:
```bash
kubectl get secretproviderclass -n rtgen
```

## Step 3 — Deploy pods (the certificate mounts automatically)

The pod YAMLs (`streamwise-pod.yaml`, `streamcast-pod.yaml`) already include the CSI volume mount.
No additional changes are needed — the certificate is detected at `/certs/tls.crt` and `/certs/tls.key` and HTTPS is enabled automatically.

## Using a custom or CA-signed certificate

To replace the auto-generated self-signed certificate with your own:
```bash
# Import a certificate from a PEM file
az keyvault certificate import \
  --vault-name $KEY_VAULT_NAME \
  --name $TLS_CERT_NAME \
  --file /path/to/fullchain.pem

# Or import a PKCS12 file
az keyvault certificate import \
  --vault-name $KEY_VAULT_NAME \
  --name $TLS_CERT_NAME \
  --file /path/to/certificate.pfx \
  --password "$PFX_PASSWORD"
```

After import, the CSI driver picks up the new certificate within the rotation poll interval (2 minutes by default) and restarts affected pods automatically.

## Verify HTTPS is working

```bash
# StreamWise (self-signed cert: -k skips verification)
curl -k https://$IP_ADDRESS:8081/health

# StreamCast
curl -k https://$IP_ADDRESS:8080/health
```

> **Note:** Browsers will show a security warning for the self-signed certificate.
> Import the certificate into your browser's trust store, or replace it with a CA-signed certificate to eliminate warnings.

## Embedding a certificate in the Docker image (alternative to Key Vault)

As an alternative to Key Vault, you can bake a certificate into the image at build time using `--certfile`/`--keyfile`:

```bash
# Generate a self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=streamwise"

# Build the StreamWise image with the certificate embedded
cd deployment/streamwise
bash setup_image.sh --push --certfile /path/to/cert.pem --keyfile /path/to/key.pem
```

The entrypoint script auto-detects `/certs/cert.pem` and `/certs/key.pem` and enables HTTPS automatically — no extra flags or pod YAML changes are needed.
