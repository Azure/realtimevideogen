# HTTPS / TLS Certificates

The Bicep template automatically provisions an Azure Key Vault with a **fallback self-signed TLS certificate** and configures the Secrets Store CSI Driver addon to authenticate via workload identity (when `enableSecureSetup=true`).
For production use the recommended approach is to replace the self-signed certificate with a **CA-signed certificate issued by Let's Encrypt via cert-manager**, which eliminates browser security warnings entirely.

## Automated setup (recommended)

The script [`deployment/aks/setup-letsencrypt.sh`](../aks/setup-letsencrypt.sh) automates the
entire process below (Steps 1–6) in a single command:

```bash
export LETSENCRYPT_EMAIL=your@email.com
export PUBLIC_FQDN        # from Bicep outputs or az network public-ip show
export LOAD_BALANCER_IP    # = $IP_ADDRESS
export RESOURCE_GROUP_NAME # = $AZ_RESOURCE_GROUP

bash deployment/aks/setup-letsencrypt.sh
```

Alternatively, set `LETSENCRYPT_EMAIL` in `quick-deploy.sh` and the script runs automatically
as part of the end-to-end deployment.

If you need finer control or want to troubleshoot, follow the manual steps below.

---

## Recommended: CA-signed certificate with cert-manager + Let's Encrypt

cert-manager is the standard Kubernetes certificate management tool.
It automatically requests, validates, and renews TLS certificates from Let's Encrypt (a free, publicly-trusted CA) using the ACME HTTP-01 challenge over port 80.

The Bicep template already handles the two prerequisites:

* **DNS label** – The public IP is provisioned with a DNS label so it has a stable FQDN
  (`<dnsLabelPrefix>.<region>.cloudapp.azure.com`) that Let's Encrypt can validate.
* **NSG rule** – Port 80 is open on the node subnet NSG so the ACME solver can receive
  validation requests from the Internet.

### Step 1 — Retrieve the public FQDN

```bash
PUBLIC_FQDN=$(az deployment group show \
  --name AKSDeployment \
  --resource-group $AZ_RESOURCE_GROUP \
  --query properties.outputs.publicFqdn.value -o tsv)

echo "Public FQDN: $PUBLIC_FQDN"
# e.g. streamwise-abc12345.eastus.cloudapp.azure.com
```

> If `PUBLIC_FQDN` is empty you can also read it directly from the public IP resource:
> ```bash
> PUBLIC_FQDN=$(az network public-ip show \
>   -g $AZ_RESOURCE_GROUP --name aks-pods-public-ip \
>   --query dnsSettings.fqdn -o tsv)
> ```

### Step 2 — Install cert-manager

Install cert-manager into the cluster (only needed once per cluster):

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

# Wait for cert-manager pods to be ready
kubectl wait --namespace cert-manager \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/instance=cert-manager \
  --timeout=120s
```

### Step 2.5 — Install nginx-ingress controller

cert-manager's HTTP-01 solver requires an Ingress controller to serve ACME
challenges on port 80.  Install the nginx-ingress controller bound to the
cluster's static public IP:

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.service.loadBalancerIP=$IP_ADDRESS \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-resource-group"=$AZ_RESOURCE_GROUP \
  --set controller.service.externalTrafficPolicy=Local

# Wait for the ingress controller to be ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

### Step 3 — Create the ClusterIssuers

```bash
export LETSENCRYPT_EMAIL=your@email.com   # REQUIRED: replace with a real, monitored address.
                                           # Let's Encrypt sends expiry notifications and uses
                                           # this for account recovery — use a shared team mailbox.

envsubst < deployment/k8s/cert-manager-issuer.yaml | kubectl apply -f -
```

Verify both issuers are ready:

```bash
kubectl get clusterissuer
# NAME                  READY   AGE
# letsencrypt-staging   True    ...
# letsencrypt-prod      True    ...
```

### Step 4 — Request the certificate

```bash
envsubst < deployment/k8s/streamwise-certificate.yaml | kubectl apply -f -
```

Monitor the certificate issuance (usually takes 1–2 minutes):

```bash
kubectl get certificate -n rtgen
# NAME             READY   SECRET                  AGE
# streamwise-tls   True    streamwise-tls-secret   90s

kubectl describe certificate streamwise-tls -n rtgen
```

Once `READY` is `True`, the Kubernetes TLS secret `streamwise-tls-secret` has been
created and will be mounted into the pods automatically.

> **Staging first:** To avoid hitting Let's Encrypt rate limits during initial setup, edit
> `deployment/k8s/streamwise-certificate.yaml` and set `issuerRef.name: letsencrypt-staging`
> before applying.  Staging certificates are issued by a test CA and are **not** trusted
> by browsers, but the issuance flow is identical.  Once everything works, switch back to
> `letsencrypt-prod` and re-apply.

### Step 5 — Deploy pods (certificate mounts automatically)

```bash
source deployment/set_properties.sh
export LOAD_BALANCER_IP=$IP_ADDRESS
export RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP

kubectl apply -f deployment/k8s/streamwise-service-account.yaml
envsubst < deployment/aks/streamwise-pod.yaml | kubectl apply -f -

kubectl apply -f deployment/k8s/streamwiseapp-service-account.yaml
envsubst < deployment/aks/streamcast-pod.yaml | kubectl apply -f -
```

The pod YAMLs mount `streamwise-tls-secret` at `/certs/` and `run_httpserver.bash`
auto-detects `/certs/tls.crt` / `/certs/tls.key` to enable HTTPS.

### Step 6 — Verify CA-signed HTTPS

```bash
# No -k flag needed — the certificate is now trusted by your OS/browser
curl https://$PUBLIC_FQDN:8081/health
curl https://$PUBLIC_FQDN:8080/health
```

Open the web UIs at `https://$PUBLIC_FQDN:8081` and `https://$PUBLIC_FQDN:8080` — the
browser padlock should be green with no security warnings.

cert-manager automatically renews the certificate 30 days before it expires (Let's Encrypt
certs are valid for 90 days).

---

## Alternative: manual CA-signed certificate import into Key Vault

If you already have a CA-signed certificate (from your organization's PKI, DigiCert,
Sectigo, etc.) you can import it directly into the Azure Key Vault and then sync it to
a Kubernetes secret.

### Import the certificate into Key Vault

```bash
# PEM format (preferred — run_httpserver.bash expects PEM)
az keyvault certificate import \
  --vault-name $KEY_VAULT_NAME \
  --name $TLS_CERT_NAME \
  --file /path/to/fullchain.pem

# Or PKCS12 format
az keyvault certificate import \
  --vault-name $KEY_VAULT_NAME \
  --name $TLS_CERT_NAME \
  --file /path/to/certificate.pfx \
  --password "$PFX_PASSWORD"
```

### Create the Kubernetes TLS secret from Key Vault

Download the certificate and key from Key Vault and create the K8s secret manually:

```bash
# Download the certificate (public part)
az keyvault certificate download \
  --vault-name $KEY_VAULT_NAME --name $TLS_CERT_NAME \
  --encoding PEM -f /tmp/tls.crt

# Download the secret (contains the full PEM bundle with private key)
az keyvault secret download \
  --vault-name $KEY_VAULT_NAME --name $TLS_CERT_NAME \
  --encoding base64 -f /tmp/bundle.pem

# Extract the private key from the PEM bundle
openssl pkey -in /tmp/bundle.pem -out /tmp/tls.key

# Create or update the Kubernetes TLS secret
kubectl create secret tls streamwise-tls-secret \
  --namespace rtgen \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key \
  --dry-run=client -o yaml | kubectl apply -f -

# Clean up temporary files
rm /tmp/tls.crt /tmp/bundle.pem /tmp/tls.key
```

### Deploy pods

Deploy the pods as usual; they mount `streamwise-tls-secret` and HTTPS is enabled automatically.

> **Secret rotation:** When you renew the certificate in Key Vault and re-run the commands
> above to update the K8s secret, restart the pods to pick up the new certificate:
> ```bash
> kubectl rollout restart pod/streamwise pod/streamcast -n rtgen
> ```

---

## Fallback: self-signed certificate (development / testing only)

The Bicep template includes a self-signed certificate in Key Vault that can be used for
local development or quick testing.  **Browsers will show a security warning** — this is
expected for self-signed certificates.  Do not use this in production.

To use the self-signed certificate directly:

```bash
# Download the self-signed cert and create the K8s secret
az keyvault certificate download \
  --vault-name $KEY_VAULT_NAME --name $TLS_CERT_NAME \
  --encoding PEM -f /tmp/tls.crt

az keyvault secret download \
  --vault-name $KEY_VAULT_NAME --name $TLS_CERT_NAME \
  --encoding base64 -f /tmp/bundle.pem

openssl pkey -in /tmp/bundle.pem -out /tmp/tls.key

kubectl create secret tls streamwise-tls-secret \
  --namespace rtgen \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key \
  --dry-run=client -o yaml | kubectl apply -f -

rm /tmp/tls.crt /tmp/bundle.pem /tmp/tls.key
```

Verify HTTPS (with `-k` to skip certificate verification):

```bash
curl -k https://$IP_ADDRESS:8081/health
curl -k https://$IP_ADDRESS:8080/health
```

To generate a self-signed certificate entirely with OpenSSL (e.g. for build-time embedding):

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=streamwise"
```

---

## Embedding a certificate in the Docker image (non-Kubernetes deployments)

As an alternative to Kubernetes secrets, you can bake a certificate into the image at
build time using `--certfile`/`--keyfile`:

```bash
cd deployment/streamwise
bash setup_image.sh --push --certfile /path/to/cert.pem --keyfile /path/to/key.pem
```

The entrypoint script auto-detects `/certs/cert.pem` and `/certs/key.pem` and enables
HTTPS automatically — no extra flags or pod YAML changes are needed.

