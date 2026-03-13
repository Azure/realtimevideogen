# Deployment
We deploy the images in Docker containers on top of Kubernetes.

Some microservices and containers:
* Text+image to video:
  * [Wan](wrappers/wan)
  * [Hunyuan FramePack](wrappers/hunyuanframepack)
* Text to image:
  * [Flux](wrappers/flux).
* Text to audio:
  * [Kokoro](wrappers/kokoro).
* Text+image+audio to video
  * [Fantasy Talking](wrappers/fantasytalking)
  * [Hunyuan Avatar](wrappers/hunyuanavatar)
* Documents to text (plot)
  * [Podcast Transcript](wrappers/podcasttranscript)
* [LLM](wrappers/gemma):
  * [Gemma](wrappers/gemma)

We also have the [StreamCast](apps/streamcast) that orchestrates the workflow across containers.

---

## Deployment Options Overview

Choose your deployment path based on your requirements:

| Option | Best For | Complexity | Prerequisites |
|--------|----------|------------|---------------|
| **1. Azure Kubernetes Service (AKS)** ✅ | Production deployments, simplest setup | ⭐ Low | Azure subscription, Azure CLI |
| **2. Manual Kubernetes Cluster** | On-premises or custom deployments | ⭐⭐ Medium | Ubuntu VMs with GPU, manual K8s setup |
| **3. VM Deployment with Docker** | Development, testing, single-node setups | ⭐⭐⭐ High | Ubuntu VMs with GPU, Docker |

**Recommended:** Start with **Option 1 (AKS)** for the simplest and fastest deployment experience.

---

## Option 1: Azure Kubernetes Service (AKS) - RECOMMENDED ✅

Azure Kubernetes Service (AKS) is the **preferred deployment method** because:
- **Simplest setup**: AKS handles cluster management, node provisioning, and infrastructure automatically
- **Production-ready**: Built-in monitoring, auto-scaling, and high availability
- **Integrated**: Seamless integration with Azure Container Registry (ACR) and other Azure services

### Prerequisites for AKS

- **Azure Account**: Active Azure subscription with appropriate permissions
- **Azure CLI**: Installed and configured ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **kubectl**: Kubernetes command-line tool ([Install Guide](https://kubernetes.io/docs/tasks/tools/))
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))

### Quick Start for AKS

Follow these steps in order:

1. **[Create ACR](acr/README.md)** (if not already created)
2. **[Login to ACR](#azure-container-registry-acr-setup)**
3. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
4. **[Deploy AKS Cluster](aks/README.md)** - Follow the complete AKS deployment guide
5. **[Configure Namespace](#kubernetes-namespace-setup)**
6. **[Setup Storage](#storage-setup)**
7. **[Configure Secrets](#secrets-configuration)**
8. **[Install Helm](#helm-installation)** (optional if using StreamWise web UI)
9. **Deploy Services** - Choose one of two approaches:
   - **[Deploy with Helm](helm/README.md)** (command-line approach)
   - **[Deploy with StreamWise Web UI](#alternative-deploy-with-streamwise-cluster-manager-web-ui)** (visual interface approach - recommended for beginners)

For detailed AKS-specific instructions, see the **[AKS Deployment Guide](aks/README.md)**.

---

## Option 2: Manual Kubernetes Cluster Setup

Use this option if you need to deploy on your own infrastructure or require custom Kubernetes configurations.

### When to Use Manual Kubernetes

- On-premises deployments
- Custom hardware configurations
- Specific Kubernetes version requirements
- Integration with existing Kubernetes infrastructure

### Prerequisites for Manual Kubernetes

- **Ubuntu VMs**: Multiple VMs with GPU support (at least one master, one or more workers)
- **Azure CLI**: Installed and configured ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **kubectl**: Kubernetes command-line tool ([Install Guide](https://kubernetes.io/docs/tasks/tools/))
- **Docker**: For building and pushing images
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))
- **Azure Container Registry (ACR)**: Created and accessible ([ACR Setup](acr/README.md))

### Manual Kubernetes Setup Steps

Follow the sections below in order:

1. **[Setup Kubernetes](#kubernetes-installation)** - Install and configure K8s cluster
2. **[Configure ACR](#azure-container-registry-acr-setup)**
3. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
4. **[Configure Namespace](#kubernetes-namespace-setup)**
5. **[Setup Storage](#storage-setup)**
6. **[Configure Secrets](#secrets-configuration)**
7. **[Install Helm](#helm-installation)** (optional if using StreamWise web UI)
8. **Deploy Services** - Choose one of two approaches:
   - **[Deploy with Helm](helm/README.md)** (command-line approach)
   - **[Deploy with StreamWise Web UI](#alternative-deploy-with-streamwise-cluster-manager-web-ui)** (visual interface approach)

---

## Option 3: VM Deployment with Docker

Use this option for development, testing, or when you need to run containers directly on VMs without Kubernetes orchestration.

### When to Use VM with Docker

- Development and testing environments
- Single-node deployments
- Learning and experimentation
- Debugging specific containers

### Prerequisites for VM Deployment

- **Ubuntu VMs**: VMs with GPU support and NVIDIA drivers installed
- **Docker**: Docker with NVIDIA container runtime
- **Azure CLI**: For ACR access ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))
- **Azure Container Registry (ACR)**: Created and accessible ([ACR Setup](acr/README.md))

### VM with Docker Setup Steps

For detailed setup instructions, see the **[VM Deployment Guide](vm/README.md)**. The guide covers:

1. **[Setup Disks](vm/README.md#setup-disks)** - Configure NVMe drives for Docker storage
2. **[Install Docker](vm/README.md#docker-installation)** - Install Docker with NVIDIA runtime
3. **[Configure ACR](vm/README.md#azure-container-registry-acr-setup)** - Login and configure ACR access
4. **[Build & Push Docker Images](#building-and-pushing-docker-images)** - Build wrapper and app images, push to ACR
5. **[Run Containers Manually](vm/README.md#running-containers-manually)** - Pull and run individual containers using `docker run`


---

## Common Setup Procedures

The following sections apply to **Options 2 and 3**. If you're using **Option 1 (AKS)**, refer to the [AKS Deployment Guide](aks/README.md) instead, which includes these steps integrated into the AKS workflow.

### VM Deployment Setup

**Applies to:** Option 3 (VM Deployment with Docker)

For detailed instructions on setting up VMs with Docker, including disk configuration, Docker installation, and ACR setup, see the **[VM Deployment Guide](vm/README.md)**.

### Azure Container Registry (ACR) Setup

**Applies to:** Options 2 and 3

For ACR login, configuration, and image management, refer to:
- **Option 3 (VM with Docker)**: See the [VM Deployment Guide](vm/README.md#azure-container-registry-acr-setup)
- **Option 2 (Manual Kubernetes)**: Basic ACR login is covered below in the Kubernetes Installation section


### Kubernetes Installation

**Applies to:** Option 2 (Manual Kubernetes Cluster)

We can orchestrate the containers using Kubernetes.
To install k8s:
```bash
mkdir -p /etc/apt/keyrings/
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.33/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.33/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list

apt-get update
apt-get install -y kubelet kubeadm kubectl

systemctl enable --now kubelet
```

Configure the containerd backend (including GPU) and start:
```bash
containerd config default | tee /etc/containerd/config.toml > /dev/null
nvidia-ctk runtime configure --runtime=containerd
```

Modify `/etc/containerd/config.toml`, to have:
```toml
[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"
  ...

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
```

This might be already correct but we can move `/var/lib/containerd` to `/mnt/raid0/containerd`:
```bash
mv /var/lib/containerd /mnt/raid0/
ln -s /mnt/raid0/containerd /var/lib/containerd
```

Start everything from the master:
```bash
systemctl restart containerd

kubeadm reset -f
kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=10.0.0.5
kubeadm token create --print-join-command
# Copy the KUBEADM_JOIN_TOKEN KUBEADM_JOIN_HASH from here
```

```bash
sudo chmod 777 /run/containerd/containerd.sock
```

From the worker nodes, join the cluster using `kubeadm token create --print-join-command` and get the command:
```bash
kubeadm join 10.0.0.5:6443 --token $KUBEADM_JOIN_TOKEN --discovery-token-ca-cert-hash sha256:$KUBEADM_JOIN_HASH
```

Setup `kubectl` to access the cluster:
```bash
mkdir -p $HOME/.kube
rm -Rf $HOME/.kube/config
Setup `kubectl` to access the cluster:
```bash
mkdir -p $HOME/.kube
rm -Rf $HOME/.kube/config
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config
```

**Verify kubectl access:**
```bash
kubectl cluster-info
kubectl get nodes
```

Expected output should show the cluster endpoint and nodes in `Ready` or `NotReady` state (will be `Ready` after network setup).

Setup network and the fabric:
```bash
modprobe br_netfilter
sysctl net.bridge.bridge-nf-call-iptables=1
sysctl net.ipv4.ip_forward=1
sysctl --system

kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

**Verify network setup:**
```bash
kubectl get pods -n kube-flannel
kubectl get nodes
```

Wait until all flannel pods are `Running` and all nodes show `Ready` status.

Common commands for monitoring:
```bash
kubectl get pods
kubectl get pods -n kube-system
kubectl get pods -A
kubectl get pods -o wide -A  # Check including pod->node mapping 

# Checking nodes
kubectl get nodes

# Checking image pulling
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | grep -i pull
```

Specify the GPU type (replace node name with your actual node):
```bash
kubectl label node vmss-a100000000 gpu-type=a100
```

Setup NVIDIA device plugin for GPU support:
```bash
kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
```

**Verify GPU device plugin:**
```bash
kubectl get pods -n kube-system | grep nvidia
kubectl get nodes -o jsonpath="{range .items[*]}{.metadata.name}{':\t'}{.status.allocatable}{'
'}{end}" | grep nvidia
```

Expected output should show nvidia device plugin pods running and nodes showing GPU resources (e.g., `nvidia.com/gpu`).


#### Kubernetes Namespace Setup

**Applies to:** Options 1, 2 (also covered in AKS guide)

Create the namespace for StreamWise:
```bash
source set_properties.sh  # provides K8S_NAMESPACE, HF_TOKEN, ACR_URL, etc.

if ! kubectl get namespace $K8S_NAMESPACE &> /dev/null; then
    kubectl create namespace $K8S_NAMESPACE
else
    echo "Namespace $K8S_NAMESPACE already exists"
fi
```

**Verify namespace creation:**
```bash
kubectl get namespace $K8S_NAMESPACE
```

Expected output should show the namespace with `Active` status.


#### Storage Setup

**Applies to:** Options 1, 2 (also covered in AKS guide)

**Note:** Ensure you have the required YAML files. For AKS deployments, use files from `aks/` directory. For Helm deployments, use files from `helm/` directory.

Deploy persistent volumes and claims:
```bash
# For AKS deployments:
kubectl apply -f aks/local-pv.yaml -n $K8S_NAMESPACE
kubectl apply -f aks/local-pvc.yaml -n $K8S_NAMESPACE

# OR for Helm deployments:
# kubectl apply -f helm/local-pv.yaml -n $K8S_NAMESPACE
# kubectl apply -f helm/local-pvc.yaml -n $K8S_NAMESPACE
```

**Verify storage setup:**
```bash
kubectl get pv -n $K8S_NAMESPACE
kubectl get pvc -n $K8S_NAMESPACE
```

Expected output should show persistent volumes with `Available` or `Bound` status and persistent volume claims with `Bound` status.


#### Secrets Configuration

**Applies to:** Options 1, 2 (also covered in AKS guide)

##### Hugging Face Token
Make sure `HF_TOKEN` is set in [`set_properties.sh`](set_properties.sh), then:
```bash
source set_properties.sh

# Falls back to prompt if not set
if [ -z "$HF_TOKEN" ]; then
    read -p "Enter your Hugging Face token: " HF_TOKEN
fi

if kubectl get secret hf-token -n $K8S_NAMESPACE &> /dev/null; then
    kubectl delete secret hf-token -n $K8S_NAMESPACE
fi

kubectl create secret generic hf-token -n $K8S_NAMESPACE --from-literal=token=$HF_TOKEN
```

**Verify Hugging Face token secret:**
```bash
kubectl get secret hf-token -n $K8S_NAMESPACE
```

##### ACR Secret (if not using attached ACR)

**Note:** If your ACR is attached to your AKS cluster, you can skip this step.

```bash
if kubectl get secret acr-secret -n $K8S_NAMESPACE &> /dev/null; then
    kubectl delete secret acr-secret -n $K8S_NAMESPACE
fi

kubectl create secret docker-registry acr-secret -n $K8S_NAMESPACE \
  --docker-server=$ACR_URL \
  --docker-username=$ACR_USERNAME \
  --docker-password=$ACR_PASSWORD \
  --docker-email=$ACR_EMAIL
```

**Verify ACR secret:**
```bash
kubectl get secret acr-secret -n $K8S_NAMESPACE
```

Expected output should show both secrets with type `Opaque` (for hf-token) and `kubernetes.io/dockerconfigjson` (for acr-secret).


#### Helm Installation

**Applies to:** Options 1, 2 (also covered in AKS guide)

Install Helm for Kubernetes deployments:
```bash
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

**Verify Helm installation:**
```bash
helm version
```

Expected output should show Helm version information (e.g., `version.BuildInfo{Version:"v3.x.x", ...}`).

For more Helm examples and deployment instructions, see [Helm README](helm/README.md).

#### Alternative: Deploy with StreamWise Cluster Manager Web UI

**Applies to:** Options 1, 2

Instead of using Helm from the command line, you can deploy the **StreamWise Cluster Manager** first and then use its web UI to deploy and manage the rest of the components.

**Benefits of this approach:**
- **Visual interface**: Manage deployments, pods, and services through a web UI
- **Simplified workflow**: Deploy components without memorizing kubectl/helm commands
- **Real-time monitoring**: View cluster status, resource usage, and logs in one place
- **Interactive management**: Start, stop, and configure services dynamically

##### Deploy StreamWise Cluster Manager

**Prerequisites:** Complete namespace setup, storage setup, and secrets configuration from the sections above.

1. **Navigate to the AKS directory** (or appropriate deployment directory):
   ```bash
   cd deployment/aks
   ```

2. **Deploy StreamWise Service Account**:
   ```bash
   kubectl apply -f streamwise-service-account.yaml
   ```

3. **Configure environment variables** (for AKS with Load Balancer):
   ```bash
   # Get your load balancer IP
   export LOAD_BALANCER_IP=$(az network public-ip show --resource-group $AZ_RESOURCE_GROUP --name aks-pods-public-ip | jq -r .ipAddress)
   export RESOURCE_GROUP_NAME=$AZ_RESOURCE_GROUP
   ```

4. **Deploy StreamWise Pod and Service**:
   ```bash
   envsubst < streamwise-pod.yaml | kubectl apply -f -
   envsubst < streamwise-service.yaml | kubectl apply -f -
   ```

5. **Verify deployment**:
   ```bash
   kubectl get pods -n rtgen | grep streamwise
   kubectl get svc -n rtgen | grep streamwise
   ```

6. **Access the Web UI**:
   ```bash
   # Get the service endpoint
   echo "Web UI URL: http://$LOAD_BALANCER_IP:8080"
   
   # Or use port forwarding for local access
   kubectl port-forward -n rtgen svc/streamwise 8080:8080
   ```
   
   Open your browser to `http://localhost:8080` (if using port-forward) or `http://$LOAD_BALANCER_IP:8080`

##### Deploy Components from Web UI

Once the StreamWise web UI is accessible:

1. **Navigate to the Web UI** in your browser
2. **View cluster status**: The dashboard shows all nodes, pods, and services
3. **Deploy services**: Use the web interface to:
   - Submit jobs for model deployments
   - Configure resource allocations (CPU, GPU, memory)
   - Set environment variables and secrets
   - Deploy individual model wrappers (Flux, Wan, Kokoro, etc.)
4. **Monitor deployments**: Track pod status, logs, and resource usage in real-time
5. **Manage services**: Start, stop, restart, or delete services as needed

##### Using GitHub Copilot CLI for Setup

GitHub Copilot CLI can assist you with the StreamWise deployment setup:

**Example prompts:**
```bash
# Get help deploying StreamWise
gh copilot suggest "deploy streamwise cluster manager to kubernetes namespace rtgen"

# Get help with port forwarding
gh copilot suggest "port forward kubernetes service streamwise on port 8080"

# Get help accessing the web UI
gh copilot suggest "get the external IP of kubernetes load balancer service in namespace rtgen"

# Get help troubleshooting
gh copilot explain "kubectl logs -n rtgen streamwise"
```

**To set up GitHub Copilot CLI:**
```bash
# Install GitHub CLI if not already installed
# On Ubuntu/Debian:
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login

# Install Copilot CLI extension
gh extension install github/gh-copilot

# Verify installation
gh copilot --version
```

For detailed StreamWise deployment examples, see the [AKS Deployment Guide](aks/README.md#step-5-deploy-streamwise-application).

---

## Additional Components (Optional)

The following components are optional but recommended for monitoring and management. These apply to **Options 1, 2**.

### Metrics

Install:
```bash
kubectl delete deployment metrics-server -n kube-system
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

Wait for it to be up:
```bash
kubectl get pods -n kube-system -l k8s-app=metrics-server
kubectl logs -n kube-system -l k8s-app=metrics-server
```

Disable the TLS setting and fix the labels:
```bash
kubectl -n kube-system patch deployment metrics-server --type=json -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args/-",
    "value": "--kubelet-insecure-tls"
  },
  {
    "op": "add",
    "path": "/spec/template/metadata/labels/app.kubernetes.io~1instance",
    "value": "metrics-server"
  },
  {
    "op": "add",
    "path": "/spec/template/metadata/labels/app.kubernetes.io~1name",
    "value": "metrics-server"
  }
]'
```

Access metrics:
```bash
kubectl top node
kubectl top pod
kubectl top pod -n rtgen
```

### Dashboard
Dashboard config in `dashboard-admin.yaml`:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kubernetes-dashboard
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kubernetes-dashboard
```

Setup dashboard with old approach:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
kubectl create namespace kubernetes-dashboard
kubectl apply -f dashboard-admin.yaml

kubectl -n kubernetes-dashboard create token admin-user
kubectl taint nodes --all node-role.kubernetes.io/control-plane- --overwrite
kubectl proxy
```

Connect to the dashboard:
```powershell
az network bastion tunnel -n vnet-bastion -g $AZ_RESOURCE_GROUP --target-ip 10.0.0.5 --resource-port 22 --port 2226
ssh -L 8001:localhost:8001 azureuser@localhost -p 2226
```
Available [here](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/).

To deploy as a K8s service:
```bash
kubectl apply -f deployment.yaml
```

We can also use [helm](helm).

Setup dashboard with helm:
```bash
kubectl delete namespace kubernetes-dashboard
helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard --create-namespace --namespace kubernetes-dashboard

kubectl get pods -n kubernetes-dashboard
kubectl describe pod -n kubernetes-dashboard kubernetes-dashboard-api
kubectl describe pod -n kubernetes-dashboard kubernetes-dashboard-kong

kubectl -n kubernetes-dashboard get svc
kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard-kong-proxy 8443:443
```

Then we can open port 8443 through bastion:
```powershell
az network bastion tunnel -n vnet-bastion -g $AZ_RESOURCE_GROUP --target-ip 10.0.0.5 --resource-port 22 --port 2226
```
And then the port:
```bash
ssh -L 8443:localhost:8443 azureuser@localhost -p 2226
```

The dashboard should be accessible [here](https://localhost:8443/#/workloads?namespace=default).


Pull images manually just in case:
```bash
docker pull kong:3.8
ctr images pull docker.io/library/kong:3.8
```

To stop it:
```bash
helm uninstall kubernetes-dashboard -n kubernetes-dashboard
```

### Nvidia Device Plugin
Install:
```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm upgrade -i nvdp nvdp/nvidia-device-plugin --namespace nvidia-device-plugin --create-namespace --version 0.17.1
```

```bash
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
--namespace nvidia-device-plugin \
--version 0.17.1 \
--set nodeSelectorLabeling=true \
--set gfd.enabled=true \
--set-string gfd.attributes="{count,product}" \
--reuse-values
```

Check labels:
```bash
kubectl get nodes --show-labels

```

### Prometheus
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-community/kube-prometheus-stack --create-namespace --namespace prometheus --generate-name --set prometheus.service.type=NodePort --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

kube-prometheus-stack has been installed. Check its status by running:
```bash
  kubectl --namespace prometheus get pods -l "release=kube-prometheus-stack-1749057185"
```

Get Grafana 'admin' user password by running:
```bash
kubectl --namespace prometheus get secrets kube-prometheus-stack-1749057185-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo
```

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install --generate-name gpu-helm-charts/dcgm-exporter
```


Setup the tunnel:
```powershell
az network bastion tunnel -n vnet-bastion -g $AZ_RESOURCE_GROUP --target-ip 10.0.0.5 --resource-port 22 --port 2225
ssh -L 3000:localhost:3000 azureuser@localhost -p 2225
```

Start the port forwarding:
```bash
export POD_NAME=$(kubectl --namespace prometheus get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=kube-prometheus-stack-1749057185" -oname)
kubectl --namespace prometheus port-forward $POD_NAME 3000
```

---

---

## Complete Deployment Example (AKS)

**Applies to:** Option 1 (Azure Kubernetes Service)

For a complete end-to-end AKS deployment workflow with all steps from prerequisites to testing, see the **[AKS Deployment Guide - Complete Example](aks/README.md#complete-end-to-end-deployment-example)**.

---

## Building and Pushing Docker Images

All model wrappers (in [`wrappers/`](wrappers/)) and applications (in [`apps/`](apps/)) follow the same Docker image build pattern. Each component directory contains:

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image definition |
| `setup_image.sh` | Build script (calls the shared [`wrappers/setup_image.sh`](wrappers/setup_image.sh)) |
| `requirements.txt` | Python dependencies (if applicable) |

Image names and tags are defined centrally in [`services.json`](../services.json).

### Prerequisites

```bash
source set_properties.sh       # loads ACR_NAME, ACR_URL, HF_TOKEN, etc.
az acr login --name $ACR_NAME  # authenticate to ACR
```

### Build All Images at Once

The [`build_docker.sh`](build_docker.sh) script reads every service from `services.json`, builds each image via its `setup_image.sh`, and pushes them all to ACR:

```bash
cd deployment
source set_properties.sh
bash build_docker.sh
```

### Build a Single Wrapper Image

Navigate to the wrapper directory and run its `setup_image.sh`:

```bash
cd deployment/wrappers/wan
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR
```

Some wrappers (e.g., `flux`, `gemma`, `llama32`) require a Hugging Face token for gated model access. These wrappers pass `--hf_token` automatically — just make sure `HF_TOKEN` is set in `set_properties.sh`.

### Build a Single App Image

```bash
cd deployment/apps/streamwise
bash setup_image.sh            # build only
bash setup_image.sh --push     # build and push to ACR
```

### Push an Image Manually

If you built an image without `--push`, you can push it separately:

```bash
source set_properties.sh
az acr login --name $ACR_NAME
docker push $ACR_URL/<image-name>:<tag>
```

### Verify Images in ACR

```bash
az acr repository list --name $ACR_NAME -o table
az acr repository show-tags --name $ACR_NAME --repository <image-name> -o table
```

For more details, see:
- [Wrappers build documentation](wrappers/README.md)
- [Apps build documentation](apps/README.md)
- [ACR setup and usage](acr/README.md)
