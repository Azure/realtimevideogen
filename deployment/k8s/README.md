# Generic Kubernetes Setup

This guide covers Kubernetes setup steps that apply to **any** cluster — on-premises, cloud-managed, or custom.
For Azure-managed clusters, see the [AKS Deployment Guide](../aks/README.md) which integrates these steps into the AKS workflow.

---

## Kubernetes Installation

Install `kubelet`, `kubeadm`, and `kubectl`:
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

---

## Namespace Setup

Create the namespace for StreamWise:
```bash
source ../set_properties.sh  # provides K8S_NAMESPACE, HF_TOKEN, ACR_URL, etc.

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

---

## Storage Setup

Deploy persistent volumes and claims ([local-pv.yaml](local-pv.yaml), [local-pvc.yaml](local-pvc.yaml)):
```bash
kubectl apply -f k8s/local-pv.yaml
kubectl apply -f k8s/local-pvc.yaml -n $K8S_NAMESPACE
```

**Verify storage setup:**
```bash
kubectl get pv
kubectl get pvc -n $K8S_NAMESPACE
```

Expected output should show persistent volumes with `Available` or `Bound` status and persistent volume claims with `Bound` status.

---

## Secrets Configuration

### Hugging Face Token

Make sure `HF_TOKEN` is set in [`set_properties.sh`](../set_properties.sh), then:
```bash
source ../set_properties.sh

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

### ACR Secret (if not using attached ACR)

> **Note:** If your ACR is attached to your AKS cluster, you can skip this step.

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

---

## Service Accounts

Deploy the service accounts needed by StreamWise and StreamCast:
```bash
kubectl apply -f k8s/streamwise-service-account.yaml
kubectl apply -f k8s/streamwiseapp-service-account.yaml
```

---

## GPU Setup

### NVIDIA Device Plugin

Setup the NVIDIA device plugin for GPU support:
```bash
kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
```

Or apply the local DaemonSet manifest ([nvidia-device-plugin-ds.yaml](nvidia-device-plugin-ds.yaml)):
```bash
kubectl create namespace gpu-resources
kubectl apply -f k8s/nvidia-device-plugin-ds.yaml
```

**Verify GPU device plugin:**
```bash
kubectl get pods -n kube-system | grep nvidia
kubectl get nodes -o jsonpath="{range .items[*]}{.metadata.name}{':\t'}{.status.allocatable}{'\n'}{end}" | grep nvidia
```

Expected output should show nvidia device plugin pods running and nodes showing GPU resources (e.g., `nvidia.com/gpu`).

### Partial GPU Support (MIG)

NVIDIA Multi-Instance GPU (MIG) partitions a single A100 or H100 GPU into smaller isolated slices, each with dedicated memory and compute resources. This lets lightweight models such as **Kokoro** (TTS) and **YOLO** (image detection) share a physical GPU instead of occupying a whole one.

Enable MIG mode on the GPU node (requires a node reboot or driver restart):
```bash
sudo nvidia-smi -i 0 -mig 1

# Verify MIG mode is on
nvidia-smi -L

# Example: partition GPU 0 into 2× 3g.40gb + 1× 1g.10gb on an H100 80 GB
sudo nvidia-smi mig -cgi 3g.40gb,3g.40gb,1g.10gb -C -i 0
```

Then configure the NVIDIA device plugin for MIG by applying the ConfigMap ([nvidia-plugin-mig-config.yaml](nvidia-plugin-mig-config.yaml)):
```bash
kubectl apply -f k8s/nvidia-plugin-mig-config.yaml
# Restart the device plugin daemon set so it picks up the new config
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-resources
```

With `migStrategy = "mixed"`, the device plugin advertises each MIG instance as a separate Kubernetes resource (e.g. `nvidia.com/mig-1g.10gb`).

> **Note:** When specifying a `mig_profile`, the pod requests `nvidia.com/mig-<profile>` instead of `nvidia.com/gpu`.

Common MIG profiles:

| Profile | GPU fraction | Memory |
|---------|-------------|--------|
| `1g.5gb` | 1/7 | 5 GB (A100 40 GB) |
| `1g.10gb` | 1/7 | 10 GB (A100/H100 80 GB) |
| `2g.20gb` | 2/7 | 20 GB (A100/H100 80 GB) |
| `3g.40gb` | 3/7 | 40 GB (A100/H100 80 GB) |
| `7g.80gb` | 7/7 | 80 GB (A100/H100 80 GB — full GPU) |

---

## Helm Installation

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

For more Helm examples and deployment instructions, see [Helm README](../helm/README.md).

---

## Optional Components

### Metrics Server

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

### Kubernetes Dashboard

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

Setup dashboard with helm:
```bash
kubectl delete namespace kubernetes-dashboard
helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard --create-namespace --namespace kubernetes-dashboard

kubectl get pods -n kubernetes-dashboard
kubectl -n kubernetes-dashboard get svc
kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard-kong-proxy 8443:443
```

### NVIDIA Device Plugin (via Helm)

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

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

Check status:
```bash
kubectl --namespace prometheus get pods -l "release=kube-prometheus-stack-1749057185"
```

Get Grafana `admin` user password:
```bash
kubectl --namespace prometheus get secrets kube-prometheus-stack-1749057185-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo
```

NVIDIA DCGM exporter for GPU metrics:
```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install --generate-name gpu-helm-charts/dcgm-exporter
```

Start port forwarding for Grafana:
```bash
export POD_NAME=$(kubectl --namespace prometheus get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=kube-prometheus-stack-1749057185" -oname)
kubectl --namespace prometheus port-forward $POD_NAME 3000
```
