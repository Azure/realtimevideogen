# VM Deployment with Docker

This guide provides detailed instructions for setting up and deploying StreamWise on Ubuntu VMs with Docker. This deployment option is suitable for development, testing, single-node deployments, or when you need to run containers directly without Kubernetes orchestration.

## When to Use VM with Docker

- Development and testing environments
- Single-node deployments
- Learning and experimentation
- Debugging specific containers

## Prerequisites

Before starting, ensure you have:
- **Ubuntu VMs**: VMs with GPU support and NVIDIA drivers installed
- **Docker**: Docker with NVIDIA container runtime
- **Azure CLI**: For ACR access ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
- **Hugging Face Account**: Token for accessing gated models ([Get Token](https://huggingface.co/settings/tokens))
- **Azure Container Registry (ACR)**: Created and accessible ([ACR Setup](../acr/README.md))

## Setup Disks

We can use the NVMe drives in the GPU VMs to store docker images:

```bash
apt update
apt install mdadm

mdadm --create --verbose /dev/md0 --level=0 --raid-devices=8 /dev/nvme[0-7]n1
mkfs.ext4 /dev/md0
mkdir -p /mnt/raid0
mount /dev/md0 /mnt/raid0

mkdir -p /mnt/raid0/docker/
mkdir -p /mnt/raid0/containerd/

ln -s /mnt/raid0/docker /var/lib/docker
ln -s /mnt/raid0/containerd /var/lib/containerd
```

## Docker Installation

Configure docker in `/etc/docker/daemon.json`:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

To install docker:
```bash
mkdir -p /etc/apt/keyrings/
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
apt-get install -y nvidia-docker2 nvidia-container-runtime nvidia-container-toolkit

systemctl restart docker
```

We may need to allow other users:
```bash
sudo chmod 777 /var/run/docker.sock
```

This might be done already but we can move `/var/lib/docker` to `/mnt/raid0/docker`:
```bash
mv /var/lib/docker /mnt/raid0/
ln -s /mnt/raid0/docker /var/lib/docker
```

## Azure Container Registry (ACR) Setup

### Login to ACR
```bash
source ../set_properties.sh  # sets AZ_SUBSCRIPTION_ID, ACR_NAME, etc.

az login
az account set --subscription $AZ_SUBSCRIPTION_ID
az acr login --name $ACR_NAME
```

### Enable Admin Access and Get Credentials
```bash
az acr update -n $ACR_NAME --admin-enabled true
az acr credential show --name $ACR_NAME
export ACR_PASSWORD=`az acr credential show --name $ACR_NAME | jq -r .passwords[0].value`
```

**Verify ACR access:**
```bash
az acr show --name $ACR_NAME
```

Expected output should show ACR details including `adminUserEnabled: true`.

### Configure ACR Variables
All ACR variables (`ACR_FULL_NAME`, `ACR_URL`, `ACR_USERNAME`, `ACR_EMAIL`) are defined in [`set_properties.sh`](../set_properties.sh).
Make sure you have sourced it before proceeding:
```bash
source ../set_properties.sh
```

### Push Images to ACR

Example of pushing an image to ACR:
```bash
docker tag wan "$ACR_URL/wan"
docker push "$ACR_URL/wan"
docker images
```

**Verify image push:**
```bash
az acr repository list --name $ACR_NAME --output table
```

## Running Containers Manually

Once Docker and ACR are configured, you can pull and run individual containers:

```bash
# Pull an image from ACR
docker pull $ACR_URL/your-image:tag

# Run a container with GPU support
docker run --gpus all -d \
  --name your-container \
  -p 8080:8080 \
  -e HF_TOKEN=your_huggingface_token \
  $ACR_URL/your-image:tag
```

**Verify container is running:**
```bash
docker ps
docker logs your-container
```

## Troubleshooting

### Docker Issues
- **Permission denied**: Run `sudo chmod 777 /var/run/docker.sock`
- **GPU not available**: Verify NVIDIA drivers and nvidia-docker2 are installed
- **Storage issues**: Check disk space with `df -h` and ensure NVMe setup is correct

### ACR Issues
- **Login failures**: Verify Azure CLI is logged in with `az account show`
- **Image pull errors**: Check ACR credentials and ensure admin access is enabled
- **Network issues**: Verify VM has internet access to reach ACR endpoints

## Next Steps

For orchestrated deployments with multiple containers, consider:
- **[Manual Kubernetes Setup](../README.md#option-2-manual-kubernetes-cluster-setup)** - For on-premises or custom deployments
- **[AKS Deployment](../aks/README.md)** - For production-ready, managed Kubernetes
