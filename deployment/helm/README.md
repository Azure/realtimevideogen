# Helm
To deploy the services to [kubernetes](https://kubernetes.io/) we use [Helm](https://helm.sh/).

## Installation
For Helm installation instructions, see [Deployment README](../README.md#helm).

Check GPUs:
```bash
kubectl get nodes -o jsonpath="{range .items[*]}{.metadata.name}{':\t'}{.status.allocatable}{'\n'}{end}"
```

## Deployment

### Prerequisites
Before deploying, ensure you have completed the following from [Deployment README](../README.md):
- [ACR Setup and Login](../README.md#azure-container-registry-acr)
- [Namespace Setup](../README.md#kubernetes-namespace-setup)
- [Storage Setup](../README.md#storage-setup)
- [Secrets Configuration](../README.md#secrets-configuration) (HF Token and ACR Secret)

### Optional: Manual Image Download
We can download the images manually:
```bash
crictl images
crictl pull --username $ACR_NAME $ACR_URL/fantasytalking:latest
```

Deploy the microservices:
```bash
helm install gpu-services . --namespace rtgen --create-namespace
```

```

**Verify deployment:**
```bash
helm list -n rtgen
```

Expected output should show `gpu-services` with status `deployed`.

List the microservices:
```bash
helm list -A
```

Check pods:
```bash
kubectl get pods -n rtgen -o wide
```

Check containers in a node:
```bash
crictl ps -a
```

Checking the status of one pod:
```bash
kubectl describe pod <POD_NAME> -n rtgen
kubectl logs <POD_NAME> -n rtgen

kubectl exec -it <POD_NAME> -n rtgen -- cat /tmp/rtgen.log
```

Run inside of the pod:
```bash
kubectl exec -it <POD_NAME> -n rtgen -- /bin/bash
```

Get addresses:
```bash
kubectl get svc -n rtgen
```

Forward address:
```bash
kubectl port-forward -n rtgen svc/<SVC_NAME> 8080:8080
```

From the containers, one can access them with their full name:
```bash
curl -s http://gemma.rtgen.svc.cluster.local:8000/
curl -s http://podcasttranscript.rtgen.svc.cluster.local:8080/health
curl -s http://<SVC_NAME>.rtgen.svc.cluster.local:8080/health
```

Remove the microservices:
```bash
helm uninstall gpu-services -n rtgen
```

Force delete:
```bash
kubectl get pods -n rtgen | awk '/Terminating/ {print $1}' |  xargs -I {} kubectl delete pod {} -n rtgen --grace-period=0 --force
```

