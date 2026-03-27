"""
API interface to add a pod for a specified container in the Kubernetes cluster.
"""

import os
import sys
import logging
import random
import string
import secrets

from http import HTTPStatus

from quart import jsonify

from typing import Optional
from typing import List
from typing import Tuple

from kubernetes_asyncio.client import CoreV1Api
from kubernetes_asyncio.client import ApiClient

from kubernetes_asyncio.client import V1Affinity
from kubernetes_asyncio.client import V1Toleration
from kubernetes_asyncio.client import V1Container
from kubernetes_asyncio.client import V1ContainerPort
from kubernetes_asyncio.client import V1Pod
from kubernetes_asyncio.client import V1PodSpec
from kubernetes_asyncio.client import V1ResourceRequirements
from kubernetes_asyncio.client import V1LocalObjectReference
from kubernetes_asyncio.client import V1VolumeMount
from kubernetes_asyncio.client import V1Volume
from kubernetes_asyncio.client import V1EnvVarSource
from kubernetes_asyncio.client import V1EnvVar
from kubernetes_asyncio.client import V1SecretKeySelector
from kubernetes_asyncio.client import V1CSIVolumeSource
from kubernetes_asyncio.client import V1EmptyDirVolumeSource
from kubernetes_asyncio.client import V1ObjectMeta
from kubernetes_asyncio.client import V1DeleteOptions

from kubernetes_asyncio.client import V1NodeAffinity
from kubernetes_asyncio.client import V1NodeSelectorRequirement
from kubernetes_asyncio.client import V1NodeSelectorTerm
from kubernetes_asyncio.client import V1NodeSelector

from kubernetes_asyncio.client import V1Service
from kubernetes_asyncio.client import V1ServiceSpec
from kubernetes_asyncio.client import V1ServicePort

from kubernetes_asyncio.client.exceptions import ApiException

from service_account_manager import get_streamwiseapp_service_account
from service_account_manager import get_streamwise_service_account

sys.path.append("..")
from quart_utils import QuartReturn
from quart_utils import get_docker_image

from k8s_utils import load_k8s_config

from streamwise_apps import STREAMWISE_APPS
from streamwise_apps import VLLM_SERVICES


def get_tls_cert_settings() -> Tuple[V1VolumeMount, V1Volume]:
    """Get the TLS certificate volume mount and volume using the Secrets Store CSI Driver.

    Fetches the certificate from Azure Key Vault (via SecretProviderClass 'streamwise-tls')
    and mounts it at /certs so the run_httpserver.bash entrypoint auto-detects it for HTTPS.
    Requires: deployment/k8s/tls-secret-provider.yaml applied first.
    """
    volume_mount = V1VolumeMount(
        name="tls-csi",
        mount_path="/certs",
        read_only=True
    )
    volume = V1Volume(
        name="tls-csi",
        csi=V1CSIVolumeSource(
            driver="secrets-store.csi.k8s.io",
            read_only=True,
            volume_attributes={"secretProviderClass": "streamwise-tls"}
        )
    )
    return volume_mount, volume


def get_vllm_settings() -> Tuple[List[V1VolumeMount], List[V1Volume]]:
    """Get the volume mounts and volumes for VLLM-based containers."""
    volume_mounts = [
        V1VolumeMount(
            mount_path="/root/.cache/huggingface",
            name="hf-cache"
        ),
        V1VolumeMount(
            name="shm",
            mount_path="/dev/shm"
        )
    ]
    volumes = [
        V1Volume(
            name="hf-cache",
            empty_dir=V1EmptyDirVolumeSource()
        ),
        V1Volume(
            name="shm",
            empty_dir=V1EmptyDirVolumeSource(
                medium="Memory",
                size_limit="1Gi")  # Original 64Mi
        )
    ]
    return volume_mounts, volumes


def get_gemma_settings(num_gpus: int) -> Tuple[List[str], List[V1VolumeMount], List[V1Volume]]:
    """vLLM parameters for Gemma."""
    args = [
        "--model", "google/gemma-3-27b-it",
        "--tensor-parallel-size", str(num_gpus),
        "--guided-decoding-backend", "xgrammar"
    ]
    volume_mounts, volumes = get_vllm_settings()
    return args, volume_mounts, volumes


def get_llama32_settings(num_gpus: int) -> Tuple[List[str], List[V1VolumeMount], List[V1Volume]]:
    """vLLM parameters for Llama."""
    args = [
        "--model", "meta-llama/Llama-3.2-90B-Vision",
        "--tensor-parallel-size", str(num_gpus),
        "--guided-decoding-backend", "xgrammar",
        "--enable-prefix-caching",
        "--max-model-len", "8192",
        "--max-num-seq", "128"
    ]
    """
    --enable-auto-tool-choice \
    --tool-call-parser pythonic \
    --chat-template examples/tool_chat_template_llama3.2_pythonic.jinja \
    --trust-remote-code \
    --limit-mm-per-prompt "image=1"
    """
    volume_mounts, volumes = get_vllm_settings()
    return args, volume_mounts, volumes


def get_whisper_settings() -> Tuple[List[str], List[V1VolumeMount], List[V1Volume]]:
    args = [
        "--model", "openai/whisper-large-v3",
    ]
    volume_mounts, volumes = get_vllm_settings()
    return args, volume_mounts, volumes


async def add_service_ip(
    k8s_api: CoreV1Api,
    pod_name: str,
    target_port: int,
    lb_rg: str,
    lb_ip: str,
    lb_port: int,
    protocol: str = "TCP",
    namespace: str = "default",
) -> bool:
    """Add a LoadBalancer service to expose the pod."""
    service_name = f"{pod_name}-svc"

    logging.info(
        f"Creating load balancer for '{service_name}' for pod {pod_name}:{target_port} "
        f"on {lb_ip}:{lb_port} in resource group '{lb_rg}'.")

    service_body = V1Service(
        metadata=V1ObjectMeta(
            name=service_name,
            annotations={
                "service.beta.kubernetes.io/azure-load-balancer-resource-group": lb_rg
            }
        ),
        spec=V1ServiceSpec(
            type="LoadBalancer",
            load_balancer_ip=lb_ip,
            selector={
                "app": pod_name
            },
            ports=[
                V1ServicePort(
                    port=lb_port,
                    target_port=target_port,
                    protocol=protocol,
                )
            ]
        )
    )
    service_response = await k8s_api.create_namespaced_service(
        namespace=namespace,
        body=service_body)
    if service_response is None:
        return False
    return True


def get_gpu_type_affinity(gpu_type: Optional[str]) -> List[str]:
    if gpu_type is None or gpu_type == "N/A" or gpu_type == "Any":
        return []
    if gpu_type == "a+":
        return get_gpu_type_affinity("a100") + get_gpu_type_affinity("h100") + get_gpu_type_affinity("h200")
    if gpu_type == "h+":
        return get_gpu_type_affinity("h100") + get_gpu_type_affinity("h200")
    if gpu_type == "a100":
        return [
            "NVIDIA-A100-SXM4-40GB",
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-A100-PCIe-40GB",
            "NVIDIA-A100-PCIe-80GB",
            "NVIDIA-A100-80GB-PCIe",
        ]
    if gpu_type == "h100":
        return [
            "NVIDIA-H100-SXM5-80GB",
            "NVIDIA-H100-PCIe-80GB",
            "NVIDIA-H100-NVL",
            "NVIDIA-H100-80GB-HBM3",
            "NVIDIA-H100",
        ]
    if gpu_type == "h200":
        return [
            "NVIDIA-H200-SXM5-141GB",
            "NVIDIA-H200"
        ]
    if gpu_type == "gb200":
        return [
            "NVIDIA-GB200-NVL",
            "NVIDIA-GB200-SXM6-192GB",
            "NVIDIA-GB200",
        ]
    if gpu_type == "gb300":
        return [
            "NVIDIA-GB300-NVL",
            "NVIDIA-GB300",
        ]
    if gpu_type == "v100":
        return [
            "Tesla-V100-PCIE-16GB",
            "Tesla-V100-SXM2-16GB",
            "Tesla-V100-SXM2-32GB"
        ]
    return []


# Valid NVIDIA MIG profiles for A100 and H100 GPUs.
# Each profile name maps to the Kubernetes resource suffix (nvidia.com/mig-<profile>).
MIG_PROFILES = {
    # A100 40 GB profiles
    "1g.5gb",
    "2g.10gb",
    "3g.20gb",
    "4g.20gb",
    "7g.40gb",
    # A100 80 GB / H100 80 GB profiles
    "1g.10gb",
    "2g.20gb",
    "3g.40gb",
    "4g.40gb",
    "7g.80gb",
}


def get_mig_resource_name(mig_profile: str) -> Optional[str]:
    """Return the Kubernetes resource name for a given MIG profile, or None if invalid."""
    if mig_profile in MIG_PROFILES:
        return f"nvidia.com/mig-{mig_profile}"
    return None


def get_container_port(container_name: str) -> int:
    """Get the default container port for a given container name, with special cases for certain containers."""
    if container_name in VLLM_SERVICES:
        return 8000
    if container_name == "streamwise":
        return 18181
    if container_name in STREAMWISE_APPS:
        return 18080
    return 8080


def generate_custom_random() -> str:
    """Generate a custom random string similar to Kubernetes random suffix generation.
    Mimics K8s random suffix generation.
    """
    part1 = secrets.token_hex(4)
    part2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{part1}-{part2}"


async def add_pod(
    container_name: Optional[str],
    cpu: int = 2,
    memory_gib: int = 4,
    ephemeral_storage_gib: int = 16,
    gpu: int = 0,
    gpu_type: Optional[str] = None,
    mig_profile: Optional[str] = None,
    tag: Optional[str] = None,
    lb_rg: Optional[str] = None,
    lb_ip: Optional[str] = None,
    lb_port: Optional[int] = None,
    namespace: str = "default",
    k8s_cluster: Optional[str] = None
) -> QuartReturn:
    """
    API interface to add a pod for the specified container.
    If this does not work, try "deployment/helm/deploy.sh" to deploy namespace, etc.

    When *mig_profile* is provided (e.g. ``"1g.5gb"``), the pod requests a MIG slice instead of a whole GPU.
    The *gpu* parameter then represents the number of MIG instances requested (usually 1).
    MIG is only supported on A100 and H100 GPUs.
    Set *gpu_type* accordingly (``"a100"`` or ``"h+"``) so the pod is scheduled on a MIG-capable node.
    """
    if not container_name:
        return jsonify({"error": "Missing required parameter 'container_name'"}), HTTPStatus.BAD_REQUEST

    # When no GPU resources are requested, ignore any provided MIG profile to keep API behavior consistent.
    if not gpu or gpu <= 0:
        mig_profile = None
    elif mig_profile and not get_mig_resource_name(mig_profile):
        return jsonify({"error": f"Invalid MIG profile '{mig_profile}'"}), HTTPStatus.BAD_REQUEST

    logging.info(
        f"Adding pod for '{container_name}' with {cpu} CPU, {memory_gib} GiB memory, "
        f"{ephemeral_storage_gib} GiB storage, {gpu} GPU(s) of type '{gpu_type}'"
        + (f" MIG profile '{mig_profile}'" if mig_profile else "") + ". "
        f"Load balancer RG: '{lb_rg}', IP: '{lb_ip}', port: '{lb_port}'.")

    # Resources
    resource_request = {
        "cpu": cpu,
        "memory": f"{memory_gib}Gi",
    }
    resource_limit = {
        "cpu": cpu * 2,
        "memory": f"{memory_gib * 2}Gi",
    }
    if ephemeral_storage_gib and ephemeral_storage_gib > 0:
        resource_request["ephemeral-storage"] = f"{ephemeral_storage_gib}Gi"
        resource_limit["ephemeral-storage"] = f"{ephemeral_storage_gib * 2}Gi"
    if gpu and gpu > 0:
        if mig_profile:
            # Request a MIG slice: nvidia.com/mig-<profile> (e.g. nvidia.com/mig-1g.5gb)
            mig_resource = get_mig_resource_name(mig_profile)
            assert mig_resource is not None  # already validated above
            resource_request[mig_resource] = gpu
            resource_limit[mig_resource] = gpu
        else:
            resource_request["nvidia.com/gpu"] = gpu
            resource_limit["nvidia.com/gpu"] = gpu

    image_url = await get_docker_image(container_name, tag=tag)
    if image_url is None:
        return jsonify({"error": f"Invalid container '{container_name}'"}), HTTPStatus.BAD_REQUEST

    container_ports = [
        V1ContainerPort(
            container_port=get_container_port(container_name),
            protocol="TCP"
        )
    ]
    # TODO set probes and other container settings

    env_vars = [
        V1EnvVar(
            name="HUGGING_FACE_HUB_TOKEN",
            value_from=V1EnvVarSource(
                secret_key_ref=V1SecretKeySelector(
                    name="hf-token",
                    key="token")))
    ]
    if gpu < 1:
        env_vars.append(V1EnvVar(
            name="NVIDIA_VISIBLE_DEVICES",
            value="none"))
    if container_name in "streamwise" or container_name in STREAMWISE_APPS:
        env_vars.append(V1EnvVar(
            name="LB_RESOURCE_GROUP",
            value=os.getenv("LB_RESOURCE_GROUP", "resource_group")))
        env_vars.append(V1EnvVar(
            name="LB_IP_ADDRESS",
            value=os.getenv("LB_IP_ADDRESS", "1.2.3.4")))

    node_selector = {
        "kubernetes.io/os": "linux",
        "kubernetes.io/arch": "amd64"
    }
    node_affinity = None
    gpu_type_affinity = get_gpu_type_affinity(gpu_type)
    if gpu_type_affinity:
        node_affinity = V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=V1NodeSelector(
                node_selector_terms=[
                    V1NodeSelectorTerm(
                        match_expressions=[
                            V1NodeSelectorRequirement(
                                key="nvidia.com/gpu.product",
                                operator="In",
                                values=gpu_type_affinity
                            ),
                        ]
                    )
                ]
            )
        )

    # AKS adds the taint to Spot VMs, so we need to add a toleration to allow scheduling on Spot nodes
    tolerations = []
    if gpu and gpu > 0:
        tolerations.append(
            V1Toleration(
                key="kubernetes.azure.com/scalesetpriority",
                operator="Equal",
                value="spot",
                effect="NoSchedule"
            )
        )

    # vLLM specific settings
    args = None
    volume_mounts: List[V1VolumeMount] = []
    volumes: List[V1Volume] = []
    if container_name == "gemma":
        if not gpu or gpu <= 0:
            return jsonify({"error": "Gemma requires at least one GPU"}), HTTPStatus.BAD_REQUEST
        args, vllm_mounts, vllm_volumes = get_gemma_settings(gpu)
        volume_mounts.extend(vllm_mounts)
        volumes.extend(vllm_volumes)
    elif container_name == "llama32":
        if not gpu or gpu <= 0:
            return jsonify({"error": "Llama 3.2 requires at least one GPU"}), HTTPStatus.BAD_REQUEST
        args, vllm_mounts, vllm_volumes = get_llama32_settings(gpu)
        volume_mounts.extend(vllm_mounts)
        volumes.extend(vllm_volumes)
    elif container_name == "whisper":
        args, vllm_mounts, vllm_volumes = get_whisper_settings()
        volume_mounts.extend(vllm_mounts)
        volumes.extend(vllm_volumes)

    # Mount TLS certificate from Azure Key Vault (Secrets Store CSI Driver) at /certs so
    # the run_httpserver.bash entrypoint auto-enables HTTPS. Requires the SecretProviderClass
    # from deployment/k8s/tls-secret-provider.yaml to be applied first.
    tls_mount, tls_volume = get_tls_cert_settings()
    volume_mounts.append(tls_mount)
    volumes.append(tls_volume)

    containers = [
        V1Container(
            name=container_name,
            image=image_url,
            args=args,
            env=env_vars,
            ports=container_ports,
            volume_mounts=volume_mounts,
            resources=V1ResourceRequirements(
                requests={k: str(v) for k, v in resource_request.items()},
                limits={k: str(v) for k, v in resource_limit.items()}
            ),
        )
    ]
    image_pull_secrets = [
        # Key for the Azure Container Registry (ACR)
        V1LocalObjectReference(name="acr-secret")
    ]

    random_suffix = generate_custom_random()
    pod_name = f"{container_name}-{random_suffix}"
    pod_labels = {"app": pod_name}

    pod_spec = V1PodSpec(
        containers=containers,
        volumes=volumes,
        image_pull_secrets=image_pull_secrets,
        node_selector=node_selector,
        affinity=V1Affinity(node_affinity=node_affinity),
        tolerations=tolerations
    )
    pod_metadata = V1ObjectMeta(name=pod_name, labels=pod_labels)

    if container_name == "streamwise":
        pod_spec.service_account_name = await get_streamwise_service_account(
            k8s_cluster=k8s_cluster,
            namespace=namespace)
    if container_name in STREAMWISE_APPS:
        pod_spec.service_account_name = await get_streamwiseapp_service_account(
            k8s_cluster=k8s_cluster,
            namespace=namespace)

    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        pod_response = await k8s_api.create_namespaced_pod(
            namespace=namespace,  # This needs to be created using deployment/helm/deploy.sh
            body=V1Pod(
                metadata=pod_metadata,
                spec=pod_spec
            )
        )
        if pod_response is None:
            return jsonify({
                "error": f"Failed to create pod for {container_name}"
            }), HTTPStatus.INTERNAL_SERVER_ERROR

        # Define the services to expose the port
        if lb_rg:
            success = await add_service_ip(
                k8s_api=k8s_api,
                pod_name=pod_name,
                target_port=get_container_port(container_name),
                lb_rg=lb_rg,
                lb_ip=lb_ip or "",
                lb_port=lb_port or 8080,
                namespace=namespace
            )
            if not success:
                return jsonify({
                    "error": f"Failed to create service for {container_name}"
                }), HTTPStatus.INTERNAL_SERVER_ERROR

        return jsonify({
            "message": "Pod creation requested",
            "pod_name": pod_name,
            "container_name": container_name,
            "image_url": image_url,
            "resource_request": resource_request,
            **({"mig_profile": mig_profile} if mig_profile else {}),
        }), HTTPStatus.OK


async def remove_pod(
    pod_name: str,
    namespace: str = "default",
    k8s_cluster: Optional[str] = None
) -> QuartReturn:
    """API interface to remove a pod by name."""
    if not pod_name:
        return jsonify({"error": "Pod name is required"}), HTTPStatus.BAD_REQUEST

    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        try:
            # kubectl delete pod <pod_name> -n rtgen --grace-period=0 --force
            del_pod_response = await k8s_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                grace_period_seconds=0,
                propagation_policy="Foreground",
                body=V1DeleteOptions())
            if del_pod_response is None:
                return jsonify({"error": f"Cannot remove pod {pod_name}"}), HTTPStatus.INTERNAL_SERVER_ERROR
        except ApiException as api_ex:
            if api_ex.status == HTTPStatus.NOT_FOUND:
                logging.error(f"Pod {pod_name} not found for removal.")
                return jsonify({"error": f"Pod {pod_name} not found"}), HTTPStatus.NOT_FOUND
            else:
                logging.error(f"Error removing pod {pod_name}: {api_ex.reason}.")
                return jsonify({"error": api_ex.reason}), api_ex.status
        except Exception as ex:
            logging.error(f"Error removing pod {pod_name} [{type(ex)}]: {ex}.")
            return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR

        # Remove associated public address (if it exists)
        service_name = f"{pod_name}-svc"
        try:
            await k8s_api.delete_namespaced_service(
                name=service_name,
                namespace=namespace)
        except ApiException as api_ex:
            if api_ex.status == HTTPStatus.NOT_FOUND:
                logging.warning(f"Service {service_name} not found for removal.")
            else:
                logging.error(f"Error removing service {service_name}: {api_ex.reason}.")
        except Exception as ex:
            logging.error(f"Error removing service {service_name}: {ex}.")

        return jsonify({"message": f"Pod {pod_name} removed successfully"}), HTTPStatus.OK
