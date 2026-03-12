"""
Module to manage Kubernetes nodes.
"""

import sys
import logging
import traceback

from http import HTTPStatus

from quart import jsonify
from quart import abort
from quart import render_template

from typing import Optional

from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client import CoreV1Api
from kubernetes_asyncio.client import V1DeleteOptions

sys.path.append("..")
from quart_utils import QuartReturn

from k8s_utils import load_k8s_config
from k8s_utils import get_k8s_nodes
from k8s_utils import get_k8s_pods


async def remove_node(
    node_name: str,
    k8s_cluster: Optional[str] = None
) -> QuartReturn:
    """API interface to remove a node by name."""
    if not node_name:
        return jsonify({"error": "Node name is required"}), HTTPStatus.BAD_REQUEST

    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        try:
            await k8s_api.delete_node(name=node_name, body=V1DeleteOptions())
            pods = await k8s_api.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}")
            for pod in pods.items:
                await k8s_api.delete_namespaced_pod(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace)
            return jsonify({"message": f"Node {node_name} removed successfully"}), HTTPStatus.OK
        except Exception as ex:
            logging.error(f"Error removing node {node_name}: {ex}.")
            return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


async def node_info(
    node_name: str,
    k8s_cluster: Optional[str] = None
) -> QuartReturn:
    """Display information about a specific node."""
    if not node_name:
        return jsonify({"error": "Node name is required"}), HTTPStatus.BAD_REQUEST
    try:
        nodes = await get_k8s_nodes(k8s_cluster)
        if not nodes:
            return jsonify({"error": f"Node '{node_name}' not found"}), HTTPStatus.NOT_FOUND
        for node in nodes:
            if node["node_name"] == node_name:
                pods = await get_k8s_pods(k8s_cluster)
                node["pods"] = [pod for pod in pods if pod["node"] == node_name]
                return await render_template(
                    "node.html",
                    nodes=[node])
    except Exception as ex:
        logging.error(f"Error fetching node info for {node_name}: {ex}.")
        abort(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            description=f"Error fetching node info for {node_name}: {ex}")
    return jsonify({"error": f"Node '{node_name}' not found"}), HTTPStatus.NOT_FOUND


async def nodes_info(
    k8s_cluster: Optional[str] = None
) -> QuartReturn:
    """Display information about all nodes."""
    try:
        nodes = await get_k8s_nodes(k8s_cluster)
        if not nodes:
            return jsonify({"error": "No nodes found"}), HTTPStatus.NOT_FOUND

        pods = await get_k8s_pods(k8s_cluster)
        for node in nodes:
            node_name = node["node_name"]
            node["pods"] = [pod for pod in pods if pod["node"] == node_name]
        return await render_template(
            "node.html",
            nodes=nodes)
    except Exception as ex:
        logging.error(f"Error fetching nodes info [{type(ex)}]: {ex}.")
        return jsonify({
            "error": str(ex),
            "type": str(type(ex)),
            "trace": traceback.format_exc(),
        }), HTTPStatus.INTERNAL_SERVER_ERROR
